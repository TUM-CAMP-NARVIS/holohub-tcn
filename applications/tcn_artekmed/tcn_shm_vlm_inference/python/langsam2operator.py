# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
import matplotlib.pyplot as plt
import numpy as np
import torch
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from PIL import Image
from utils import CupyArrayPainter, DecoderInputData, PointMover, save_cupy_tensor

from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor



SAM_MODELS = {
    "sam2.1_hiera_tiny": {
        "url": "file:///srv/models/active/sam2/sam2.1_hiera_tiny.pt",
        "config": "/srv/models/active/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small": {
        "url": "file:///srv/models/active/sam2/sam2.1_hiera_small.pt",
        "config": "/srv/models/active/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus": {
        "url": "file:///srv/models/active/sam2/sam2.1_hiera_base_plus.pt",
        "config": "/srv/models/active/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large": {
        "url": "file:///srv/models/active/sam2/092824/sam2.1_hiera_large.pt",
        "config": "/srv/models/active/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


class SAM:

    def __init__(self, sam_type: str, ckpt_path: str | None = None, device: torch.device | None = None, compile_model: bool = False):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device
        self.compile_model = compile_model
        self.model = None
        self.mask_generator = None
        self.predictor = None
        # Last per-call sub-stage timings (ms), populated when predict_batch(timing=True):
        #   encode = set_image_batch (Hiera image encoder)
        #   decode = predict_batch   (mask decoder + postprocess_masks upsampling)
        self.last_encode_ms = 0.0
        self.last_decode_ms = 0.0

    def _sync(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()


    def build_model(self):
        config_path = SAM_MODELS[self.sam_type]["config"]
        # Load config using OmegaConf directly instead of Hydra compose
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        self.model = instantiate(cfg.model, _recursive_=True)
        self._load_checkpoint(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.predictor = SAM2ImagePredictor(self.model)
        # Optionally torch.compile the image encoder (fixed 1024^2 input -> static shapes,
        # good for compile). Only the encoder is compiled; the decoder runs over a variable
        # number of boxes, which would trigger recompilation. Falls back to eager on error.
        if self.compile_model:
            try:
                self.model.image_encoder = torch.compile(self.model.image_encoder)
                print("SAM2 image encoder compiled (torch.compile)")
            except Exception as e:
                print(f"torch.compile(SAM image_encoder) failed; using eager. {e}")

    def _load_checkpoint(self, model: torch.nn.Module):
        if self.ckpt_path is None:
            checkpoint_url = SAM_MODELS[self.sam_type]["url"]
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
        else:
            checkpoint_url = self.ckpt_path  # Ensure checkpoint_url is defined
            state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)["model"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(
                f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                re-downloading it. Error: {e}"
            )

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Output format
        SAM2AutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information
        about the mask:

        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        """

        sam2_result = self.mask_generator.generate(image_rgb)
        return sam2_result

    def _autocast(self):
        # bf16 autocast on the SAM2 image encoder + decoder: ~1.5-2x with no visible
        # quality change (matches the sibling applications/sam2 operator). SAM2's
        # set_image/_predict are already @torch.no_grad, so no_grad is not needed here.
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=(self.device is not None and self.device.type == "cuda"),
        )

    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._autocast():
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(box=xyxy, multimask_output=False)
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
        timing: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        with self._autocast():
            if timing:
                self._sync(); _t0 = time.perf_counter()
            self.predictor.set_image_batch(images_rgb)
            if timing:
                self._sync(); self.last_encode_ms = (time.perf_counter() - _t0) * 1000.0
                _t1 = time.perf_counter()
            masks, scores, logits = self.predictor.predict_batch(box_batch=xyxy, multimask_output=False)
            if timing:
                self._sync(); self.last_decode_ms = (time.perf_counter() - _t1) * 1000.0

        masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
        scores = [np.squeeze(score) for score in scores]
        logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
        return masks, scores, logits

    def predict_batch_gpu(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
        timing: bool = False,
    ) -> tuple[list, list, None]:
        """Same as predict_batch but keeps masks ON THE GPU.

        SAM2's predict_batch does `masks.float().detach().cpu().numpy()` -- shipping N
        full-resolution FLOAT masks to the host every frame, which we then push straight
        back to the GPU for compositing. This replicates SAM2's per-image loop (using its
        _prep_prompts/_predict) but returns cupy uint8 (N, H, W) masks + cupy scores, so the
        masks never leave the device. The `decode` timer here therefore excludes the host
        transfer -- comparing it to predict_batch's tells us if the transfer was the cost.
        """
        p = self.predictor
        with self._autocast():
            if timing:
                self._sync(); _t0 = time.perf_counter()
            p.set_image_batch(images_rgb)
            if timing:
                self._sync(); self.last_encode_ms = (time.perf_counter() - _t0) * 1000.0
                _t1 = time.perf_counter()
            num_images = len(p._features["image_embed"])
            all_masks, all_scores = [], []
            for img_idx in range(num_images):
                box = xyxy[img_idx] if xyxy is not None else None
                mask_input, unnorm_coords, labels, unnorm_box = p._prep_prompts(
                    None, None, box, None, True, img_idx=img_idx
                )
                masks, iou, _ = p._predict(
                    unnorm_coords, labels, unnorm_box, mask_input,
                    multimask_output=False, return_logits=False, img_idx=img_idx,
                )
                if masks.ndim == 4:
                    masks = masks[:, 0]                      # (num_boxes, H, W), bool
                masks_u8 = masks.to(torch.uint8).contiguous()
                # torch -> cupy (zero-copy view) then .copy() so cupy owns the memory and
                # it survives after the torch tensors are freed.
                all_masks.append(cp.from_dlpack(masks_u8).copy())
                all_scores.append(cp.from_dlpack(iou.reshape(-1).float().contiguous()).copy())
            if timing:
                self._sync(); self.last_decode_ms = (time.perf_counter() - _t1) * 1000.0
        return all_masks, all_scores, None


class GDINO:
    def __init__(self, model_ckpt_path: str | None = None, processor_ckpt_path: str | None = None, device: torch.device | None = None, model_id: str = "IDEA-Research/grounding-dino-base", input_size: int | None = None, compile_model: bool = False):
        self.model_ckpt_path = model_ckpt_path
        self.processor_ckpt_path = processor_ckpt_path
        self.device = device
        self.compile_model = compile_model
        # Hugging Face Hub id used when no local checkpoint paths are given (or as a
        # fallback). Swap to "IDEA-Research/grounding-dino-tiny" for a smaller backbone.
        self.model_id = model_id
        # Optional override of the detector's resize target (shortest edge, px). The HF
        # default is shortest_edge=800 -- which UPSCALES a typical camera frame and makes
        # the deformable-attention encoder (the real GDINO cost) do extra work. Lowering
        # this is the main GDINO speedup lever; large objects tolerate it well.
        self.input_size = input_size
        self.model = None
        self.processor = None

    def _maybe_override_size(self):
        if self.input_size and self.processor is not None:
            try:
                s = int(self.input_size)
                # longest_edge = 2*s so the shortest edge stays the binding constraint for
                # typical (wide) camera aspect ratios.
                self.processor.image_processor.size = {"shortest_edge": s, "longest_edge": 2 * s}
                print(f"GDINO input size set to shortest_edge={s}, longest_edge={2 * s}")
            except Exception as e:
                print(f"Failed to override GDINO input size: {e}")

    def build_model(self):
        if not self.model_ckpt_path or not self.processor_ckpt_path: # indicates that we somehow able to load the model from internet
            model_id = self.model_id
            print(f"One or both local paths not provided. Loading from Hugging Face Hub: {model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        else:
            print(f"Attempting to load processor from local path: {self.processor_ckpt_path}")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.processor_ckpt_path,
                    local_files_only=True,        # never goes online
                    trust_remote_code=True,       # Grounding-DINO uses custom code
                )
            except Exception as e:
                print(f"Failed to load processor from local path: {e}")
                print("Falling back to Hugging Face Hub")
                model_id = self.model_id
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
                self._maybe_override_size()
                self._init_preprocess()
                self._maybe_compile()
                return

            print(f"Attempting to load model from local path: {self.model_ckpt_path}")
            try:
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.model_ckpt_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    use_safetensors=True,
                ).to(self.device)
            except Exception as e:
                print(f"Failed to load model from local path: {e}")
                print("Falling back to Hugging Face Hub")
                model_id = self.model_id
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        self._maybe_override_size()
        self._init_preprocess()
        self._maybe_compile()

    def _maybe_compile(self):
        # Optionally torch.compile the whole detector. Inputs are static (fixed resize +
        # cached text tokens), so no recompilation. Falls back to eager on error.
        if self.compile_model and self.model is not None:
            try:
                self.model = torch.compile(self.model)
                print("Grounding DINO compiled (torch.compile)")
            except Exception as e:
                print(f"torch.compile(GDINO) failed; using eager. {e}")

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        # For Grounding DINO, when processing multiple prompts for a single image,
        # they should be concatenated into one string separated by ". "
        # e.g., ["hand", "tool"] -> "hand. tool."
        if len(images_pil) == 1 and len(texts_prompt) > 1:
            # Multiple prompts for single image - concatenate them
            combined_prompt = ". ".join(texts_prompt)
            if not combined_prompt.endswith("."):
                combined_prompt += "."
            texts_prompt = [combined_prompt]
        else:
            # Single prompt per image or multiple images - ensure each ends with "."
            texts_prompt = [prompt if prompt.endswith(".") else prompt + "." for prompt in texts_prompt]

        inputs = self.processor(
            images=images_pil, text=texts_prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)
        use_amp = self.device is not None and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type if self.device is not None else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
        return results

    # ------------------------------------------------------------------ #
    # GPU-native inference path: no PIL, no CPU image processor, no        #
    # host<->device copies, and text tokenization cached across frames.    #
    # ------------------------------------------------------------------ #
    def _init_preprocess(self):
        """Cache image-preprocessing params from the HF image processor so we can run the
        resize/normalize on GPU, matching the processor exactly."""
        ip = getattr(self.processor, "image_processor", None)
        if ip is None or self.device is None:
            return
        self._img_mean = torch.tensor(ip.image_mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self._img_std = torch.tensor(ip.image_std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self._rescale = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
        size = ip.size if isinstance(ip.size, dict) else {}
        self._shortest_edge = int(size.get("shortest_edge", 800))
        self._longest_edge = int(size.get("longest_edge", 1333))
        self._text_cache_key = None
        self._text_cache = None

    def _resize_hw(self, h, w):
        """Aspect-preserving target (H, W) matching HF get_size_with_aspect_ratio."""
        size = self._shortest_edge
        max_size = self._longest_edge
        if max_size is not None:
            min_o = float(min(h, w))
            max_o = float(max(h, w))
            if max_o / min_o * size > max_size:
                size = int(round(max_size * min_o / max_o))
        if (h <= w and h == size) or (w <= h and w == size):
            return h, w
        if w < h:
            return int(round(size * h / w)), size
        return size, int(round(size * w / h))

    def _preprocess_image_gpu(self, image_gpu):
        """(H, W, C>=3) uint8 GPU tensor -> (1, 3, H', W') normalized float on device."""
        img = image_gpu[..., :3].to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0).to(torch.float32).contiguous()  # (1,3,H,W)
        img = img * self._rescale
        h, w = img.shape[-2], img.shape[-1]
        nh, nw = self._resize_hw(h, w)
        if (nh, nw) != (h, w):
            img = torch.nn.functional.interpolate(
                img, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True
            )
        return (img - self._img_mean) / self._img_std

    def _encode_text(self, texts_prompt):
        """Tokenize the (static) prompt set once and reuse the GPU tensors every frame."""
        if len(texts_prompt) > 1:
            combined = ". ".join(texts_prompt)
            if not combined.endswith("."):
                combined += "."
            prompts = [combined]
        else:
            prompts = [p if p.endswith(".") else p + "." for p in texts_prompt]
        key = tuple(prompts)
        if key != getattr(self, "_text_cache_key", None):
            self._text_cache = self.processor.tokenizer(
                prompts, padding=True, return_tensors="pt"
            ).to(self.device)
            self._text_cache_key = key
        return self._text_cache

    def predict_gpu(self, image_gpu, texts_prompt, box_threshold, text_threshold, orig_hw):
        """Run detection entirely on GPU. image_gpu: (H, W, C) uint8 CUDA tensor.
        orig_hw: (H, W) of the original frame for box rescaling."""
        enc = self._encode_text(texts_prompt)
        pixel_values = self._preprocess_image_gpu(image_gpu)
        use_amp = self.device is not None and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type if self.device is not None else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=enc.input_ids,
                token_type_ids=enc.get("token_type_ids"),
                attention_mask=enc.attention_mask,
            )
        return self.processor.post_process_grounded_object_detection(
            outputs,
            enc.input_ids,
            box_threshold,
            text_threshold=text_threshold,
            target_sizes=[orig_hw],
        )





class TextPromptPublisher(Operator):
    """Operator that publishes text prompts for LangSAM"""

    def __init__(self, *args, prompts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = prompts if prompts else ["object"]

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Create output message with text prompts
        # Use a dict since Entity.add() doesn't support plain Python lists
        op_output.emit({"text_prompts": self.prompts}, "out")


class LangSAM2Operator(Operator):
    """Operator to perform inference using LangSAM (Grounding DINO + SAM2)"""

    def __init__(self, *args, sam_type="sam2.1_hiera_small", sam_ckpt_path: str | None = None, gdino_model_ckpt_path: str | None = None, gdino_processor_ckpt_path: str | None = None, gdino_model_id: str = "IDEA-Research/grounding-dino-base", gdino_input_size: int | None = None, gdino_gpu_preprocess: bool = True, sam_gpu_output: bool = True, gdino_compile: bool = False, sam_compile: bool = False, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), timing=False, timing_log_every=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_type = sam_type
        self.device = device
        # When True, run Grounding DINO fully on GPU (no PIL, no CPU image processor, no
        # host<->device copies for detection, cached text tokenization). False falls back
        # to the original PIL + HF-processor path.
        self.gdino_gpu_preprocess = gdino_gpu_preprocess
        # When True, keep SAM masks on the GPU (avoids SAM2's full-res float mask
        # device->host->device roundtrip). False uses the stock numpy predict_batch.
        self.sam_gpu_output = sam_gpu_output

        # Opt-in per-stage timing to split the per-frame cost into GDINO vs SAM. Off by
        # default (zero overhead). When on, logs a rolling average every timing_log_every
        # frames. GPU work is async, so we cuda.synchronize() around each stage to measure
        # real device time (adds negligible overhead only while timing is enabled).
        self.timing = timing
        self._timing_log_every = max(1, int(timing_log_every))
        self._t_gdino = 0.0
        self._t_sam = 0.0
        self._t_sam_enc = 0.0  # SAM: set_image_batch (encoder)
        self._t_sam_dec = 0.0  # SAM: predict_batch (decoder + postprocess upsampling)
        self._t_frames = 0

        # Initialize SAM model
        self.sam = SAM(sam_type, sam_ckpt_path, device=device, compile_model=sam_compile)
        self.sam.build_model()

        # Initialize Grounding DINO model
        self.gdino = GDINO(model_ckpt_path=gdino_model_ckpt_path, processor_ckpt_path=gdino_processor_ckpt_path, device=device, model_id=gdino_model_id, input_size=gdino_input_size, compile_model=gdino_compile)
        self.gdino.build_model()

    def _sync(self):
        # Force completion of async GPU work so perf_counter measures real device time.
        if self.timing and self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()

    def setup(self, spec: OperatorSpec):
        # input port for the image tensor(s)
        spec.input("image")
        # input port for text prompts (list of strings)
        spec.input("text_prompts")
        # output port for the results (boxes, scores, masks, mask_scores)
        spec.output("out")
        # Parameters for detection thresholds
        spec.param("box_threshold", 0.3)
        spec.param("text_threshold", 0.25)

    def compute(self, op_input, op_output, context):
        # Get the image tensor from the input port
        image_message = op_input.receive("image")
        # Get text prompts from the input port
        text_prompts_message = op_input.receive("text_prompts")

        # Convert image tensor to numpy array
        # Expected format: (H, W, C) from ConvertBgraToRgbaOp
        image_tensor = image_message.get("")
        if image_tensor is None:
            print("Warning: No image tensor received")
            return

        # The frame arrives on the GPU. Grounding DINO now consumes it directly on-device
        # (see gdino_gpu_preprocess). SAM2's public API is numpy-only, so we still copy the
        # frame to host once for it -- that single .get() is the only remaining gpu->host.
        cp_img = cp.asarray(image_tensor)          # (H, W, C) uint8 on GPU
        H0, W0 = int(cp_img.shape[0]), int(cp_img.shape[1])

        image_np = cp_img.get()
        rgb_np = image_np[..., :3] if image_np.shape[-1] == 4 else image_np
        rgb_np = np.ascontiguousarray(rgb_np, dtype=np.uint8)
        rgb_images = [rgb_np]  # stays index-parallel to gdino_results for SAM below

        # Get text prompts (assume it's a list of strings or a single string)
        text_prompts = text_prompts_message.get("text_prompts")
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        texts_prompt = text_prompts if isinstance(text_prompts, list) else [text_prompts]

        # Get threshold parameters
        box_threshold = self.box_threshold
        text_threshold = self.text_threshold

        # Run Grounding DINO to get bounding boxes
        self._sync(); _t0 = time.perf_counter()
        if self.gdino_gpu_preprocess:
            # Zero-copy view of the GPU frame as a torch tensor; no host round-trip.
            img_gpu = torch.from_dlpack(cp_img)
            gdino_results = self.gdino.predict_gpu(
                img_gpu, texts_prompt, box_threshold, text_threshold, (H0, W0)
            )
        else:
            gdino_results = self.gdino.predict(
                [Image.fromarray(rgb_np)], texts_prompt, box_threshold, text_threshold
            )
        self._sync(); _t_gdino = time.perf_counter() - _t0
        _t_sam = 0.0

        # Process results and prepare for SAM
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []

        for idx, result in enumerate(gdino_results):
            # Convert tensors to numpy arrays. Float tensors are cast to float32 first
            # because bf16 (from autocast) has no numpy equivalent.
            converted = {}
            for k, v in result.items():
                if hasattr(v, "numpy"):
                    if v.is_floating_point():
                        v = v.float()
                    converted[k] = v.detach().cpu().numpy()
                else:
                    converted[k] = v
            result = converted
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            # Check if any objects were detected
            if result.get("labels") and len(result["labels"]) > 0:
                sam_images.append(rgb_images[idx])
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)

        # Run SAM2 to generate masks if any boxes were detected
        _t_sam_enc = 0.0
        _t_sam_dec = 0.0
        if sam_images:
            print(f"Predicting {len(sam_boxes)} masks")
            self._sync(); _t1 = time.perf_counter()
            if self.sam_gpu_output:
                masks, mask_scores, _ = self.sam.predict_batch_gpu(sam_images, xyxy=sam_boxes, timing=self.timing)
            else:
                masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes, timing=self.timing)
            self._sync(); _t_sam = time.perf_counter() - _t1
            if self.timing:
                _t_sam_enc = self.sam.last_encode_ms / 1000.0
                _t_sam_dec = self.sam.last_decode_ms / 1000.0
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            print(f"Predicted {len(all_results)} masks")

            # Reset SAM predictor to clear cached embeddings and free GPU memory
            if hasattr(self.sam.predictor, 'reset_predictor'):
                self.sam.predictor.reset_predictor()
            else:
                # Manual reset of cached features
                self.sam.predictor._features = None
                self.sam.predictor._orig_hw = None
                self.sam.predictor._is_image_set = False

        # Per-stage timing: rolling GDINO vs SAM average (with SAM encode/decode split),
        # logged every N frames.
        if self.timing:
            self._t_gdino += _t_gdino
            self._t_sam += _t_sam
            self._t_sam_enc += _t_sam_enc
            self._t_sam_dec += _t_sam_dec
            self._t_frames += 1
            if self._t_frames % self._timing_log_every == 0:
                n = self._timing_log_every
                print(f"[langsam timing] last {n} frames avg: "
                      f"GDINO={1000 * self._t_gdino / n:.1f} ms, "
                      f"SAM={1000 * self._t_sam / n:.1f} ms "
                      f"(encode={1000 * self._t_sam_enc / n:.1f}, "
                      f"decode+post={1000 * self._t_sam_dec / n:.1f})")
                self._t_gdino = 0.0
                self._t_sam = 0.0
                self._t_sam_enc = 0.0
                self._t_sam_dec = 0.0

        # Convert results for output. For the single-image case, use the first result.
        result = all_results[0]

        # Emit a plain dict (not an Entity) so we can carry the per-detection string
        # labels alongside the numeric tensors -- an Entity/tensor can't hold strings,
        # and the labels are what let the postprocessor assign a stable color per class.
        out_message = {}

        for key in ["boxes", "scores"]:
            if key in result and len(result[key]) > 0:
                out_message[key] = cp.asarray(result[key])

        # Add masks and mask_scores if available
        if len(result["masks"]) > 0:
            out_message["masks"] = cp.asarray(result["masks"])
            out_message["mask_scores"] = cp.asarray(result["mask_scores"])

        # Per-detection class labels (strings), aligned with boxes/masks order.
        # transformers >=4.51 renames this to "text_labels"; fall back to "labels".
        raw_labels = result.get("text_labels", result.get("labels", []))
        if hasattr(raw_labels, "tolist"):
            raw_labels = raw_labels.tolist()
        out_message["labels"] = [str(label) for label in raw_labels]

        # Always forward the camera frame size (H, W) so the postprocessor emits the RGBA
        # mask at the camera resolution even on no-detection frames -- keeps the output
        # pixel-1:1 with the color image for direct texture lookup.
        out_message["image_hw"] = (int(rgb_np.shape[0]), int(rgb_np.shape[1]))

        op_output.emit(out_message, "out")

        # NOTE: intentionally NOT calling gc.collect() / torch.cuda.empty_cache() here.
        # Doing so every frame forces a full device sync and frees the CUDA caching
        # allocator, so the next frame re-allocates from scratch -- a large per-frame
        # cost. Inference shapes are stable, so the caching allocator keeps memory bounded
        # on its own. (bf16 autocast also lowers the working-set size.)


class LangSamPostprocessorOp(Operator):
    """Operator to post-process LangSAM inference output for visualization"""

    def __init__(
        self,
        *args,
        save_intermediate=False,
        verbose=False,
        mask_alpha=180,
        prompts=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.counter = 0
        self.painter = CupyArrayPainter()
        self.save_intermediate = save_intermediate
        # Opacity applied to every mask when compositing onto the RGBA output.
        self.mask_alpha = mask_alpha
        # Qualitative palette (tab20 -> 20 distinct colors). A given class always maps to
        # the same palette slot, so colors stay stable across frames (per class, not per
        # detection order).
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3] * 255
        self.palette = cp.asarray(palette, dtype=cp.uint8)  # (20, 3)
        # Persistent class -> palette-index map. Seeded from the configured prompt order so
        # colors are deterministic across runs; labels not in the prompts get the next free
        # slot the first time they are seen and keep it for the rest of the session.
        self._label_to_idx = {}
        for prompt in (prompts or []):
            self._register_label(prompt)

    @staticmethod
    def _normalize_label(label):
        return str(label).strip().lower()

    def _register_label(self, label):
        key = self._normalize_label(label)
        if key and key not in self._label_to_idx:
            self._label_to_idx[key] = len(self._label_to_idx)
        return self._label_to_idx.get(key, 0)

    def _color_index_for_label(self, label):
        """Stable palette index for a class label (exact, then substring, then new slot)."""
        key = self._normalize_label(label)
        if key in self._label_to_idx:
            return self._label_to_idx[key]
        # GDINO may return a partial phrase (e.g. "computer monitor" vs prompt "monitor").
        for known, idx in self._label_to_idx.items():
            if known and (known in key or key in known):
                return idx
        # Unknown label: assign and remember the next free slot.
        return self._register_label(label)

    def _composite_masks(self, masks, labels=None):
        """Composite all detection masks into a single (H, W, 4) RGBA image.

        Each instance is colored by its class label (stable across frames); where masks
        overlap the later (lower-scoring) instance is drawn on top. Background stays
        transparent. `masks` is a cupy array of shape (N, H, W), (N, 1, H, W) or (H, W).
        """
        if masks.ndim == 4:
            # (N, num_masks_per_detection, H, W) -> (N, H, W): SAM ran multimask_output=False
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]

        num_instances, h, w = masks.shape
        rgba = cp.zeros((h, w, 4), dtype=cp.uint8)
        for i in range(num_instances):
            m = masks[i] > 0.5
            if labels is not None and i < len(labels):
                color_idx = self._color_index_for_label(labels[i])
            else:
                color_idx = i  # no label available -> fall back to per-instance color
            color = self.palette[color_idx % self.palette.shape[0]]
            rgba[m, 0:3] = color
            rgba[m, 3] = self.mask_alpha
        if self.verbose:
            print(f"Composited {num_instances} mask(s) into {rgba.shape} RGBA; "
                  f"labels={list(labels) if labels is not None else None}")
        return rgba

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors from LangSAM2Operator (boxes, scores, labels, masks, mask_scores)
        output: "out"  - Visualization-ready RGBA mask tensors
        """
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Extract data from input message
        try:
            masks = cp.asarray(in_message.get("masks"))
            mask_scores = cp.asarray(in_message.get("mask_scores"))
            boxes = cp.asarray(in_message.get("boxes"))
            scores = cp.asarray(in_message.get("scores"))
            # Per-detection class labels (strings), aligned with masks; used for coloring.
            labels = in_message.get("labels") or []
        except Exception as e:
            if self.verbose:
                print(f"Error extracting data from input message: {e}")
            # No masks this frame: emit a transparent mask sized to the camera frame (from
            # image_hw) so the output stays pixel-1:1 with the color image. Fall back to
            # 1024x1024 only if the frame size wasn't provided.
            hw = in_message.get("image_hw") if hasattr(in_message, "get") else None
            h, w = (int(hw[0]), int(hw[1])) if hw else (1024, 1024)
            empty_mask = cp.zeros((h, w, 4), dtype=cp.uint8)
            out_message = Entity(context)
            out_message.add(hs.as_tensor(empty_mask), "masks")
            op_output.emit(out_message, "out")
            return

        if self.verbose:
            print("-------------------LangSAM postprocessing")
            print(f"Masks shape: {masks.shape}")
            print(f"Mask scores shape: {mask_scores.shape}")
            print(f"Boxes shape: {boxes.shape}")
            print(f"Detection scores shape: {scores.shape}")

        # Save intermediate results
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/tcn_artekmed/downloads/numpy",
                tensor=masks,
                counter=self.counter,
                word="langsam_masks",
                verbose=self.verbose,
            )

        # Composite ALL detected masks into one RGBA image, each instance in a distinct
        # color. (Previously this selected only the single argmax-scoring mask, which is
        # why just one object showed up and it flipped between objects frame-to-frame.)
        rgba_mask = self._composite_masks(masks, labels)

        # Make array contiguous
        rgba_mask = cp.ascontiguousarray(rgba_mask)

        if self.verbose:
            print(f"Output RGBA mask shape: {rgba_mask.shape}, dtype: {rgba_mask.dtype}")

        # Save final output
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/tcn_artekmed/downloads/numpy",
                tensor=rgba_mask,
                counter=self.counter,
                word="langsam_rgba",
                verbose=self.verbose,
            )

        self.counter += 1

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(rgba_mask), "masks")
        op_output.emit(out_message, "out")

