# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Shared LangSAM building blocks: the SAM2 (SAM) and Grounding DINO (GDINO) wrappers plus the
# pure helper functions, imported by both langsam2operator.py (single-camera) and
# langsam_multicam_fragment.py (multi-camera).

import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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





