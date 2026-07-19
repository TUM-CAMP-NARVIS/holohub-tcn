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
import gc

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
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

    def __init__(self, sam_type: str, ckpt_path: str | None = None, device: torch.device | None = None):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = None
        self.mask_generator = None
        self.predictor = None


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

    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.predictor.set_image(image_rgb)
        masks, scores, logits = self.predictor.predict(box=xyxy, multimask_output=False)
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        self.predictor.set_image_batch(images_rgb)

        masks, scores, logits = self.predictor.predict_batch(box_batch=xyxy, multimask_output=False)

        masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
        scores = [np.squeeze(score) for score in scores]
        logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
        return masks, scores, logits


class GDINO:
    def __init__(self, model_ckpt_path: str | None = None, processor_ckpt_path: str | None = None, device: torch.device | None = None):
        self.model_ckpt_path = model_ckpt_path
        self.processor_ckpt_path = processor_ckpt_path
        self.device = device
        self.model = None
        self.processor = None

    def build_model(self):
        if not self.model_ckpt_path or not self.processor_ckpt_path: # indicates that we somehow able to load the model from internet
            model_id = "IDEA-Research/grounding-dino-base"
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
                model_id = "IDEA-Research/grounding-dino-base"
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
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
                model_id = "IDEA-Research/grounding-dino-base"
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

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
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
        return results





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

    def __init__(self, *args, sam_type="sam2.1_hiera_small", sam_ckpt_path: str | None = None, gdino_model_ckpt_path: str | None = None, gdino_processor_ckpt_path: str | None = None, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_type = sam_type
        self.device = device

        # Initialize SAM model
        self.sam = SAM(sam_type, sam_ckpt_path, device=device)
        self.sam.build_model()

        # Initialize Grounding DINO model
        self.gdino = GDINO(model_ckpt_path=gdino_model_ckpt_path, processor_ckpt_path=gdino_processor_ckpt_path, device=device)
        self.gdino.build_model()

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

        image_np = cp.asarray(image_tensor).get()

        # Convert RGBA to RGB (drop alpha channel)
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        # Convert to PIL Image
        image_pil = Image.fromarray(image_np.astype(np.uint8))

        # Get text prompts (assume it's a list of strings or a single string)
        text_prompts = text_prompts_message.get("text_prompts")
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # Wrap single image and prompts in lists for batch processing
        images_pil = [image_pil]
        texts_prompt = text_prompts if isinstance(text_prompts, list) else [text_prompts]

        # Get threshold parameters
        box_threshold = self.box_threshold
        text_threshold = self.text_threshold

        # Run Grounding DINO to get bounding boxes
        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)

        # Process results and prepare for SAM
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []

        for idx, result in enumerate(gdino_results):
            # Convert tensors to numpy arrays
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            # Check if any objects were detected
            if result.get("labels") and len(result["labels"]) > 0:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)

        # Run SAM2 to generate masks if any boxes were detected
        if sam_images:
            print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
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

        # Convert results to CuPy tensors for output
        # For single image case, extract the first result
        result = all_results[0]

        # Create output message with the result
        out_message = Entity(context)

        # Convert numpy arrays to CuPy tensors
        # Note: Skip "labels" as it contains strings which CuPy doesn't support
        for key in ["boxes", "scores"]:
            if key in result and len(result[key]) > 0:
                out_message.add(hs.as_tensor(cp.asarray(result[key])), key)

        # Add masks and mask_scores if available
        if len(result["masks"]) > 0:
            out_message.add(hs.as_tensor(cp.asarray(result["masks"])), "masks")
            out_message.add(hs.as_tensor(cp.asarray(result["mask_scores"])), "mask_scores")

        op_output.emit(out_message, "out")

        # Clean up GPU memory
        # Delete large intermediate variables
        del sam_images, sam_boxes, all_results, result
        if 'masks' in locals():
            del masks, mask_scores

        # Clear PyTorch and Python garbage collection
        gc.collect()
        torch.cuda.empty_cache()


class LangSamPostprocessorOp(Operator):
    """Operator to post-process LangSAM inference output for visualization"""

    def __init__(
        self,
        *args,
        save_intermediate=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.counter = 0
        self.painter = CupyArrayPainter()
        self.save_intermediate = save_intermediate

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
        except Exception as e:
            if self.verbose:
                print(f"Error extracting data from input message: {e}")
            # If no masks detected, create empty output
            empty_mask = cp.zeros((1024, 1024, 4), dtype=cp.uint8)
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

        # Find the best mask based on mask scores
        if len(mask_scores.shape) > 1:
            # Multiple masks per detection
            best_mask_idx = cp.argmax(mask_scores[:, 0])
        else:
            # Single mask score per detection
            best_mask_idx = cp.argmax(mask_scores)

        # Extract the best mask
        if len(masks.shape) == 4:
            # Shape: (num_detections, num_masks_per_detection, H, W)
            best_mask = masks[best_mask_idx, 0]
        elif len(masks.shape) == 3:
            # Shape: (num_detections, H, W)
            best_mask = masks[best_mask_idx]
        else:
            # Shape: (H, W)
            best_mask = masks

        if self.verbose:
            print(f"Selected mask shape: {best_mask.shape}")

        # Convert binary mask to RGBA for visualization
        # Ensure mask is 2D
        if len(best_mask.shape) > 2:
            best_mask = cp.squeeze(best_mask)

        # Convert to RGBA using the painter
        rgba_mask = self.painter.to_rgba(best_mask)

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

