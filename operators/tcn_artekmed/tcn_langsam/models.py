# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Shared LangSAM building blocks: the SAM2 (SAM) and Grounding DINO (GDINO) wrappers plus the
# pure helper functions, imported by both langsam2operator.py (single-camera) and
# langsam_multicam_fragment.py (multi-camera).

import os
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

# Pure helpers live in a numpy-only module so they unit-test on the host.
from .helpers import (  # noqa: F401
    resolve_workers,
    worker_batch,
    worker_engine_path,
    class_id_map,
    class_id_for_label,
    build_label_map,
    build_panoptic_map,
    plan_panoptic_paint,
    panoptic_class,
    panoptic_instance,
    gdino_postprocess,
    gdino_postprocess_batch,
    build_class_token_masks,
    build_prompt_remap,
    plan_batch_padding,
    plan_batched_decode,
)


# --- CUDA-kernel-backed panoptic map (docs/specs/2026-08-10-panoptic-cuda-design.md) -----------
#
# `build_panoptic_map`'s cupy path paints one detection at a time (`pmap[masks[j] > 0] = v`),
# and cupy's boolean-mask assignment forces a `nonzero()`-driven device sync PER DETECTION.
# `tcn_panoptic_map` (operators/tcn_artekmed/tcn_panoptic_map) does the whole per-camera paint
# in one fused kernel launch instead. Opt-in via `langsam_inference.panoptic_backend: "cuda"`
# (default stays "cupy" -- see `build_panoptic_map_auto`).

_panoptic_cuda_ext = None       # None = not yet attempted; False = attempted and failed
_panoptic_cuda_warned = False


def _load_panoptic_cuda_ext():
    """Lazy-import the compiled `tcn_panoptic_map` extension; caches success AND failure so the
    import (and, on failure, the warning) happens at most once per process, not once per tick.
    """
    global _panoptic_cuda_ext, _panoptic_cuda_warned
    if _panoptic_cuda_ext is not None:
        return _panoptic_cuda_ext
    try:
        from holohub.tcn_panoptic_map._tcn_panoptic_map import (
            build_panoptic_map_cuda as _ext_build_panoptic_map_cuda,
        )
        _panoptic_cuda_ext = _ext_build_panoptic_map_cuda
    except ImportError as e:
        _panoptic_cuda_ext = False
        if not _panoptic_cuda_warned:
            _panoptic_cuda_warned = True
            print(
                "WARNING: langsam_inference.panoptic_backend is \"cuda\" but the compiled "
                f"tcn_panoptic_map extension is not importable ({e!r}). Falling back to the "
                "cupy panoptic-map backend for the rest of this run. This almost always means "
                "the extension was never built: check that "
                "add_holohub_operator(tcn_panoptic_map) is present in "
                "operators/tcn_artekmed/CMakeLists.txt and rebuild HoloHub."
            )
    return _panoptic_cuda_ext


def build_panoptic_map_cuda_backed(masks, labels, scores, cmap, height, width):
    """CUDA-kernel-backed panoptic map: same inputs/output contract as `build_panoptic_map`,
    but the per-pixel paint is one fused kernel launch (see `tcn_panoptic_map`'s design doc)
    instead of one cupy boolean-mask assignment per detection.

    `masks` must be a C-contiguous cupy uint8 array -- this does NOT silently accept a strided
    view (e.g. a slice of a larger buffer); it raises instead, because passing a strided view's
    raw pointer to the kernel would silently read the wrong bytes. Falls back to
    `build_panoptic_map` (never raises for this reason) if `masks` is empty/None, since that
    case is already a cheap no-sync zero-fill with nothing for the kernel to do, or if the
    compiled extension is not importable (see `_load_panoptic_cuda_ext`).
    """
    if masks is None or len(masks) == 0:
        return build_panoptic_map(None, [], None, cmap, height, width, xp=cp)

    ext_fn = _load_panoptic_cuda_ext()
    if ext_fn is False:
        return build_panoptic_map(masks, labels, scores, cmap, height, width, xp=cp)

    if not masks.flags.c_contiguous:
        raise ValueError(
            "build_panoptic_map_cuda_backed: masks must be C-contiguous, got a strided view "
            f"(shape={masks.shape}, strides={masks.strides}). The caller must "
            "cp.ascontiguousarray() the mask stack before calling this -- silently accepting a "
            "strided view would have the kernel read the wrong bytes.")
    if masks.dtype != cp.uint8:
        raise ValueError(
            f"build_panoptic_map_cuda_backed: masks must be uint8, got {masks.dtype}.")

    M = len(masks)
    values, priorities = plan_panoptic_paint(labels, scores, cmap, xp=cp)
    values_gpu = cp.asarray(values, dtype=cp.uint16)
    priorities_gpu = cp.asarray(priorities, dtype=cp.int32)
    out = cp.empty((height, width), dtype=cp.uint16)   # kernel writes every pixel; no zero-fill

    ext_fn(
        int(masks.data.ptr), int(values_gpu.data.ptr), int(priorities_gpu.data.ptr),
        int(M), int(height), int(width),
        int(out.data.ptr), int(cp.cuda.get_current_stream().ptr),
    )
    return out


# --- mask erosion (bleed at object edges) ------------------------------------------------------
#
# The packed map is sampled through the depth image's texcoords (`tcn_label_sampler`), so a mask
# that overshoots its object by a few COLOUR pixels labels background depth pixels as that object.
# Those points are real, finite and metres away -- they are exactly what `tcn_instance_stats`'
# percentile trim spends its breakdown point on. Eroding the map before the lookup removes the
# bleed at its source instead of paying for it downstream.

_panoptic_erode_ext = None      # None = not yet attempted; False = attempted and failed
_panoptic_erode_warned = False


def _load_panoptic_erode_ext():
    """Lazy-import the erosion entry point of the compiled `tcn_panoptic_map` extension. Caches
    success AND failure, so the import (and the warning) happens once per process, not per tick.
    """
    global _panoptic_erode_ext, _panoptic_erode_warned
    if _panoptic_erode_ext is not None:
        return _panoptic_erode_ext
    try:
        from holohub.tcn_panoptic_map._tcn_panoptic_map import (
            erode_panoptic_map_cuda as _ext_erode_panoptic_map_cuda,
        )
        _panoptic_erode_ext = _ext_erode_panoptic_map_cuda
    except ImportError as e:
        _panoptic_erode_ext = False
        if not _panoptic_erode_warned:
            _panoptic_erode_warned = True
            print(
                "WARNING: langsam_inference.mask_erosion_px is set but the compiled "
                f"tcn_panoptic_map extension does not provide erode_panoptic_map_cuda ({e!r}). "
                "Falling back to the cupy erosion for the rest of this run -- same result, more "
                "kernel launches. This usually means the extension predates the erosion and "
                "needs a rebuild."
            )
    return _panoptic_erode_ext


def erode_panoptic_map_cupy(pmap, radius):
    """Fallback erosion, and the GPU-side oracle for the kernel.

    `minimum_filter == maximum_filter` over the window is precisely "every pixel in the window
    carries the same label", so this is the same operation `launch_panoptic_erode` performs, just
    via two generic filters instead of one fused pass.

    `mode="nearest"` is exactly the kernel's clamped border: replicating the edge pixel adds only
    values already inside the truncated window, so the min and the max are unchanged by it.
    """
    from cupyx.scipy.ndimage import maximum_filter, minimum_filter

    size = 2 * int(radius) + 1
    lo = minimum_filter(pmap, size=size, mode="nearest")
    hi = maximum_filter(pmap, size=size, mode="nearest")
    return cp.where(lo == hi, pmap, cp.uint16(0)).astype(cp.uint16)


def erode_panoptic_map_auto(pmap, radius):
    """Erode every instance region of a packed panoptic map by `radius` pixels.

    `radius <= 0` returns `pmap` unchanged without touching the GPU, so leaving
    `langsam_inference.mask_erosion_px` at 0 costs nothing at all -- not even an allocation.

    Returns a NEW array; `pmap` is never mutated.
    """
    radius = int(radius)
    if radius <= 0:
        return pmap

    ext_fn = _load_panoptic_erode_ext()
    if ext_fn is False:
        return erode_panoptic_map_cupy(pmap, radius)

    if not pmap.flags.c_contiguous:
        raise ValueError(
            "erode_panoptic_map_auto: pmap must be C-contiguous, got a strided view "
            f"(shape={pmap.shape}, strides={pmap.strides}). Passing a strided view's raw pointer "
            "to the kernel would silently read the wrong bytes.")
    if pmap.dtype != cp.uint16:
        raise ValueError(f"erode_panoptic_map_auto: pmap must be uint16, got {pmap.dtype}.")

    height, width = pmap.shape
    scratch = cp.empty_like(pmap)     # separable intermediate; must not alias in or out
    out = cp.empty_like(pmap)         # kernel writes every pixel; no zero-fill
    ext_fn(
        int(pmap.data.ptr), int(scratch.data.ptr), int(out.data.ptr),
        int(height), int(width), radius,
        int(cp.cuda.get_current_stream().ptr),
    )
    return out


def build_panoptic_map_auto(masks, labels, scores, cmap, height, width, backend="cupy"):
    """Dispatch to the cupy (default, existing/oracle) or CUDA-kernel panoptic-map
    implementation by `backend` -- the `langsam_inference.panoptic_backend` config toggle.
    """
    if backend == "cuda":
        return build_panoptic_map_cuda_backed(masks, labels, scores, cmap, height, width)
    return build_panoptic_map(masks, labels, scores, cmap, height, width, xp=cp)


def build_panoptic_lut(num_classes, max_instances=64, alpha=180):
    """RGBA LUT indexed directly by a packed uint16 panoptic value (``lut[panoptic_map]``).

    Classes get large color differences (tab20 base colors); instances of a class get a
    subtle per-instance brightness variation. Class 0 (background) -> transparent.
    Returns a cupy ``uint8`` array of shape ``((num_classes + 1) << 8, 4)``.
    """
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3] * 255.0
    factors = np.array([1.0, 0.78, 0.60, 0.90, 0.68, 0.50])   # subtle per-instance brightness
    size = (num_classes + 1) << 8
    lut = np.zeros((size, 4), np.uint8)
    for c in range(1, num_classes + 1):
        base = pal[(c - 1) % 20]
        lut[c << 8, :3] = np.clip(base, 0, 255)               # instance 0 fallback -> base
        lut[c << 8, 3] = alpha
        for i in range(1, min(max_instances, 255) + 1):
            f = factors[(i - 1) % len(factors)]
            lut[(c << 8) | i, :3] = np.clip(base * f, 0, 255)
            lut[(c << 8) | i, 3] = alpha
    return cp.asarray(lut)

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

    def __init__(self, sam_type: str, ckpt_path: str | None = None, device: torch.device | None = None, compile_model: bool = False,
                 sam_backend: str = "pytorch", sam_trt_engine: str | None = None,
                 batched_decode: bool = False):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device
        self.compile_model = compile_model
        self.sam_backend = sam_backend
        self.sam_trt_engine = sam_trt_engine
        # Opt-in batched decode across cameras (predict_batch_gpu); see
        # docs/specs/2026-08-10-sam-batched-decode-design.md. Default false keeps the
        # per-camera loop (the reference implementation) live.
        self.batched_decode = batched_decode
        self.trt_encoder = None
        self.model = None
        self.mask_generator = None
        self.predictor = None
        # Last per-call sub-stage timings (ms), populated when predict_batch(timing=True):
        #   encode = set_image_batch (Hiera image encoder)
        #   decode = predict_batch   (mask decoder + postprocess_masks upsampling)
        self.last_encode_ms = 0.0
        self.last_decode_ms = 0.0
        # Guards the "falling back to the loop because resolutions differ" WARNING so it logs
        # once per instance, not once per tick (see predict_batch_gpu).
        self._warned_resolution_fallback = False

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
        if self.sam_backend == "trt":
            if not self.sam_trt_engine:
                raise ValueError('sam_backend: "trt" requires sam_trt_engine')
            self.trt_encoder = SamTrtEncoder(self.sam_trt_engine, self.device)
            print(f"SAM2 image encoder: TensorRT engine (batch {self.trt_encoder.engine_batch})")

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

    @torch.no_grad()
    def _set_image_batch_gpu(self, images_gpu):
        """GPU-native SAM2 `set_image_batch`: `images_gpu` is a list of (H,W,3) uint8 CUDA
        tensors (RGB). Replicates `predictor.set_image_batch()` exactly but runs the
        resize+normalize on the GPU (reusing SAM2's own scripted transform), so the images
        never round-trip through host memory. Sets the predictor's `_features`/`_orig_hw`
        state so `_predict` works as usual.
        """
        p = self.predictor
        p.reset_predictor()
        p._orig_hw = [(int(im.shape[0]), int(im.shape[1])) for im in images_gpu]
        resize_norm = p._transforms.transforms   # scripted Resize(1024)+Normalize (runs on GPU)
        batch = torch.stack(
            [resize_norm(im.permute(2, 0, 1).to(torch.float32).div_(255.0)) for im in images_gpu],
            dim=0,
        ).to(self.device)
        model = p.model
        if self.trt_encoder is not None:
            feats = self.trt_encoder.encode(batch)
        else:
            backbone_out = model.forward_image(batch)
            _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
            if model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
            bsz = batch.shape[0]
            feats = [
                feat.permute(1, 2, 0).view(bsz, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], p._bb_feat_sizes[::-1])
            ][::-1]
        p._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        p._is_image_set = True
        p._is_batch = True

    def _predict_batch_gpu_batched(self, p, xyxy, num_images):
        """The batched-decode path: all cameras' boxes decoded in ONE call.

        Inlines the body of the vendored SAM 2's `SAM2ImagePredictor._predict`
        (sam2_image_predictor.py, ~line 336) and `MaskDecoder.forward`/`predict_masks`
        (modeling/sam/mask_decoder.py, ~line 110/168) at
        /home/ecku/develop/vision/sam2/sam2/, with the per-image `image_embeddings`/
        `high_res_features` broadcast (`repeat_image=True`) replaced by an explicit gather
        (`repeat_image=False`) over `box_img` -- the image index each box came from -- so one
        decoder call covers every camera's boxes instead of one call per camera. See
        docs/specs/2026-08-10-sam-batched-decode-design.md §1-2.

        `_predict` cannot be called directly here: its signature commits to a single
        `img_idx`. This means a future SAM 2 upgrade that changes `_predict` or
        `predict_masks` will NOT automatically change this method -- the loop path
        (`predict_batch_gpu`'s else branch) stays the reference implementation and is not
        deleted, precisely so drift between the two is A/B-detectable rather than silent.

        `_predict` itself is `@torch.no_grad()`; since we bypass it and call
        `sam_prompt_encoder`/`sam_mask_decoder` directly, we wrap the equivalent span in
        `torch.no_grad()` explicitly below to match.

        Returns (all_masks, all_scores) in the predict_batch_gpu contract: one cupy (K,H,W)
        uint8 entry and one cupy (K,) float32 entry per input image, in input order.
        """
        counts = [int(b.shape[0]) for b in xyxy]        # cupy/torch .shape is host metadata; no sync
        total, box_img_np = plan_batched_decode(counts)
        orig_hw0 = p._orig_hw[0]
        if total == 0:
            # No boxes anywhere in this tick's batch -- skip the decoder entirely.
            empty_masks = cp.empty((0, orig_hw0[0], orig_hw0[1]), dtype=cp.uint8)
            empty_scores = cp.empty((0,), dtype=cp.float32)
            return [empty_masks.copy() for _ in range(num_images)], \
                   [empty_scores.copy() for _ in range(num_images)]

        with torch.no_grad():
            box_img = torch.as_tensor(box_img_np, device=self.device)
            boxes = torch.cat(
                [torch.as_tensor(b, dtype=torch.float, device=self.device) for b in xyxy], dim=0
            )                                             # (M,4)
            unnorm_box = p._transforms.transform_boxes(boxes, normalize=True, orig_hw=orig_hw0)
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)
            box_labels = box_labels.repeat(box_coords.size(0), 1)
            sparse, dense = p.model.sam_prompt_encoder(
                points=(box_coords, box_labels), boxes=None, masks=None
            )
            low_res, iou, _, _ = p.model.sam_mask_decoder(
                image_embeddings=p._features["image_embed"][box_img],
                image_pe=p.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=[f[box_img] for f in p._features["high_res_feats"]],
            )
            masks = p._transforms.postprocess_masks(low_res, orig_hw0) > p.mask_threshold

        masks_per_image = torch.split(masks, counts)
        scores_per_image = torch.split(iou.reshape(-1).float().contiguous(), counts)
        all_masks, all_scores = [], []
        for m, s in zip(masks_per_image, scores_per_image):
            if m.ndim == 4:
                m = m[:, 0]                                # (num_boxes, H, W), bool
            m_u8 = m.to(torch.uint8).contiguous()
            # torch -> cupy (zero-copy view) then .copy() so cupy owns the memory and it
            # survives after the torch tensors are freed.
            all_masks.append(cp.from_dlpack(m_u8).copy())
            all_scores.append(cp.from_dlpack(s.contiguous()).copy())
        return all_masks, all_scores

    def predict_batch_gpu(
        self,
        images,
        xyxy: list,
        timing: bool = False,
    ) -> tuple[list, list, None]:
        """Fully GPU-resident batched SAM: `images` is a list of (H,W,3) uint8 CUDA tensors.

        Encodes them via `_set_image_batch_gpu` (no host round-trip -- the old numpy
        `set_image_batch` path uploaded each frame from host every tick), then decodes boxes
        into masks either per-camera (the reference loop, `_prep_prompts`/`_predict`) or, when
        `self.batched_decode` is enabled and all cameras share a resolution, in a single
        batched call (`_predict_batch_gpu_batched`); see
        docs/specs/2026-08-10-sam-batched-decode-design.md. Returns cupy uint8 (N,H,W) masks +
        cupy scores so masks also never leave the device.
        """
        p = self.predictor
        with self._autocast():
            if timing:
                self._sync(); _t0 = time.perf_counter()
            self._set_image_batch_gpu(images)
            if timing:
                self._sync(); self.last_encode_ms = (time.perf_counter() - _t0) * 1000.0
                _t1 = time.perf_counter()
            num_images = len(p._features["image_embed"])

            use_batched = self.batched_decode and xyxy is not None
            if use_batched and len(set(p._orig_hw)) != 1:
                if not self._warned_resolution_fallback:
                    print(
                        f"WARNING: SAM batched_decode is enabled but cameras in this batch have "
                        f"different resolutions ({sorted(set(p._orig_hw))}); falling back to the "
                        f"per-camera decode loop for this worker. postprocess_masks and "
                        f"transform_boxes each take a single orig_hw, so batching requires a "
                        f"uniform resolution (see design §5)."
                    )
                    self._warned_resolution_fallback = True
                use_batched = False

            if use_batched:
                all_masks, all_scores = self._predict_batch_gpu_batched(p, xyxy, num_images)
            else:
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

    def predict_gpu_batch(self, images_gpu, texts_prompt, box_threshold, text_threshold, orig_hw):
        """Batched GPU detection. images_gpu: list of (H, W, C>=3) uint8 CUDA tensors, all
        the same spatial size. Returns one HF post-processed result dict per image."""
        enc = self._encode_text(texts_prompt)
        pixel_values = torch.cat([self._preprocess_image_gpu(im) for im in images_gpu], dim=0)
        n = pixel_values.shape[0]
        input_ids = enc.input_ids.repeat(n, 1)
        attention_mask = enc.attention_mask.repeat(n, 1)
        tti = enc.get("token_type_ids")
        token_type_ids = tti.repeat(n, 1) if tti is not None else None
        use_amp = self.device is not None and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type if self.device is not None else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            torch.cuda.nvtx.range_push("gdino_forward")
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("gdino_postprocess")
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold,
            text_threshold=text_threshold,
            target_sizes=[orig_hw] * n,
        )
        torch.cuda.nvtx.range_pop()
        return results




class SamTrtEncoder:
    """SAM 2 Hiera image encoder as a prebuilt fixed-batch TensorRT engine.

    Replaces the forward_image / _prepare_backbone_features / no_mem_embed / reshape block of
    SAM._set_image_batch_gpu with one execute_async_v3. Inputs and outputs stay on the GPU, so
    the decode path is unchanged. The engine runs at exactly `engine_batch` images (baked at
    ONNX trace time); a worker with fewer cameras pads and the extra slices are discarded.
    """

    def __init__(self, engine_path, device):
        import tensorrt as trt
        self.trt = trt
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self._engine_path = engine_path
        with torch.cuda.device(self.device):
            logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(
                    f"Failed to load SAM TRT engine: {engine_path}\n"
                    f"Runtime TensorRT is {trt.__version__}. Engines only load in the TRT that "
                    f"built them -- rebuild INSIDE this container:\n"
                    f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                    f"sam_trt_export.py --batch <N> --out {os.path.dirname(engine_path)}\n"
                    f"Or set langsam_inference.sam_backend: \"pytorch\" to fall back.")
            self.ctx = self.engine.create_execution_context()
            try:
                self.engine_batch = int(self.engine.get_tensor_profile_shape("image", 0)[0][0])
            except Exception as e:
                print(f"Failed to read SAM TRT engine profile (assuming batch=1): {e}")
                self.engine_batch = 1
            self._out_names = [self.engine.get_tensor_name(i)
                               for i in range(self.engine.num_io_tensors)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.OUTPUT]

    def batch_error(self, n):
        return (f"{n} images requested but the SAM encoder engine is built for batch "
                f"{self.engine_batch}. The batch is baked at ONNX trace time, so rebuild at "
                f"--batch {n}:\n"
                f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                f"sam_trt_export.py --batch {n} --out {os.path.dirname(self._engine_path)}")

    def encode(self, batch_gpu):
        """batch_gpu: (n,3,1024,1024) float32 CUDA, already resized+normalised.

        Returns [high_res_feats_0, high_res_feats_1, image_embed], each sliced back to n.
        One execution and one stream sync regardless of n.
        """
        n = int(batch_gpu.shape[0])
        try:
            pad = plan_batch_padding(n, self.engine_batch)
        except ValueError as e:
            raise ValueError(f"{e}\n{self.batch_error(n)}") from None
        with torch.cuda.device(self.device):
            img = batch_gpu
            if pad:
                img = torch.cat([img, torch.zeros((pad,) + tuple(img.shape[1:]),
                                                  device=img.device, dtype=img.dtype)], dim=0)
            img = img.contiguous()
            self.ctx.set_input_shape("image", tuple(img.shape))
            self.ctx.set_tensor_address("image", img.data_ptr())
            outs = {}
            for nm in self._out_names:
                outs[nm] = torch.empty(tuple(self.ctx.get_tensor_shape(nm)),
                                       device=self.device, dtype=torch.float32)
                self.ctx.set_tensor_address(nm, outs[nm].data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            return [outs["high_res_feats_0"][:n], outs["high_res_feats_1"][:n],
                    outs["image_embed"][:n]]


class GDinoTrtDetector:
    """Runtime Grounding DINO via a prebuilt TensorRT engine (built offline by
    docs/gdino_trt_export.py). Loads the engine + baked constant text tensors once; the engine's
    batch-dynamic optimization profile means one execute_async_v3 covers an entire WORKER's
    cameras, not one per frame.

    detect_batch(frames) -> list of (boxes_xyxy_gpu, class_ids_list, scores_gpu), one per frame
    in input order; boxes are in that frame's ORIGINAL camera resolution (pixel xyxy). Class ids
    are 1-based (active-prompt order; see set_prompts); background is never returned. This is the
    entry point used by LangSamBatchOp. detect(rgb_gpu) is a single-frame convenience wrapper
    around detect_batch, for callers with exactly one frame.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, engine_path, text_npz, prompts, device,
                 box_threshold=0.3, hw=(512, 672)):
        import tensorrt as trt
        self.trt = trt
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.H, self.W = int(hw[0]), int(hw[1])
        self.box_threshold = float(box_threshold)
        data = np.load(text_npz, allow_pickle=True)
        self._baked_prompts = [str(p) for p in list(data["prompts"])]
        with cp.cuda.Device(self.device.index):
            self._token_class_ids_baked = cp.asarray(data["token_class_ids"])
            # set_prompts indexes `cp.asarray(remap)[self._token_class_ids_baked]`, where remap
            # has length len(self._baked_prompts) + 1, so valid ids are 0 .. len(baked_prompts).
            # NumPy would raise IndexError on an out-of-range id; cupy advanced indexing does NOT
            # bounds-check the same way, so it can silently produce wrong class ids for every
            # detection with no error at all. Catch it here, once, at load time. .max()/.min()
            # are a host sync, which is fine -- this runs once in the constructor.
            tcid_max = int(self._token_class_ids_baked.max())
            tcid_min = int(self._token_class_ids_baked.min())
            if tcid_max > len(self._baked_prompts) or tcid_min < 0:
                raise ValueError(
                    f"token_class_ids in {text_npz} has values outside the valid range "
                    f"[0, {len(self._baked_prompts)}] for {len(self._baked_prompts)} baked "
                    f"prompts (found min={tcid_min}, max={tcid_max}). The likely cause is a "
                    f"caption/prompt category mismatch during export -- gdino_trt_export.py's "
                    f"build_text() prints a WARNING when the caption splits into a different "
                    f"number of categories than prompts; do not ship a build that printed it. "
                    f"Re-run --stage export with prompts that tokenize cleanly.")
        self._prompt_key = None
        self.prompts = None
        self.num_classes = 0
        self.token_class_ids = None
        self._class_masks = None
        self._text_batched = {}
        with torch.cuda.device(self.device):
            logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            self._engine_path = engine_path
            if self.engine is None:
                # Most often a TRT version mismatch: engines are version-locked, so one built
                # outside this container will not deserialize here (see the "Version tag does not
                # match ... Serialized Engine Version" line TRT logs just above).
                raise RuntimeError(
                    f"Failed to load GDINO TRT engine: {engine_path}\n"
                    f"Runtime TensorRT is {trt.__version__}. A serialized engine only loads in the "
                    f"TRT that built it -- rebuild it INSIDE this container from the exported ONNX:\n"
                    f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                    f"gdino_trt_export.py --stage build --hw {self.H} {self.W} "
                    f"--out {os.path.dirname(engine_path)}\n"
                    f"Or set langsam_inference.gdino_backend: \"pytorch\" to fall back.")
            self.ctx = self.engine.create_execution_context()
            self._text = {}
            for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"):
                want = trt.nptype(self.engine.get_tensor_dtype(n))
                t = torch.as_tensor(np.ascontiguousarray(data[n])).to(self.device)
                self._text[n] = t.to(self._torch_dtype(want)).contiguous()
            self._mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

            # The engine runs at exactly one batch size (baked at ONNX trace time), so min and
            # max of its profile are equal; read either.
            try:
                pr = self.engine.get_tensor_profile_shape("img", 0)
                self.engine_batch = int(pr[0][0])
                if int(pr[2][0]) != self.engine_batch:
                    print(f"WARNING: GDINO engine profile is not a fixed batch "
                          f"({tuple(pr[0])}..{tuple(pr[2])}); using {self.engine_batch}")
            except Exception as e:
                print(f"Failed to read GDINO TRT engine profile shape (assuming batch=1): {e}")
                self.engine_batch = 1
            self.set_prompts(prompts)

    @staticmethod
    def _torch_dtype(np_t):
        return {np.int32: torch.int32, np.int64: torch.int64, np.float32: torch.float32,
                np.float16: torch.float16, np.bool_: torch.bool}[np_t]

    def set_prompts(self, active):
        """Switch the active prompt set, adapting the baked class mapping where possible.

        Cached on the normalised prompt tuple: an unchanged set is a tuple build plus a
        comparison, which is what makes it safe to hoist the class-token masks out of the
        per-frame path. A subset and/or reordering of the baked prompts is applied by
        renumbering token_class_ids; anything else raises (see build_prompt_remap).
        """
        key = tuple(str(p).strip().lower() for p in active)
        if key == self._prompt_key:
            return
        remap = build_prompt_remap(self._baked_prompts, list(active))
        with cp.cuda.Device(self.device.index):
            self.token_class_ids = cp.asarray(remap)[self._token_class_ids_baked]
            self.num_classes = len(active)
            self._class_masks = build_class_token_masks(
                self.token_class_ids, self.num_classes, xp=cp)
        self.prompts = list(active)
        self._prompt_key = key

    def batch_error(self, n):
        """Error text for 'n cameras but the engine is built for a different fixed batch'."""
        return (
            f"{n} cameras requested but the GDINO engine is built for batch "
            f"{self.engine_batch}. The batch is baked at ONNX TRACE time, so this needs a "
            f"re-export AND a rebuild, both at --batch {n}:\n"
            f"  host:      python3 gdino_trt_export.py --stage export --prompts <...> "
            f"--hw {self.H} {self.W} --batch {n}\n"
            f"  container: python3 gdino_trt_export.py --stage build --hw {self.H} {self.W} "
            f"--batch {n} --out {os.path.dirname(self._engine_path)}\n"
            f"Or set langsam_inference.gdino_backend: \"pytorch\" to fall back.")

    def _text_for_batch(self, n):
        """Baked text tensors replicated to batch n, cached per n (they never change)."""
        t = self._text_batched.get(n)
        if t is None:
            t = {k: (v if n == 1 else v.repeat(*([n] + [1] * (v.dim() - 1)))).contiguous()
                 for k, v in self._text.items()}
            self._text_batched[n] = t
        return t

    def detect_batch(self, frames):
        """All of one worker's cameras in ONE engine execution.

        frames: list of (H0,W0,3) uint8 CUDA tensors (RGB). Returns a list of
        (boxes_xyxy_gpu, class_ids_list, scores_gpu), one per frame, in input order; boxes are
        pixel xyxy in that frame's ORIGINAL resolution.

        Exactly two synchronisation points per call regardless of camera count: the stream sync
        after the execution, and one device->host copy of the class ids and scores. The
        per-camera version cost ~5 apiece (stream sync, a D2H per class inside the decode, cupy
        boolean indexing, and cls.get()).
        """
        if not frames:
            return []
        n = len(frames)
        try:
            pad = plan_batch_padding(n, self.engine_batch)
        except ValueError as e:
            raise ValueError(f"{e}\n{self.batch_error(n)}") from None
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            hw0 = [(int(f.shape[0]), int(f.shape[1])) for f in frames]
            chw = [torch.nn.functional.interpolate(
                       f.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div(255.0),
                       size=(self.H, self.W), mode="bilinear", align_corners=False,
                       antialias=True)
                   for f in frames]
            img = ((torch.cat(chw, dim=0) - self._mean) / self._std).contiguous()
            if pad:
                # Fixed-batch engine: fill the unused slices. Their outputs are discarded, so
                # the content does not matter -- zeros avoid copying a real frame.
                img = torch.cat([img, torch.zeros((pad,) + tuple(img.shape[1:]),
                                                  device=img.device, dtype=img.dtype)], dim=0)
            self.ctx.set_input_shape("img", tuple(img.shape))
            self.ctx.set_tensor_address("img", img.data_ptr())
            for k, t in self._text_for_batch(self.engine_batch).items():
                self.ctx.set_input_shape(k, tuple(t.shape))
                self.ctx.set_tensor_address(k, t.data_ptr())
            outs = {}
            for i in range(self.engine.num_io_tensors):
                nm = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(nm) == self.trt.TensorIOMode.OUTPUT:
                    outs[nm] = torch.empty(tuple(self.ctx.get_tensor_shape(nm)),
                                           device=self.device, dtype=torch.float32)
                    self.ctx.set_tensor_address(nm, outs[nm].data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()          # sync 1 of 2
            # Basic slicing is a view (no sync, no copy): drop the padded slices here so the
            # post-process (and the sync-2 device->host copy below) sees exactly the n real
            # frames, matching hw0.
            logits = cp.from_dlpack(outs["logits"])[:n]        # (n,900,256)
            boxes = cp.from_dlpack(outs["boxes"])[:n]          # (n,900,4) cxcywh
            xyxy, best_cls, best_score = gdino_postprocess_batch(
                logits, boxes, self._class_masks, hw0, xp=cp)
            # sync 2 of 2: one copy for the whole batch. Stacked so it is a single transfer;
            # class ids are small ints, exact in float32.
            head = cp.asnumpy(cp.stack([best_cls.astype(cp.float32), best_score]))  # (2,n,Q)
            cls_h, score_h = head[0], head[1]
            results = []
            for i in range(n):
                keep = np.nonzero(score_h[i] > self.box_threshold)[0]
                if len(keep) == 0:
                    results.append((xyxy[i][:0], [], best_score[i][:0]))
                    continue
                gidx = cp.asarray(keep)      # integer (not boolean) indexing -> no sync
                results.append((xyxy[i][gidx],
                                [int(c) for c in cls_h[i][keep]],
                                best_score[i][gidx]))
            return results

    def detect(self, rgb_gpu):
        """Single-frame convenience wrapper. rgb_gpu: (H0, W0, 3) uint8 CUDA tensor (RGB)."""
        return self.detect_batch([rgb_gpu])[0]
