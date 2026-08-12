# SPDX-License-Identifier: Apache-2.0
"""Multi-camera LangSAM: batched Grounding DINO + SAM2 over all color cameras, split across
GPUs, emitting a composite entity of per-camera uint16 panoptic maps (class<<8 | instance)
plus a tiled colorized view.

See docs/specs/2026-07-28-langsam-multicam-design.md.
"""

import logging
from operators.tcn_artekmed.tcn_util.frame_identity import acq_timestamp, acq_timestamp_consensus, tensor_names
import math
import os

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
import holoscan as hs
from holoscan.core import ConditionType, Operator, OperatorSpec, Subgraph, IOSpec
from holoscan.operators import HolovizOp

from .models import (
    SAM, GDINO, GDinoTrtDetector, resolve_workers, worker_batch, worker_engine_path, class_id_map, build_panoptic_map, build_panoptic_map_auto, build_panoptic_lut,
)
# Shared with langsam_pipelined.py so the output-key convention can't drift between the
# monolithic and split ops; imported directly (not via langsam_common's re-export list).
from .helpers import mask_name, resolve_flipped_cameras, validate_flip_cameras
from operators.tcn_artekmed.tcn_util.rotate import rotate180
from .realtime_ops import GdinoOp, SamOp, PanopticOp

log = logging.getLogger(__name__)


class LangSamBatchOp(Operator):
    """Batched LangSAM over a subset of cameras on one GPU.

    In:  `color_input` (the full color_outputs entity, named GPU tensors).
    Out: `masks` (dict of `<cam>_mask` -> cupy uint16 (H,W) panoptic maps `(class<<8|inst)`
         on `device`).
    """

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, promptable=False, flip_cameras=None, **kwargs):
        self.cameras = list(cameras)
        # This worker's engines are built for exactly its camera count. Resolving the paths
        # here -- rather than passing one fixed path for every worker -- is what lets a
        # 2-camera worker stop paying 3-camera cost.
        self.batch = len(self.cameras)
        gdino_engine = worker_engine_path(langsam_cfg.get("gdino_trt_engine"), self.batch)
        sam_engine = worker_engine_path(langsam_cfg.get("sam_trt_engine"), self.batch)
        for kind, path in (("GDINO", gdino_engine), ("SAM", sam_engine)):
            backend = langsam_cfg.get("gdino_backend" if kind == "GDINO" else "sam_backend",
                                      "pytorch")
            if backend == "trt" and path and not os.path.exists(path):
                raise FileNotFoundError(
                    f"{kind} engine for batch {self.batch} not found: {path}\n"
                    f"This worker owns {self.batch} cameras, so it needs a batch-{self.batch} "
                    f"engine. Build it with --from-config, or for this batch alone:\n"
                    f"  GDINO (host then container): gdino_trt_export.py --stage export "
                    f"--batch {self.batch} ... ; --stage build --batch {self.batch} ...\n"
                    f"  SAM (container):             sam_trt_export.py --batch {self.batch} ...")
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.prompts = list(prompts)
        self._prompt_key = None          # set by _apply_prompts; see _receive_prompts
        self.promptable = bool(promptable)
        # Cameras mounted upside down: rotate 180 degrees on the way INTO the models and rotate the
        # resulting mask back on the way out. Recognition improves because both models are trained on
        # upright scenes; nothing outside this operator ever sees the rotated data, so every geometric
        # consumer downstream still works in the camera's native orientation.
        self._flip = resolve_flipped_cameras(self.cameras, flip_cameras)
        if self._flip:
            log.info(f"LangSamBatchOp[{device}]: rotating 180 deg for {sorted(self._flip)}")
        self.box_threshold = float(langsam_cfg.get("box_threshold", 0.3))
        self.text_threshold = float(langsam_cfg.get("text_threshold", 0.25))
        self.gdino_backend = langsam_cfg.get("gdino_backend", "pytorch")
        self.panoptic_backend = langsam_cfg.get("panoptic_backend", "cupy")
        super().__init__(fragment, *args, **kwargs)
        with torch.cuda.device(self.device):
            self.sam = SAM(
                langsam_cfg.get("sam_type", "sam2.1_hiera_tiny"),
                langsam_cfg.get("sam_ckpt_path"),
                device=self.device,
                compile_model=bool(langsam_cfg.get("sam_compile", False)),
                sam_backend=langsam_cfg.get("sam_backend", "pytorch"),
                sam_trt_engine=sam_engine,
                batched_decode=bool(langsam_cfg.get("sam_batched_decode", False)),
            )
            self.sam.build_model()
            self.gdino = None
            self.gdino_trt = None
            if self.gdino_backend == "trt":
                hw = tuple(langsam_cfg.get("gdino_trt_hw", [512, 672]))
                self.gdino_trt = GDinoTrtDetector(
                    gdino_engine, langsam_cfg["gdino_trt_text"],
                    self.prompts, self.device, self.box_threshold, hw,
                )
                # Catch a stale/undersized engine at construction, before the whole Holoscan
                # graph is composed and every model is loaded -- detect_batch would otherwise
                # only raise this deep inside compute() on the first tick.
                if len(self.cameras) > self.gdino_trt.engine_batch:
                    raise ValueError(self.gdino_trt.batch_error(len(self.cameras)))
            else:
                self.gdino = GDINO(
                    model_ckpt_path=langsam_cfg.get("gdino_model_ckpt_path"),
                    processor_ckpt_path=langsam_cfg.get("gdino_processor_ckpt_path"),
                    device=self.device,
                    model_id=langsam_cfg.get("gdino_model_id", "IDEA-Research/grounding-dino-tiny"),
                    input_size=langsam_cfg.get("gdino_input_size"),
                    compile_model=bool(langsam_cfg.get("gdino_compile", False)),
                )
                self.gdino.build_model()

            self._apply_prompts(self.prompts)

    def _apply_prompts(self, prompts):
        """Single point of truth for prompt-derived state.

        Three things are derived from the prompt list and MUST move together, or class ids and
        mask colours silently disagree with the detections: the detector's token->class map,
        `self.prompts` (used to label boxes for SAM), and `self._cmap` (used to build the
        panoptic map). The TRT detector accepts any subset/reordering of its baked prompts and
        raises otherwise; the pytorch backend tokenises per call and accepts anything.
        """
        prompts = list(prompts)
        if [str(x).strip().lower() for x in prompts] == self._prompt_key:
            return False                             # unchanged: the publisher re-sends every tick
        if self.gdino_trt is not None:
            self.gdino_trt.set_prompts(prompts)      # raises first, before any state moves
        self.prompts = prompts
        self._cmap = class_id_map(self.prompts)
        self._prompt_key = [str(x).strip().lower() for x in prompts]
        return True

    def _receive_prompts(self, op_input):
        """Apply a prompt update if one is waiting. No-op when the port is absent or empty.

        The port carries no condition, so a promptable worker still ticks on colour frames alone --
        prompts arrive only when someone changes them, and gating compute() on them would stall the
        pipeline until the first update.

        A prompt change is NOT atomic across workers: each applies the update on its own next tick,
        so for one frame two workers can label with different class ids and the collector will merge
        them. That is visible as a single frame of mixed colours, and is the reason a prompt change
        should be treated as a scene change rather than a per-frame control.
        """
        if not self.promptable:
            return
        try:
            msg = op_input.receive("prompts")
        except Exception:
            return
        if not msg:
            return
        new_prompts = msg.get("text_prompts") if hasattr(msg, "get") else None
        if not new_prompts:
            return
        try:
            if self._apply_prompts(list(new_prompts)):
                log.info(f"LangSamBatchOp[{self.device}]: prompts now {self.prompts}")
        except ValueError as e:
            # A TRT-backed detector can only express a subset/reordering of its baked vocabulary.
            # Keep running on the previous prompts rather than killing the graph mid-stream.
            log.error(f"LangSamBatchOp[{self.device}]: rejected prompt update {list(new_prompts)}: "
                      f"{e}; staying on {self.prompts}")

    def setup(self, spec: OperatorSpec):
        spec.input("color_input")
        if self.promptable:
            # See _receive_prompts: conditionless, so colour frames alone keep the worker ticking.
            spec.input("prompts").condition(ConditionType.NONE)
        spec.output("masks")

    def _extract_result(self, r):
        """One HF GDINO result dict -> (boxes kept ON GPU, labels as list[str]).

        Boxes stay on the GPU: SAM's `_prep_prompts` does `torch.as_tensor(box, device=...)`,
        which is a no-op for a GPU tensor -- so we avoid the box D2H + H2D round-trip (and the
        sync it forces) that the old numpy conversion caused per camera.
        """
        boxes = r.get("boxes")
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().float()          # stays on GPU
        labels = r.get("text_labels", r.get("labels", []))
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return boxes, [str(x) for x in labels]

    def compute(self, op_input, op_output, context):
        self._receive_prompts(op_input)
        msg = op_input.receive("color_input")
        # Frame identity must be forwarded explicitly -- see acq_timestamp's docstring.
        acq = acq_timestamp(op_input, "color_input")
        out = {}
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            rgb_gpu, names, hw = [], [], None
            for cam in self.cameras:
                t = msg.get(cam)
                if t is None:
                    log.warning(f"LangSamBatchOp[{self.device}]: missing tensor '{cam}'")
                    continue
                img = torch.from_dlpack(cp.asarray(t))               # (H,W,4) BGRA uint8, cuda:0
                img = img.to(self.device)[..., [2, 1, 0]].contiguous()  # -> RGB on this device
                if cam in self._flip:
                    # 180 deg = reverse both spatial axes. Exactly invertible, no resampling.
                    img = torch.flip(img, dims=(0, 1)).contiguous()
                rgb_gpu.append(img)
                names.append(cam)
                hw = (int(img.shape[0]), int(img.shape[1]))

            if not rgb_gpu:
                op_output.emit(out, "masks", acq_timestamp=acq)
                return

            # --- Grounding DINO over this worker's cameras (TRT engine or PyTorch) ---
            # Partition: only cameras with >=1 detection go to SAM; others are all-background.
            # Boxes stay on the GPU for SAM's _prep_prompts (no D2H/H2D round-trip).
            sam_imgs, sam_boxes, sam_labels, sam_idx = [], [], [], []
            torch.cuda.nvtx.range_push("gdino")
            if self.gdino_backend == "trt":
                # One engine execution for every camera on this worker (see detect_batch).
                for i, (boxes, cls, _) in enumerate(self.gdino_trt.detect_batch(rgb_gpu)):
                    if len(cls) > 0:
                        sam_imgs.append(rgb_gpu[i])
                        sam_boxes.append(boxes)
                        sam_labels.append([self.prompts[c - 1] for c in cls])
                        sam_idx.append(i)
            else:
                gres = self.gdino.predict_gpu_batch(
                    rgb_gpu, self.prompts, self.box_threshold, self.text_threshold, hw)
                for i, r in enumerate(gres):
                    boxes, labels = self._extract_result(r)
                    if boxes is not None and len(boxes) > 0:
                        sam_imgs.append(rgb_gpu[i])
                        sam_boxes.append(boxes)
                        sam_labels.append(labels)
                        sam_idx.append(i)
            torch.cuda.nvtx.range_pop()

            pmaps = {i: build_panoptic_map(None, [], None, self._cmap, hw[0], hw[1], xp=cp)
                     for i in range(len(names))}

            if sam_imgs:
                torch.cuda.nvtx.range_push("sam")
                masks, mscores, _ = self.sam.predict_batch_gpu(sam_imgs, xyxy=sam_boxes, timing=False)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("panoptic")
                for k, i in enumerate(sam_idx):
                    pmaps[i] = build_panoptic_map_auto(
                        masks[k], sam_labels[k], mscores[k], self._cmap, hw[0], hw[1],
                        backend=self.panoptic_backend)
                torch.cuda.nvtx.range_pop()

            for i, cam in enumerate(names):
                pmap = pmaps[i]
                if cam in self._flip:
                    # Undo the input rotation so the map is in the camera's native orientation --
                    # which is what tcn_label_sampler's texcoords index.
                    pmap = rotate180(pmap)
                out[mask_name(cam)] = hs.as_tensor(cp.ascontiguousarray(pmap))
        op_output.emit(out, "masks", acq_timestamp=acq)


class MaskCollectorOp(Operator):
    """Merge per-worker label-map dicts into one composite dict, all consolidated on GPU 0."""

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=IOSpec.ANY_SIZE)
        spec.output("masks")

    def compute(self, op_input, op_output, context):
        messages = op_input.receive("receivers")     # tuple of per-worker dicts
        # All workers process the SAME source frame, so they should agree; disagreement is warned
        # about rather than hidden, because it would make downstream grouping quietly wrong.
        acq = acq_timestamp_consensus(op_input, "receivers", log)
        out = {}
        with cp.cuda.Device(0):
            for msg in messages:
                if msg is None:
                    continue
                for name in tensor_names(msg):
                    arr = cp.asarray(msg.get(name))
                    if arr.device.id != 0:            # cross-GPU -> consolidate on GPU 0
                        t = torch.from_dlpack(arr).to("cuda:0")
                        arr = cp.ascontiguousarray(cp.from_dlpack(t))
                    out[name] = hs.as_tensor(arr)
        op_output.emit(out, "masks", acq_timestamp=acq)


class LabelMapColorizeOp(Operator):
    """Colorize per-camera label maps via a class LUT -> RGBA, and emit tiled Holoviz specs."""

    def __init__(self, fragment, *args, num_classes, alpha=180, promptable=False, **kwargs):
        # Panoptic LUT indexed directly by the packed uint16 value (class<<8|instance):
        # class = large color difference, instance = subtle brightness variation.
        self._alpha = alpha
        self._num_classes = int(num_classes)
        self._lut = build_panoptic_lut(self._num_classes, alpha=alpha)
        self.promptable = bool(promptable)
        super().__init__(fragment, *args, **kwargs)

    def _receive_prompts(self, op_input):
        """Grow the LUT when a prompt update adds classes.

        The LUT is indexed by the packed label, so it must cover the highest class id in use; a map
        containing class 6 against a 5-class LUT indexes out of bounds. It is only ever grown, never
        shrunk, so a class that disappears and comes back keeps its colour -- stable colours across
        prompt edits matter more than a few unused LUT rows.
        """
        if not self.promptable:
            return
        try:
            msg = op_input.receive("prompts")
        except Exception:
            return
        prompts = msg.get("text_prompts") if msg and hasattr(msg, "get") else None
        if not prompts or len(prompts) <= self._num_classes:
            return
        self._num_classes = len(prompts)
        self._lut = build_panoptic_lut(self._num_classes, alpha=self._alpha)
        log.info(f"LabelMapColorizeOp: LUT grown to {self._num_classes} classes")

    def setup(self, spec: OperatorSpec):
        spec.input("masks")
        if self.promptable:
            spec.input("prompts").condition(ConditionType.NONE)
        spec.output("viz")
        spec.output("specs")

    def compute(self, op_input, op_output, context):
        self._receive_prompts(op_input)
        msg = op_input.receive("masks")
        acq = acq_timestamp(op_input, "masks")
        names = tensor_names(msg)
        out = {}
        with cp.cuda.Device(0):
            for name in names:
                pmap = cp.asarray(msg.get(name))        # (H,W) uint16 panoptic (class<<8|inst)
                rgba = self._lut[pmap]                   # (H,W,4) uint8
                out[name] = hs.as_tensor(cp.ascontiguousarray(rgba))
        op_output.emit(out, "viz", acq_timestamp=acq)
        op_output.emit(self._tiled_specs(names), "specs", acq_timestamp=acq)

    def _tiled_specs(self, names):
        grid = int(math.ceil(math.sqrt(len(names)))) if names else 1
        tile = 1.0 / grid
        specs = []
        for i, name in enumerate(names):
            spec = HolovizOp.InputSpec(name, HolovizOp.InputType.COLOR)
            view = HolovizOp.InputSpec.View()
            view.offset_x = (i % grid) * tile
            view.offset_y = (i // grid) * tile
            view.width = tile
            view.height = tile
            spec.views = [view]
            specs.append(spec)
        return specs


class RealtimeLangSamSubgraph(Subgraph):
    """Wires N per-GPU LangSamBatchOp workers -> collector -> colorize."""

    def __init__(self, fragment, name, kwargs, all_color_cameras):
        self.kwargs = kwargs
        self.all_color_cameras = list(all_color_cameras)
        super().__init__(fragment, name)

    def _n(self, s):
        return f"{self.name}_{s}"

    def _get(self, key):
        try:
            return self.kwargs(key)
        except Exception:
            return None

    def compose(self):
        log.info("Compose subgraph: LangSamMultiCamProcessing")
        multicam_cfg = self._get("gpu_workers") or {}
        langsam_cfg = self.kwargs("langsam_inference")
        prompts = (self._get("text_prompts") or {}).get("prompts", [])
        # Cameras mounted upside down: rotated only while passing through the models. Validated
        # here, against every camera, because a worker sees only its own share.
        flip_cameras = list(langsam_cfg.get("flip_cameras") or [])
        validate_flip_cameras(self.all_color_cameras, flip_cameras)
        if flip_cameras:
            log.info(f"Rotating 180 deg through LangSAM for: {flip_cameras}")
        workers = resolve_workers(multicam_cfg, self.all_color_cameras)
        log.info(f"LangSAM multicam workers: {workers}")

        collector = MaskCollectorOp(self, name=self._n("collector"))
        colorize = LabelMapColorizeOp(self, name=self._n("colorize"), num_classes=len(prompts))
        self.add_flow(collector, colorize, {("masks", "masks")})

        pipelined = bool(multicam_cfg.get("pipelined", False))
        log.info(f"LangSAM worker structure: {'pipelined (3 ops)' if pipelined else 'monolithic'}")
        for i, w in enumerate(workers):
            kw = dict(cameras=w["cameras"], device=w["device"],
                      langsam_cfg=langsam_cfg, prompts=prompts,
                      flip_cameras=flip_cameras)
            if pipelined:
                gd = GdinoOp(self, name=self._n(f"worker{i}_gdino"), **kw)
                sm = SamOp(self, name=self._n(f"worker{i}_sam"), **kw)
                pn = PanopticOp(self, name=self._n(f"worker{i}_panoptic"), **kw)
                self.add_flow(gd, sm, {("det", "det")})
                self.add_flow(sm, pn, {("seg", "seg")})
                self.add_flow(pn, collector, {("masks", "receivers")})
                self.add_input_interface_port("input", gd, "color_input")
            else:
                op = LangSamBatchOp(self, name=self._n(f"worker{i}"), **kw)
                self.add_flow(op, collector, {("masks", "receivers")})
                self.add_input_interface_port("input", op, "color_input")

        self.add_output_interface_port("output_masks", collector, "masks")
        self.add_output_interface_port("output_viz", colorize, "viz")
        self.add_output_interface_port("output_specs", colorize, "specs")
