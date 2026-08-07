# SPDX-License-Identifier: Apache-2.0
"""Pipelined LangSAM: the three stages of LangSamBatchOp as separate operators.

LangSamBatchOp runs gdino -> sam -> panoptic sequentially inside one compute(), so a worker's
tick costs the SUM of its stages. Holoscan's event-based scheduler runs *different* operators
concurrently, so splitting the chain lets stage N of frame k overlap stage N-1 of frame k+1 and
the period tends toward the MAX stage instead of the sum.

Measured justification and the two floors this is expected to hit are in
docs/specs/2026-08-07-langsam-pipelining-design.md. Selected by `gpu_workers.pipelined: true`;
LangSamBatchOp remains the default so the two can be A/B'd.

This is a behaviour-neutral refactor: same stage boundaries, same NVTX names, same outputs --
including quirks (see the spec's "behaviour that must not change").
"""
import logging
import os

import cupy as cp
import torch
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from langsam_common import (
    SAM, GDINO, GDinoTrtDetector, class_id_map, build_panoptic_map, worker_engine_path,
)

log = logging.getLogger(__name__)


def _mask_name(cam_port):
    """`camera01_colorimage` -> `camera01_mask`. Mirrors langsam_multicam_fragment."""
    return cam_port.replace("_colorimage", "") + "_mask"


def _resolve_engine(langsam_cfg, path_key, backend_key, batch, kind):
    """Resolve a worker's `{batch}` engine path and fail fast if it is missing.

    Same check LangSamBatchOp does, but each pipelined operator only validates the engine it
    actually loads -- GdinoOp the detector's, SamOp the encoder's.
    """
    path = worker_engine_path(langsam_cfg.get(path_key), batch)
    if langsam_cfg.get(backend_key, "pytorch") == "trt" and path and not os.path.exists(path):
        raise FileNotFoundError(
            f"{kind} engine for batch {batch} not found: {path}\n"
            f"This worker owns {batch} cameras, so it needs a batch-{batch} engine. Build it "
            f"with --from-config (docs/gdino_trt_export.py / docs/sam_trt_export.py).")
    return path


class GdinoOp(Operator):
    """Stage 1: frames in, detections out.

    Owns the Grounding DINO detector and the prompt list used to label boxes. Drops frames for
    cameras with no detection, so only what SAM needs crosses to the next stage.
    """

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, **kwargs):
        self.cameras = list(cameras)
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.prompts = list(prompts)
        self.batch = len(self.cameras)
        self.box_threshold = float(langsam_cfg.get("box_threshold", 0.3))
        self.text_threshold = float(langsam_cfg.get("text_threshold", 0.25))
        self.gdino_backend = langsam_cfg.get("gdino_backend", "pytorch")
        gdino_engine = _resolve_engine(langsam_cfg, "gdino_trt_engine", "gdino_backend",
                                       self.batch, "GDINO")
        super().__init__(fragment, *args, **kwargs)
        with torch.cuda.device(self.device):
            self.gdino = None
            self.gdino_trt = None
            if self.gdino_backend == "trt":
                hw = tuple(langsam_cfg.get("gdino_trt_hw", [512, 672]))
                self.gdino_trt = GDinoTrtDetector(
                    gdino_engine, langsam_cfg["gdino_trt_text"],
                    self.prompts, self.device, self.box_threshold, hw,
                )
                if len(self.cameras) > self.gdino_trt.engine_batch:
                    raise ValueError(self.gdino_trt.batch_error(len(self.cameras)))
                self.gdino_trt.set_prompts(self.prompts)
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

    def setup(self, spec: OperatorSpec):
        spec.input("color_input")
        spec.output("det")

    @staticmethod
    def _extract_result(r):
        """One HF GDINO result dict -> (boxes kept ON GPU, labels as list[str])."""
        boxes = r.get("boxes")
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().float()
        labels = r.get("text_labels", r.get("labels", []))
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return boxes, [str(x) for x in labels]

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("color_input")
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            rgb_gpu, names, hw = [], [], None
            for cam in self.cameras:
                t = msg.get(cam)
                if t is None:
                    log.warning(f"GdinoOp[{self.device}]: missing tensor '{cam}'")
                    continue
                img = torch.from_dlpack(cp.asarray(t))
                img = img.to(self.device)[..., [2, 1, 0]].contiguous()
                rgb_gpu.append(img)
                names.append(cam)
                hw = (int(img.shape[0]), int(img.shape[1]))   # last camera's shape, preserved

            if not rgb_gpu:
                op_output.emit({"names": [], "hw": None, "sam_idx": [], "sam_imgs": [],
                                "sam_boxes": [], "sam_labels": []}, "det")
                return

            sam_imgs, sam_boxes, sam_labels, sam_idx = [], [], [], []
            torch.cuda.nvtx.range_push("gdino")
            if self.gdino_backend == "trt":
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

        op_output.emit({"names": names, "hw": hw, "sam_idx": sam_idx, "sam_imgs": sam_imgs,
                        "sam_boxes": sam_boxes, "sam_labels": sam_labels}, "det")


class SamOp(Operator):
    """Stage 2: detections in, masks out. Owns the SAM 2 model."""

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, **kwargs):
        self.cameras = list(cameras)
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.batch = len(self.cameras)
        sam_engine = _resolve_engine(langsam_cfg, "sam_trt_engine", "sam_backend",
                                     self.batch, "SAM")
        super().__init__(fragment, *args, **kwargs)
        with torch.cuda.device(self.device):
            self.sam = SAM(
                langsam_cfg.get("sam_type", "sam2.1_hiera_tiny"),
                langsam_cfg.get("sam_ckpt_path"),
                device=self.device,
                compile_model=bool(langsam_cfg.get("sam_compile", False)),
                sam_backend=langsam_cfg.get("sam_backend", "pytorch"),
                sam_trt_engine=sam_engine,
            )
            self.sam.build_model()

    def setup(self, spec: OperatorSpec):
        spec.input("det")
        spec.output("seg")

    def compute(self, op_input, op_output, context):
        p = op_input.receive("det")
        out = {"names": p["names"], "hw": p["hw"], "sam_idx": p["sam_idx"],
               "sam_labels": p["sam_labels"], "masks": [], "scores": []}
        if not p["sam_idx"]:
            op_output.emit(out, "seg")
            return
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            torch.cuda.nvtx.range_push("sam")
            masks, mscores, _ = self.sam.predict_batch_gpu(
                p["sam_imgs"], xyxy=p["sam_boxes"], timing=False)
            torch.cuda.nvtx.range_pop()
        out["masks"], out["scores"] = masks, mscores
        op_output.emit(out, "seg")


class PanopticOp(Operator):
    """Stage 3: masks in, packed (class<<8|instance) maps out. Owns the class-id map."""

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, **kwargs):
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.prompts = list(prompts)
        self._cmap = class_id_map(self.prompts)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("seg")
        spec.output("masks")

    def compute(self, op_input, op_output, context):
        p = op_input.receive("seg")
        names, hw = p["names"], p["hw"]
        out = {}
        if not names:
            op_output.emit(out, "masks")
            return
        with cp.cuda.Device(self.device.index):
            pmaps = {i: build_panoptic_map(None, [], None, self._cmap, hw[0], hw[1], xp=cp)
                     for i in range(len(names))}
            if p["sam_idx"]:
                torch.cuda.nvtx.range_push("panoptic")
                for k, i in enumerate(p["sam_idx"]):
                    pmaps[i] = build_panoptic_map(
                        p["masks"][k], p["sam_labels"][k], p["scores"][k],
                        self._cmap, hw[0], hw[1], xp=cp)
                torch.cuda.nvtx.range_pop()
            for i, cam in enumerate(names):
                out[_mask_name(cam)] = hs.as_tensor(cp.ascontiguousarray(pmaps[i]))
        op_output.emit(out, "masks")
