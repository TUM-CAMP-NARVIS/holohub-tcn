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
    SAM, GDINO, GDinoTrtDetector, class_id_map, build_panoptic_map, build_panoptic_map_auto,
    worker_engine_path,
)
# Shared with langsam_multicam_fragment.py so the output-key convention can't drift between the
# monolithic and split ops; imported directly (not via langsam_common's re-export list).
from langsam_helpers import mask_name

log = logging.getLogger(__name__)


def _resolve_engine(langsam_cfg, path_key, backend_key, batch, kind):
    """Resolve a worker's `{batch}` engine path and fail fast if it is missing.

    Same check LangSamBatchOp does, but each pipelined operator only validates the engine it
    actually loads -- GdinoOp the detector's, SamOp the encoder's.
    """
    path = worker_engine_path(langsam_cfg.get(path_key), batch)
    if langsam_cfg.get(backend_key, "pytorch") == "trt" and path and not os.path.exists(path):
        if kind == "GDINO":
            build_cmd = (f"  GDINO (host then container): gdino_trt_export.py --stage export "
                        f"--batch {batch} ... ; --stage build --batch {batch} ...")
        else:
            build_cmd = f"  SAM (container):             sam_trt_export.py --batch {batch} ..."
        raise FileNotFoundError(
            f"{kind} engine for batch {batch} not found: {path}\n"
            f"This worker owns {batch} cameras, so it needs a batch-{batch} engine. Build it "
            f"with --from-config, or for this batch alone:\n"
            f"{build_cmd}")
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


def _assert_shared_default_stream_env():
    """Fail fast if cupy is configured for per-thread default streams -- process-global (an env
    var read at process start), so this is safe and sufficient to check once, at construction
    time. See `SamOp`'s docstring for the full stream-ordering explanation this guards, and
    `_stream_is_default` below for the OTHER half of that guard, which is thread-local and
    cannot be checked here.
    """
    per_thread = os.environ.get("CUPY_CUDA_PER_THREAD_DEFAULT_STREAM", "0")
    if per_thread not in ("", "0"):
        raise RuntimeError(
            "CUPY_CUDA_PER_THREAD_DEFAULT_STREAM is set. SamOp emits masks whose "
            "device-to-device copies are only ENQUEUED, not complete (predict_batch_gpu runs "
            "with timing=False, so it never synchronises), and PanopticOp may run on a "
            "different EventBasedScheduler worker thread than SamOp. With per-thread default "
            "streams, cupy's default stream on PanopticOp's thread no longer serialises "
            "against SamOp's, so PanopticOp's reads can race SamOp's writes -- corrupting or "
            "emptying panoptic maps intermittently, not crashing. Fix: unset this env var and "
            "keep the shared legacy default stream, or make SamOp synchronise explicitly "
            "(e.g. cp.cuda.Stream.null.synchronize()) before it emits.")


def _stream_is_default(device):
    """True if torch's CURRENT stream on THIS thread is `device`'s default stream.

    torch's "current stream" is THREAD-LOCAL, and so is whatever a Holoscan `CudaStreamPool`
    hands out around `compute()` -- neither can be observed from `__init__`/`compose`, which
    runs on a different thread than the `EventBasedScheduler` worker thread that later runs
    `compute()`. Callers MUST invoke this from inside `compute()`, not from `__init__`.
    """
    with torch.cuda.device(device):
        return torch.cuda.current_stream(device) == torch.cuda.default_stream(device)


class SamOp(Operator):
    """Stage 2: detections in, masks out. Owns the SAM 2 model.

    Cross-thread stream-ordering assumption (SamOp -> PanopticOp): `SAM.predict_batch_gpu`
    (langsam_common.py) does not call `torch.cuda.synchronize()` when `timing=False` (the mode
    used below), so the device-to-device work it enqueues is only ENQUEUED when `compute()`
    emits, not necessarily complete. In the monolithic `LangSamBatchOp` this was safe by
    construction: gdino/sam/panoptic ran back-to-back on the SAME thread, hence the SAME CUDA
    stream, so later work was implicitly ordered after it. Split into operators, `PanopticOp`
    may run on a DIFFERENT thread of the `EventBasedScheduler(worker_thread_number=24)` pool.
    It remains correct ONLY because torch's and cupy's per-thread "current stream" both default
    to the shared LEGACY DEFAULT STREAM, which serialises against every other use of it --
    so PanopticOp's kernels on its own thread still wait for SamOp's enqueued work on its own
    thread. Any of the following would silently invalidate this, and the symptom would be
    intermittently corrupt or empty panoptic maps, NOT a crash:
      1. A Holoscan `CudaStreamPool` attached to this path -- assigns non-default streams
         around `compute()`, well AFTER every operator's `__init__` has already returned.
      2. Code that creates and uses an explicit non-default `cp.cuda.Stream` / `torch.cuda.Stream`.
      3. Setting `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1` (cupy's default stream becomes
         per-thread and no longer synchronises with other threads' default streams).
    What is actually checked, and where: #3 is process-global, so it is checked once, at
    construction time (`_assert_shared_default_stream_env` in `SamOp.__init__`). #1 and #2 only
    take effect once `compute()` is running on an `EventBasedScheduler` worker thread, so they
    CANNOT be observed at construction time -- they are instead checked on each of SamOp's and
    PanopticOp's first `compute()` call (`_stream_is_default`, gated by `self._stream_checked`
    so it costs one boolean test per tick after the first). That first-tick check catches a
    stream pool that is in effect for the whole run, which is the realistic failure mode; it
    would NOT catch a pool that only starts handing out non-default streams partway through a
    run. If one of the invalidating changes is ever needed, make SamOp (or PanopticOp)
    synchronise explicitly across the edge instead of removing the guard -- do NOT add an
    unconditional `synchronize()` to SamOp's hot path, since that would block its thread on its
    own GPU work and destroy the overlap this split exists to create.
    """

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
                batched_decode=bool(langsam_cfg.get("sam_batched_decode", False)),
            )
            self.sam.build_model()
            _assert_shared_default_stream_env()
        self._stream_checked = False

    def setup(self, spec: OperatorSpec):
        spec.input("det")
        spec.output("seg")

    def compute(self, op_input, op_output, context):
        if not self._stream_checked:
            if not _stream_is_default(self.device):
                raise RuntimeError(
                    f"SamOp's compute() is not running on torch's default CUDA stream on "
                    f"{self.device} (checked on its first tick). SamOp emits masks whose "
                    f"device-to-device copies are only ENQUEUED, not complete "
                    f"(predict_batch_gpu runs with timing=False, so it never synchronises), and "
                    f"PanopticOp may run on a different EventBasedScheduler worker thread than "
                    f"SamOp; correctness depends on both operators sharing the legacy default "
                    f"stream on every worker thread so PanopticOp's kernels are implicitly "
                    f"ordered after SamOp's. A Holoscan CudaStreamPool on this path, or any "
                    f"explicit non-default stream, breaks that. Fix: keep this path off "
                    f"non-default streams, or make SamOp synchronise explicitly before it "
                    f"emits.")
            self._stream_checked = True
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
    """Stage 3: masks in, packed (class<<8|instance) maps out. Owns the class-id map.

    Consumer side of the SamOp -> PanopticOp stream-ordering assumption -- see `SamOp`'s
    docstring for the full explanation. This operator's reads must stay ordered after SamOp's
    enqueued-but-not-complete writes via the shared default CUDA stream; `compute()` checks
    that on its first tick, same as SamOp (see `self._stream_checked`).
    """

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, **kwargs):
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.prompts = list(prompts)
        self._cmap = class_id_map(self.prompts)
        self.panoptic_backend = langsam_cfg.get("panoptic_backend", "cupy")
        self._stream_checked = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("seg")
        spec.output("masks")

    def compute(self, op_input, op_output, context):
        if not self._stream_checked:
            if not _stream_is_default(self.device):
                raise RuntimeError(
                    f"PanopticOp's compute() is not running on torch's default CUDA stream on "
                    f"{self.device} (checked on its first tick). PanopticOp is the CONSUMER "
                    f"side of the SamOp -> PanopticOp stream-ordering assumption described in "
                    f"SamOp's docstring: masks cross that edge as enqueued-but-not-complete GPU "
                    f"work, so this operator's reads must stay ordered after SamOp's writes via "
                    f"the shared default stream. Fix: keep this path off non-default streams, "
                    f"or make SamOp synchronise explicitly before it emits.")
            self._stream_checked = True
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
                    pmaps[i] = build_panoptic_map_auto(
                        p["masks"][k], p["sam_labels"][k], p["scores"][k],
                        self._cmap, hw[0], hw[1], backend=self.panoptic_backend)
                torch.cuda.nvtx.range_pop()
            for i, cam in enumerate(names):
                out[mask_name(cam)] = hs.as_tensor(cp.ascontiguousarray(pmaps[i]))
        op_output.emit(out, "masks")
