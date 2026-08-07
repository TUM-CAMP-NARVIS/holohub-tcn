# Pipelined LangSAM operators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Split `LangSamBatchOp` into `GdinoOp → SamOp → PanopticOp` so a worker's stages overlap across frames, opt-in via `gpu_workers.pipelined`.

**Spec:** [`../specs/2026-08-07-langsam-pipelining-design.md`](../specs/2026-08-07-langsam-pipelining-design.md)
**Flow being preserved:** [`../dataflow-and-pipelining-roadmap.md`](../dataflow-and-pipelining-roadmap.md) Part 1.

## Global Constraints

- **Behaviour-neutral refactor.** The pipelined path must produce byte-identical masks to the monolithic one. Do not fix bugs, simplify round-trips, or change `hw` semantics along the way — several are deliberately preserved (spec §5).
- **Boxes and masks never leave the GPU.**
- The operators import torch/cupy/holoscan, so they **cannot** be host-tested. Host verification is `py_compile` plus greps; correctness comes from the container A/B.
- Host suites must stay green: `test_gpu_workers` 8/8, `test_prompt_remap` 11/11, `test_gdino_postprocess` 7/7, `test_langsam_multicam` 9/9 (run from `applications/tcn_artekmed/tcn_shm_vlm_inference/python`).
- Commit style `feat(tcn_artekmed): ...` ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`; never `git add -A`.
- **Baselines:** gdino 76.3/69.1, sam 60.4/86.9, panoptic 6.9/6.9 ms (GPU 0 / GPU 1); Σ 143.6/162.8; period 177.2–180.7 ms; 5.54–5.64 fps.

---

### Task N1: The three operators

**Files:** Create `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_pipelined.py`

**Interfaces produced:** `GdinoOp`, `SamOp`, `PanopticOp`, all constructed with the same
`(fragment, *, cameras, device, langsam_cfg, prompts, **kwargs)` shape as `LangSamBatchOp`.

- [ ] **Step 1: Write the module**

```python
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
```

- [ ] **Step 2: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_pipelined.py && echo "compile OK"
grep -c "nvtx.range_push" langsam_pipelined.py     # expect 3 — one per stage, names unchanged
grep -n "worker_engine_path\|class_id_map\|build_panoptic_map" langsam_pipelined.py | head
```
Expected: `compile OK`; exactly 3 NVTX pushes. If `worker_engine_path` or `class_id_map` is not
re-exported by `langsam_common`, import it from `langsam_helpers` instead — check before assuming
(a previous task in this repo was tripped by exactly that).

- [ ] **Step 3: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_pipelined.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): GdinoOp/SamOp/PanopticOp as separate pipelined stages

LangSamBatchOp runs its three stages sequentially inside one compute(), so a worker's
tick costs the sum of them. As separate operators the event-based scheduler can
overlap stage N of frame k with stage N-1 of frame k+1, and the period tends toward
the max stage rather than the sum.

Behaviour-neutral: same stage boundaries, same NVTX names, same outputs, including
preserved quirks (hw is the last camera's shape). Not yet wired up.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task N2: Wire it up behind the toggle

**Files:**
- Modify: `python/langsam_multicam_fragment.py` (`compose` only)
- Modify: `python/tcn_shm_vlm_inference.yaml`

- [ ] **Step 1: YAML**

Add to the `gpu_workers` node, directly under `gpu_workers:` and above `workers:`:

```yaml
  # Run each worker's gdino/sam/panoptic as three chained operators instead of one, so stages
  # overlap across frames (period tends toward max(stage) rather than sum(stage)). Costs ~2 ticks
  # of extra mask latency. false -> the original single LangSamBatchOp per worker.
  # See docs/specs/2026-08-07-langsam-pipelining-design.md.
  pipelined: false
```

- [ ] **Step 2: Branch in `compose`**

In `langsam_multicam_fragment.py`, add to the imports at the top:

```python
from langsam_pipelined import GdinoOp, SamOp, PanopticOp
```

Then replace the worker loop body. Currently:

```python
        for i, w in enumerate(workers):
            op = LangSamBatchOp(
                self, name=self._n(f"worker{i}"),
                cameras=w["cameras"], device=w["device"],
                langsam_cfg=langsam_cfg, prompts=prompts,
            )
            self.add_flow(op, collector, {("masks", "receivers")})
            self.add_input_interface_port("input", op, "color_input")
```

with:

```python
        pipelined = bool(multicam_cfg.get("pipelined", False))
        log.info(f"LangSAM worker structure: {'pipelined (3 ops)' if pipelined else 'monolithic'}")
        for i, w in enumerate(workers):
            kw = dict(cameras=w["cameras"], device=w["device"],
                      langsam_cfg=langsam_cfg, prompts=prompts)
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
```

- [ ] **Step 3: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python
python3 -m py_compile langsam_multicam_fragment.py langsam_pipelined.py && echo "compile OK"
python3 -c "
import yaml; c=yaml.safe_load(open('tcn_shm_vlm_inference.yaml'))['gpu_workers']
print('pipelined:', c.get('pipelined'))
print('workers  :', [(w['device'], len(w['cameras'])) for w in c['workers']])"
python3 tests/test_gpu_workers.py && python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py && python3 tests/test_langsam_multicam.py
```
Expected: `compile OK`; `pipelined: False`; workers `[(0, 2), (1, 3)]`; suites 8/8, 11/11, 7/7, 9/9.

- [ ] **Step 4: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): gpu_workers.pipelined selects the 3-operator worker

compose builds either one LangSamBatchOp per worker (default, unchanged) or the
GdinoOp -> SamOp -> PanopticOp chain. Opt-in so the two can be A/B'd on one config
line and reverted instantly, following the gdino_backend/sam_backend precedent.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task N3: Validate and measure

Verification only; needs the container. **Correctness before performance.**

- [ ] **Step 1: `pipelined: false` still works** — run the app as-is, confirm masks unchanged and
  the log says `monolithic`. This proves the refactor did not disturb the existing path.

- [ ] **Step 2: Flip to `pipelined: true`, confirm masks are IDENTICAL** — same scene, compare
  against step 1 visually. This is the correctness gate; the operators cannot be host-tested.
  Startup should log `pipelined (3 ops)` and construct three operators per worker.

- [ ] **Step 3: Profile and compare.** Use the roadmap's Part 2/§3 recipes. Report:
  - per-stage NVTX wall vs the baselines (gdino 76.3/69.1, sam 60.4/86.9, panoptic 6.9/6.9)
  - tick period vs 177.2–180.7 ms and fps vs 5.54–5.64
  - **whether the stages actually overlap** — the point of the exercise. Compute the pairwise
    overlap of `gdino`/`sam` busy intervals *within one worker*, the same way worker-vs-worker
    overlap was computed for the 2-GPU split (playbook §2.4).
  - GPU utilisation vs 47.8 / 56.5%.
  - Expected floors: **103.8 ms** (GPU) and **115.2 ms** (pessimistic all-GIL).

- [ ] **Step 4: Measure the latency cost** with `--tracking`, and record it. The design accepted
  ~2 extra ticks; confirm what it actually is.

- [ ] **Step 5: Run roadmap step 1 Recipe B** (py-spy GIL sampling, in the roadmap) against the
  *pipelined* app. If the period landed near the 115 ms GIL floor rather than the 104 ms GPU
  floor, this says so directly and tells you whether steps 2 and 4 would unlock more.

- [ ] **Step 6: Record.** Append `## Results` to the spec; update the playbook's §4 arc and the
  roadmap's step 3 with the outcome. If the win holds, flip the yaml default to `true`; if not,
  leave it `false` and write down what the GIL actually cost.

---

## Self-Review notes

- **Spec coverage:** §1 module/three ops → N1; §2 payload keys → N1's emit/receive; §3 model
  ownership and the prompt invariant → N1 (`GdinoOp` owns detector+prompts, `PanopticOp` owns
  `_cmap`); §4 toggle → N2; §5 behaviour-neutrality → N1 (preserved `hw`, NVTX names, empty-case
  emits). Testing table → N1 Step 2, N2 Step 3, N3 Steps 1–4.
- **Payload key consistency:** `names`, `hw`, `sam_idx`, `sam_imgs`, `sam_boxes`, `sam_labels`
  emitted by `GdinoOp`; `SamOp` reads exactly those and emits `names`, `hw`, `sam_idx`,
  `sam_labels`, `masks`, `scores`; `PanopticOp` reads exactly those. Checked by hand across the
  three `compute` bodies.
- **Empty paths:** three of them — no cameras (`GdinoOp` emits empty names; `PanopticOp` emits
  `{}`), no detections (`SamOp` short-circuits with empty `masks`/`scores`; `PanopticOp` still
  emits an all-background map per camera), and both.
- **Ordering:** N1 → N2 → N3.
