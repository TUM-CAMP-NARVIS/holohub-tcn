# Multi-camera LangSAM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run LangSAM (Grounding DINO + SAM2) over all color cameras, batched and split across both A40 GPUs, emitting a composite entity of full-res per-camera `uint8` class-label maps plus a tiled Holoviz view.

**Architecture:** Extract GDINO/SAM and pure helpers into `langsam_common.py`. A device-parameterized `LangSamBatchOp` (one instance per worker) self-selects its cameras from the fanned-out `color_outputs` entity, batches inference on its GPU, and emits per-camera label maps. `MaskCollectorOp` consolidates them onto GPU0; `LabelMapColorizeOp` produces a tiled view.

**Tech Stack:** Python, Holoscan SDK, PyTorch, cupy, HuggingFace transformers (Grounding DINO), SAM2, numpy.

## Global Constraints

- Class ids follow `text_prompts.prompts` order: `prompt[i]` → class `i+1`; `0` = background. Unknown labels → `0`.
- Overlapping detections resolved to one class per pixel: painted in ascending mask-score order (highest score wins).
- Label maps are full camera resolution (`[H, W]` `uint8`), one named tensor `camera0X_mask` per camera.
- Empty/missing `langsam_multicam.workers` → single worker `{device: 0, cameras: <all colorimage ports>}`.
- Reuse existing perf paths: `gdino_gpu_preprocess`, `sam_gpu_output`, bf16 autocast, `gdino_input_size`, per-stage timing.
- Pure helpers must be array-module-agnostic (`xp=np|cp`) so they unit-test on host with numpy (cupy is GPU-only / container-only).
- Host can run numpy unit tests + syntax checks only; operator/GPU/model tests run in the container.

---

### Task 1: Extract shared module `langsam_common.py`

**Files:**
- Create: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py`
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam2operator.py`

**Interfaces:**
- Produces: classes `SAM`, `GDINO`, dict `SAM_MODELS`, importable from `langsam_common`.

- [ ] **Step 1:** Move `SAM_MODELS`, `SAM`, and `GDINO` (with all their current methods: `build_model`, `predict`, `predict_batch`, `predict_batch_gpu`, `predict_gpu`, `_preprocess_image_gpu`, `_encode_text`, `_resize_hw`, `_init_preprocess`, `_maybe_override_size`, `_maybe_compile`, `_autocast`, `_sync`) verbatim from `langsam2operator.py` into `langsam_common.py`, along with the imports they need (`torch`, `cupy as cp`, `numpy as np`, `time`, `matplotlib.pyplot as plt`, `PIL.Image`, `hydra`, `omegaconf`, `transformers`, `sam2.*`).
- [ ] **Step 2:** In `langsam2operator.py`, delete those moved definitions and add `from langsam_common import SAM, GDINO, SAM_MODELS`.
- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile langsam_common.py langsam2operator.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam2operator.py
git commit -m "refactor(tcn_artekmed): extract GDINO/SAM into langsam_common"
```

---

### Task 2: Pure helper `resolve_workers`

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py`
- Test: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_langsam_multicam.py`

**Interfaces:**
- Produces: `resolve_workers(multicam_cfg: dict | None, all_color_cameras: list[str]) -> list[dict]` where each dict is `{"device": int, "cameras": list[str]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_langsam_multicam.py
from langsam_common import resolve_workers

ALL = ["camera01_colorimage", "camera02_colorimage", "camera03_colorimage"]

def test_empty_config_defaults_to_single_gpu0_worker():
    assert resolve_workers(None, ALL) == [{"device": 0, "cameras": ALL}]
    assert resolve_workers({}, ALL) == [{"device": 0, "cameras": ALL}]
    assert resolve_workers({"workers": []}, ALL) == [{"device": 0, "cameras": ALL}]

def test_explicit_workers_preserved_and_device_coerced():
    cfg = {"workers": [
        {"device": "0", "cameras": ["camera01_colorimage"]},
        {"device": 1, "cameras": ["camera02_colorimage", "camera03_colorimage"]},
    ]}
    out = resolve_workers(cfg, ALL)
    assert out == [
        {"device": 0, "cameras": ["camera01_colorimage"]},
        {"device": 1, "cameras": ["camera02_colorimage", "camera03_colorimage"]},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m pytest tests/test_langsam_multicam.py -v`
Expected: FAIL (ImportError: cannot import name 'resolve_workers').

- [ ] **Step 3: Implement**

```python
# in langsam_common.py
def resolve_workers(multicam_cfg, all_color_cameras):
    """Resolve the per-GPU worker assignment.

    Empty/missing 'workers' -> a single worker on device 0 with all color cameras.
    """
    workers = (multicam_cfg or {}).get("workers") or []
    if not workers:
        return [{"device": 0, "cameras": list(all_color_cameras)}]
    return [
        {"device": int(w.get("device", 0)), "cameras": list(w.get("cameras") or [])}
        for w in workers
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_langsam_multicam.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_langsam_multicam.py
git commit -m "feat(tcn_artekmed): add resolve_workers config helper"
```

---

### Task 3: Pure helpers `class_id_map` / `class_id_for_label`

**Files:**
- Modify: `langsam_common.py`
- Test: `tests/test_langsam_multicam.py`

**Interfaces:**
- Produces: `class_id_map(prompts: list[str]) -> dict[str, int]` (normalized prompt → 1-based id); `class_id_for_label(label: str, cmap: dict[str,int]) -> int` (exact → substring → 0).

- [ ] **Step 1: Write the failing test**

```python
from langsam_common import class_id_map, class_id_for_label

PROMPTS = ["floor", "person", "robot"]

def test_class_id_map_is_one_based():
    assert class_id_map(PROMPTS) == {"floor": 1, "person": 2, "robot": 3}

def test_class_id_for_label_exact_substring_and_unknown():
    cmap = class_id_map(PROMPTS)
    assert class_id_for_label("Floor", cmap) == 1          # case-insensitive exact
    assert class_id_for_label("a person", cmap) == 2       # substring (prompt in label)
    assert class_id_for_label("lamp", cmap) == 0           # unknown -> background
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_langsam_multicam.py -k class_id -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# in langsam_common.py
def class_id_map(prompts):
    """Normalized prompt -> 1-based class id (0 reserved for background)."""
    return {str(p).strip().lower(): i + 1 for i, p in enumerate(prompts)}

def class_id_for_label(label, cmap):
    """Class id for a detected label: exact match, then substring, else 0 (background)."""
    key = str(label).strip().lower()
    if key in cmap:
        return cmap[key]
    for known, cid in cmap.items():
        if known and (known in key or key in known):
            return cid
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_langsam_multicam.py -k class_id -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add class-id mapping helpers"
```

---

### Task 4: Pure helper `build_label_map` (array-agnostic)

**Files:**
- Modify: `langsam_common.py`
- Test: `tests/test_langsam_multicam.py`

**Interfaces:**
- Produces: `build_label_map(masks, labels, scores, cmap, height, width, xp=np) -> xp.ndarray` — `(H,W)` `uint8` label map; higher-score class wins overlaps; `masks` is `(M,H,W)` truthy-per-pixel.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from langsam_common import class_id_map, build_label_map

def test_build_label_map_higher_score_wins_overlap():
    cmap = class_id_map(["floor", "person"])
    H = W = 4
    m_floor = np.zeros((H, W), bool); m_floor[0:3, 0:3] = True   # floor, low score
    m_person = np.zeros((H, W), bool); m_person[1:4, 1:4] = True  # person, high score
    masks = np.stack([m_floor, m_person])
    lm = build_label_map(masks, ["floor", "person"], np.array([0.5, 0.9]),
                         cmap, H, W, xp=np)
    assert lm.dtype == np.uint8
    assert lm[0, 0] == 1                 # floor only
    assert lm[3, 3] == 2                 # person only
    assert lm[2, 2] == 2                 # overlap -> higher score (person) wins
    assert lm[3, 0] == 0                 # background

def test_build_label_map_empty_is_all_background():
    cmap = class_id_map(["floor"])
    lm = build_label_map(np.empty((0, 4, 4)), [], np.array([]), cmap, 4, 4, xp=np)
    assert lm.shape == (4, 4) and lm.max() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_langsam_multicam.py -k build_label -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# in langsam_common.py  (np imported at module top)
def build_label_map(masks, labels, scores, cmap, height, width, xp=np):
    """(M,H,W) masks + labels + scores -> (H,W) uint8 class-label map.

    Detections are painted in ascending score order so the highest-confidence class wins
    on overlap. Unknown labels (class id 0) are skipped.
    """
    label_map = xp.zeros((height, width), dtype=xp.uint8)
    if masks is None or len(masks) == 0:
        return label_map
    order = xp.argsort(scores)
    order = [int(i) for i in (order.tolist() if hasattr(order, "tolist") else order)]
    for j in order:
        cid = class_id_for_label(labels[j], cmap)
        if cid == 0:
            continue
        label_map[masks[j] > 0] = cid
    return label_map
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_langsam_multicam.py -k build_label -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add build_label_map helper"
```

---

### Task 5: Batched GDINO — `GDINO.predict_gpu_batch`

**Files:**
- Modify: `langsam_common.py`

**Interfaces:**
- Consumes: existing `GDINO._preprocess_image_gpu`, `GDINO._encode_text`, `self.model`, `self.processor`.
- Produces: `GDINO.predict_gpu_batch(images_gpu: list, box_threshold, text_threshold, orig_hw: tuple) -> list[dict]` — one `{boxes, labels, scores}`-style HF result per image (post-processed), where `images_gpu[i]` is an `(H,W,C)` uint8 CUDA tensor.

- [ ] **Step 1: Implement**

```python
# method on GDINO in langsam_common.py
def predict_gpu_batch(self, images_gpu, box_threshold, text_threshold, orig_hw):
    """Batched GPU detection. images_gpu: list of (H,W,C>=3) uint8 CUDA tensors, all the
    same spatial size. Returns a per-image list of HF post-processed results."""
    enc = self._encode_text(texts_prompt=self._last_texts)  # set by caller; see note
    pixel_values = torch.cat([self._preprocess_image_gpu(im) for im in images_gpu], dim=0)
    n = pixel_values.shape[0]
    input_ids = enc.input_ids.repeat(n, 1)
    attention_mask = enc.attention_mask.repeat(n, 1)
    tti = enc.get("token_type_ids")
    token_type_ids = tti.repeat(n, 1) if tti is not None else None
    use_amp = self.device is not None and self.device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=self.device.type if self.device is not None else "cpu",
        dtype=torch.bfloat16, enabled=use_amp,
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
    return self.processor.post_process_grounded_object_detection(
        outputs, input_ids, box_threshold,
        text_threshold=text_threshold, target_sizes=[orig_hw] * n,
    )
```

Note: to keep `_encode_text` reusable, change `predict_gpu`/`predict_gpu_batch` to take `texts_prompt` and call `self._encode_text(texts_prompt)` directly (it already caches). Update `predict_gpu_batch` to accept `texts_prompt` and drop the `self._last_texts` shim:

```python
def predict_gpu_batch(self, images_gpu, texts_prompt, box_threshold, text_threshold, orig_hw):
    enc = self._encode_text(texts_prompt)
    ...
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_common.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add batched GDINO predict_gpu_batch"
```

---

### Task 6: `LangSamBatchOp` (batched, device-parameterized)

**Files:**
- Create: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py`

**Interfaces:**
- Consumes: `SAM`, `GDINO`, `resolve_workers`, `class_id_map`, `build_label_map` from `langsam_common`.
- Produces: `LangSamBatchOp(fragment, name, cameras, device, langsam_cfg, prompts)` operator with input port `color_input`, output port `masks` emitting `{"<cam>_mask": cupy uint8 (H,W)}` on `device`; also `self.gdino`, `self.sam`.

- [ ] **Step 1: Implement the operator**

```python
import logging
import cupy as cp
import torch
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from langsam_common import (SAM, GDINO, class_id_map, build_label_map)

log = logging.getLogger(__name__)


def _mask_name(cam_port):
    # "camera01_colorimage" -> "camera01_mask"
    return cam_port.replace("_colorimage", "") + "_mask"


class LangSamBatchOp(Operator):
    """Batched LangSAM over a subset of cameras on one GPU."""

    def __init__(self, fragment, *args, cameras, device, langsam_cfg, prompts, **kwargs):
        self.cameras = list(cameras)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.prompts = list(prompts)
        self._cmap = class_id_map(self.prompts)
        self.box_threshold = float(langsam_cfg.get("box_threshold", 0.3))
        self.text_threshold = float(langsam_cfg.get("text_threshold", 0.25))
        super().__init__(fragment, *args, **kwargs)
        with torch.cuda.device(self.device):
            self.sam = SAM(langsam_cfg.get("sam_type", "sam2.1_hiera_tiny"),
                           langsam_cfg.get("sam_ckpt_path"), device=self.device,
                           compile_model=bool(langsam_cfg.get("sam_compile", False)))
            self.sam.build_model()
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
        spec.output("masks")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("color_input")
        out = {}
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            rgb_gpu, rgb_np, names, hw = [], [], [], None
            for cam in self.cameras:
                t = msg.get(cam)
                if t is None:
                    log.warning(f"LangSamBatchOp: missing tensor {cam}")
                    continue
                img = torch.from_dlpack(cp.asarray(t))          # (H,W,4) BGRA uint8, cuda:0
                img = img.to(self.device)[..., [2, 1, 0]].contiguous()  # -> RGB on device
                rgb_gpu.append(img)
                rgb_np.append(cp.asnumpy(cp.from_dlpack(img)))  # host copy for SAM API
                names.append(cam)
                hw = (int(img.shape[0]), int(img.shape[1]))
            if not rgb_gpu:
                op_output.emit(out, "masks")
                return

            gres = self.gdino.predict_gpu_batch(
                rgb_gpu, self.prompts, self.box_threshold, self.text_threshold, hw)
            boxes_per_img, labels_per_img = [], []
            for r in gres:
                r = {k: (v.detach().float().cpu().numpy() if hasattr(v, "numpy") else v)
                     for k, v in r.items()}
                labels = r.get("text_labels", r.get("labels", []))
                labels = [str(x) for x in (labels.tolist() if hasattr(labels, "tolist") else labels)]
                boxes_per_img.append(r.get("boxes"))
                labels_per_img.append(labels)

            masks, mscores, _ = self.sam.predict_batch_gpu(
                rgb_np, xyxy=[b for b in boxes_per_img], timing=False)

            for i, cam in enumerate(names):
                lm = build_label_map(masks[i], labels_per_img[i], mscores[i],
                                     self._cmap, hw[0], hw[1], xp=cp)
                out[_mask_name(cam)] = hs.as_tensor(lm)
        op_output.emit(out, "masks")
```

Note: `predict_batch_gpu` currently skips images with no boxes upstream; here we pass all images and rely on it returning a per-image mask list aligned with input order. If an image has zero boxes, `build_label_map` receives empty masks and returns an all-background map — verify this path in the container.

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_multicam_fragment.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py
git commit -m "feat(tcn_artekmed): add batched LangSamBatchOp"
```

---

### Task 7: `MaskCollectorOp`

**Files:**
- Modify: `langsam_multicam_fragment.py`

**Interfaces:**
- Produces: `MaskCollectorOp` with multi-receiver input `receivers`, output `masks`; consolidates all worker label maps onto GPU 0 and emits one composite dict `{"<cam>_mask": cupy uint8 (H,W) on cuda:0}`.

- [ ] **Step 1: Implement**

```python
class MaskCollectorOp(Operator):
    """Merge per-worker label maps into one composite entity, all on GPU 0."""

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=hs.core.IOSpec.ANY_SIZE)
        spec.output("masks")

    def compute(self, op_input, op_output, context):
        messages = op_input.receive("receivers")   # list of dicts, one per worker
        out = {}
        with cp.cuda.Device(0):
            for msg in messages:
                if msg is None:
                    continue
                for name in msg.keys():
                    arr = cp.asarray(msg.get(name))
                    if arr.device.id != 0:
                        # cross-GPU copy via torch device-to-device
                        t = torch.from_dlpack(arr).to("cuda:0")
                        arr = cp.from_dlpack(t).copy()
                    out[name] = hs.as_tensor(arr)
        op_output.emit(out, "masks")
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_multicam_fragment.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add MaskCollectorOp"
```

---

### Task 8: `LabelMapColorizeOp`

**Files:**
- Modify: `langsam_multicam_fragment.py`

**Interfaces:**
- Produces: `LabelMapColorizeOp(fragment, name, num_classes)` with input `masks`, output `viz` emitting `{"<cam>_mask": cupy uint8 (H,W,4) RGBA on cuda:0}` colorized via a tab20-based LUT; class 0 -> transparent.

- [ ] **Step 1: Implement**

```python
import numpy as np
import matplotlib.pyplot as plt

class LabelMapColorizeOp(Operator):
    def __init__(self, fragment, *args, num_classes, alpha=180, **kwargs):
        pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3] * 255
        lut = np.zeros((num_classes + 1, 4), np.uint8)          # 0 = transparent bg
        for c in range(1, num_classes + 1):
            lut[c, :3] = pal[(c - 1) % 20]
            lut[c, 3] = alpha
        self._lut = cp.asarray(lut)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("masks")
        spec.output("viz")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("masks")
        out = {}
        with cp.cuda.Device(0):
            for name in msg.keys():
                lm = cp.asarray(msg.get(name))          # (H,W) uint8
                rgba = self._lut[lm]                     # (H,W,4) uint8
                out[name] = hs.as_tensor(cp.ascontiguousarray(rgba))
        op_output.emit(out, "viz")
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_multicam_fragment.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add LabelMapColorizeOp"
```

---

### Task 9: `LangSamMultiCamProcessingSubgraph`

**Files:**
- Modify: `langsam_multicam_fragment.py`

**Interfaces:**
- Consumes: `resolve_workers`, `LangSamBatchOp`, `MaskCollectorOp`, `LabelMapColorizeOp`.
- Produces: `LangSamMultiCamProcessingSubgraph(fragment, name, kwargs, all_color_cameras)` with input interface port `input` (→ every worker's `color_input`), output interface ports `output_masks` (collector `masks`) and `output_viz` (colorize `viz`).

- [ ] **Step 1: Implement**

```python
from holoscan.core import Subgraph

class LangSamMultiCamProcessingSubgraph(Subgraph):
    def __init__(self, fragment, name, kwargs, all_color_cameras):
        self.kwargs = kwargs
        self.all_color_cameras = list(all_color_cameras)
        super().__init__(fragment, name)

    def _n(self, s):
        return f"{self.name}_{s}"

    def compose(self):
        multicam_cfg = self.kwargs("langsam_multicam") if _has_key(self.kwargs, "langsam_multicam") else {}
        langsam_cfg = self.kwargs("langsam_inference")
        prompts = self.kwargs("text_prompts").get("prompts", [])
        workers = resolve_workers(multicam_cfg, self.all_color_cameras)

        collector = MaskCollectorOp(self, name=self._n("collector"))
        colorize = LabelMapColorizeOp(self, name=self._n("colorize"), num_classes=len(prompts))
        self.add_flow(collector, colorize, {("masks", "masks")})

        for i, w in enumerate(workers):
            op = LangSamBatchOp(self, name=self._n(f"worker{i}"),
                                cameras=w["cameras"], device=w["device"],
                                langsam_cfg=langsam_cfg, prompts=prompts)
            self.add_flow(op, collector, {("masks", "receivers")})
            self.add_input_interface_port("input", op, "color_input")

        self.add_output_interface_port("output_masks", collector, "masks")
        self.add_output_interface_port("output_viz", colorize, "viz")
```

Add a `_has_key` helper that returns True if the YAML has the section (guard for the default):

```python
def _has_key(kwargs_fn, key):
    try:
        return bool(kwargs_fn(key)) or kwargs_fn(key) == {}
    except Exception:
        return False
```

Note: verify in the container how `self.kwargs("langsam_multicam")` behaves when the section is absent; if it raises, the `_has_key` guard returns `{}` so `resolve_workers` applies the single-GPU default.

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_multicam_fragment.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): add LangSamMultiCamProcessingSubgraph"
```

---

### Task 10: App wiring + YAML config

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.py`
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml`

**Interfaces:**
- Consumes: `LangSamMultiCamProcessingSubgraph`.

- [ ] **Step 1: YAML** — add the config block and an enable flag:

```yaml
camera_stream_processing:
  ...
  enable_langsam_multicam: True

langsam_multicam:
  workers:
    - device: 0
      cameras: ["camera01_colorimage", "camera02_colorimage"]
    - device: 1
      cameras: ["camera03_colorimage", "camera04_colorimage", "camera05_colorimage"]

langsam_multicam_holoviz:
  width: 1920
  height: 1080
  framerate: 30
```

- [ ] **Step 2: App wiring** — in `App.compose`, after the subscriber/`color_streams_config` are built, add:

```python
from langsam_multicam_fragment import LangSamMultiCamProcessingSubgraph

if camera_streams_config.get("enable_langsam_multicam", False):
    all_color_cams = [c["name"] for c in color_streams_config]
    mc = LangSamMultiCamProcessingSubgraph(self, "langsam_multicam", self.kwargs, all_color_cams)
    mc_holoviz = HolovizOp(self, allocator=device_memory_pool, name="langsam_multicam_holoviz",
                           window_title="LangSAM Multi-Camera Masks",
                           **self.kwargs("langsam_multicam_holoviz"))
    self.add_flow(subscriber_op, mc, {("color_outputs", "input")})
    self.add_flow(mc, mc_holoviz, {("output_viz", "receivers")})
    have_camera_consumer = True
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile tcn_shm_vlm_inference.py && python3 -c "import yaml; yaml.safe_load(open('tcn_shm_vlm_inference.yaml'))"`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): wire multi-camera LangSAM into the app"
```

---

### Task 11: Container smoke run + timing

**Files:** none (verification).

- [ ] **Step 1:** Run the app in the container with the multi-cam path enabled (single-worker default first — comment out `workers` — then the 2-GPU split). Watch startup for model loads on each device and the Holoviz tiled window.
- [ ] **Step 2:** Confirm the tiled mask view shows all cameras and that `output_masks` carries `camera0X_mask` uint8 tensors at full resolution.
- [ ] **Step 3:** Enable `timing: true` in `langsam_inference`; confirm each worker logs its GDINO/SAM timing; compare aggregate throughput single-GPU vs split.
- [ ] **Step 4:** If the cuda:1 worker errors on cross-GPU dlpack/stream handling, add explicit `torch.cuda.synchronize()` around the P2P copies and re-test.

---

## Self-Review notes

- **Spec coverage:** config + default (Tasks 2, 9, 10), class-label output (Tasks 3–4, 6), batched GDINO/SAM (Tasks 5–6), collector/GPU consolidation (Task 7), colorized tiled viz (Tasks 8, 10), device handling (Task 6/7), staged delivery (single-worker default validated in Task 11 before split). All covered.
- **Known verification risks (flagged in-task, must be checked in container):** (a) `predict_batch_gpu` behavior when an image has zero boxes; (b) `self.kwargs()` on an absent YAML section; (c) cross-GPU dlpack/stream correctness for the cuda:1 worker; (d) Holoscan multi-receiver `receivers` returning a list of per-worker dicts. These cannot be exercised on host (no GPU/models/Holoscan runtime).
