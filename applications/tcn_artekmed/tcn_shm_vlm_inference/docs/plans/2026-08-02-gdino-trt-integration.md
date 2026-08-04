# Grounding DINO TensorRT Integration — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the launch-bound PyTorch Grounding DINO forward with a prebuilt TensorRT engine, run behind a `gdino_backend` toggle, cutting per-frame detection latency (~230 ms → ~40 ms/img, spike-measured) with no detection-quality change.

**Architecture:** An offline tool exports GroundingDINO-T → fixed-resolution ONNX → TF32 TRT engine and precomputes the fixed-prompt text tensors, gated by a PyTorch-vs-engine parity check. A runtime `GDinoTrtDetector` loads the engine + text tensors, runs one `execute_async_v3` per camera frame, and a pure `gdino_postprocess` turns raw outputs into `(box, class, score)`. `LangSamBatchOp`/`LangSAM2Operator` select the backend via config; the SAM path is unchanged.

**Tech Stack:** Python, TensorRT 11 (strongly-typed), PyTorch, cupy, Holoscan, IDEA-Research GroundingDINO (Swin-T), numpy, onnx.

## Global Constraints

- Precision: **TF32** for v1 (spike parity IoU 0.9997); FP16 only behind the same parity gate.
- Batch: **batch-1 engine, looped per camera** in v1 (engine shape fixed at build).
- Prompts: **fixed, baked**; text tensors precomputed offline; prompt change ⇒ offline re-export.
- **Parity gate:** an engine is only accepted if PyTorch-vs-engine top-box **IoU ≥ 0.99** on a real image.
- Runtime: **custom Python TRT op** (one `execute_async_v3`), not `InferenceOp`.
- Model: IDEA-Research `groundingdino_swint_ogc.pth` (Swin-T); default build `(H,W)=(512,672)`.
- Engine + text artifacts live at `/srv/models/active/groundingdino/` (host `/data/models/...`).
- Export env quirks (from the spike): transformers **4.x**; `torch.onnx.export(..., dynamo=False)`; ONNX is fixed to its traced resolution; TRT 11 has **no `FP16`/`EXPLICIT_BATCH` builder flags** (TF32 via `BuilderFlag.TF32`, `create_network(0)`).
- Fallback: `gdino_backend: "pytorch"` reproduces the current path exactly.
- Class-id contract unchanged: class id = `text_prompts.prompts` order + 1 (0 = background), consistent with the panoptic map.
- Host can run numpy unit tests + syntax checks only; engine/GPU/Holoscan steps verify in the container.

---

### Task 1: Offline export + build + parity tool

**Files:**
- Create: `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py`
- Create: `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md`

**Interfaces:**
- Produces (on disk): `<out>/gdino_swint_<H>x<W>_tf32.engine`, and `<out>/gdino_swint_prompts.npz`
  containing `input_ids,attention_mask,position_ids,token_type_ids,text_token_mask` (all `(1,L)`
  / `(1,L,L)`), `token_class_ids` `(256,)` int (text-token position → class id, 0 otherwise),
  and `prompts` (list[str]) for runtime verification.

- [ ] **Step 1: Consolidate the spike scripts into one tool.** Adapt the working spike scripts
  (`export_onnx.py` patched with `dynamo=False` + `img=(H,W)`, `gdino_trt_bench.py` build,
  `gdino_trt_realparity.py`) in `/home/ecku/develop/vision/GroundingDINO-TensorRT-and-ONNX-Inference/`
  into a single `gdino_trt_export.py` with CLI: `--checkpoint --config --prompts "floor. person." --hw 512 672 --out <dir>`.
  It must: (a) build the caption from `--prompts`, tokenize, and derive `text_token_mask`/`position_ids`
  (reuse `get_caption_mask.py` logic) and `token_class_ids`; (b) export the fixed-resolution ONNX
  (`dynamo=False`, opset 17, CPU); (c) build the TF32 engine (`create_network(0)`,
  `BuilderFlag.TF32`, fixed optimization profile); (d) save the `.npz`.

- [ ] **Step 2: Add the parity gate.** After building, run the real-image parity (PyTorch CPU vs
  engine GPU) from `gdino_trt_realparity.py`; compute top-box IoU; **raise SystemExit if IoU < 0.99**.
  Print the IoU and both boxes/scores.

- [ ] **Step 3: Write `gdino_trt_export.md`** documenting: the transformers-4.x / `dynamo=False`
  requirements, the exact command used for our config, the parity-gate meaning, and where the
  artifacts are copied (`/data/models/active/groundingdino/`).

- [ ] **Step 4: Produce the artifacts.** Run the tool for `--prompts "floor. person." --hw 512 672`;
  confirm the parity gate passes (IoU ≥ 0.99) and copy the `.engine` + `.npz` to
  `/data/models/active/groundingdino/`.

Run: `python gdino_trt_export.py --checkpoint weights/groundingdino_swint_ogc.pth --config groundingdino/config/GroundingDINO_SwinT_OGC.py --prompts "floor. person." --hw 512 672 --out /data/models/active/groundingdino`
Expected: `... parity IoU=0.99xx OK` and two files written.

- [ ] **Step 5: Commit** the tool + doc (not the large binary artifacts).

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md
git commit -m "feat(tcn_artekmed): GDINO->ONNX->TRT export tool with parity gate"
```

---

### Task 2: Pure lean post-process (`gdino_postprocess`) + host tests

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py`
- Test: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_gdino_postprocess.py`

**Interfaces:**
- Produces: `gdino_postprocess(logits, boxes, token_class_ids, num_classes, box_threshold, img_h, img_w, xp=np) -> (boxes_xyxy, class_ids, scores)` where inputs are `logits (Q,256)`, `boxes (Q,4)` cxcywh-normalized, `token_class_ids (256,)`; outputs are `(N,4)` pixel xyxy, `(N,)` int class ids (1-based), `(N,)` float scores, for queries whose best per-class score exceeds `box_threshold`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gdino_postprocess.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import gdino_postprocess

def _sig_inv(p):  # logit for a target prob
    return np.log(p / (1 - p))

def test_gdino_postprocess_thresholds_classes_and_scales():
    # 256 text tokens; tokens 1,2 -> class 1 (floor); token 4 -> class 2 (person); rest 0
    tcid = np.zeros(256, np.int64); tcid[1] = 1; tcid[2] = 1; tcid[4] = 2
    Q = 3
    logits = np.full((Q, 256), _sig_inv(0.01), np.float32)   # baseline low
    logits[0, 1] = _sig_inv(0.90)     # query0 -> floor, score .90
    logits[1, 4] = _sig_inv(0.80)     # query1 -> person, score .80
    logits[2, 2] = _sig_inv(0.10)     # query2 -> floor .10 (below threshold)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2],
                      [0.25, 0.25, 0.1, 0.1],
                      [0.9, 0.9, 0.1, 0.1]], np.float32)
    bx, cls, sc = gdino_postprocess(logits, boxes, tcid, num_classes=2,
                                    box_threshold=0.3, img_h=100, img_w=200, xp=np)
    assert list(cls) == [1, 2]                       # query2 dropped
    assert np.allclose(sc, [0.90, 0.80], atol=1e-4)
    # query0 box cxcywh (.5,.5,.2,.2) on 200x100 -> xyxy pixels
    assert np.allclose(bx[0], [80, 40, 120, 60], atol=1e-3)   # (0.4..0.6)*200, (0.4..0.6)*100

def test_gdino_postprocess_empty_when_all_below():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    logits = np.full((5, 256), _sig_inv(0.05), np.float32)
    boxes = np.tile(np.array([0.5, 0.5, 0.1, 0.1], np.float32), (5, 1))
    bx, cls, sc = gdino_postprocess(logits, boxes, tcid, 1, 0.3, 100, 100, xp=np)
    assert len(bx) == 0 and len(cls) == 0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try: fn(); print("PASS", fn.__name__)
        except AssertionError as e: bad += 1; print("FAIL", fn.__name__, repr(e))
    print(f"{len(fns)-bad}/{len(fns)} passed"); raise SystemExit(1 if bad else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gdino_postprocess.py`
Expected: FAIL (ImportError: cannot import name 'gdino_postprocess').

- [ ] **Step 3: Implement**

```python
# in langsam_helpers.py
def gdino_postprocess(logits, boxes, token_class_ids, num_classes,
                      box_threshold, img_h, img_w, xp=np):
    """Grounding DINO outputs -> detections, using a fixed token->class map.

    logits (Q,256), boxes (Q,4) cxcywh in [0,1], token_class_ids (256,) with class id
    (1..num_classes) for prompt tokens else 0. Returns (boxes_xyxy_px, class_ids, scores)
    for queries whose best per-class score > box_threshold.
    """
    probs = 1.0 / (1.0 + xp.exp(-logits))                       # (Q,256)
    tcid = xp.asarray(token_class_ids)
    Q = probs.shape[0]
    class_score = xp.zeros((Q, num_classes + 1), dtype=probs.dtype)  # col 0 unused
    for c in range(1, num_classes + 1):
        mask = (tcid == c)
        if bool(mask.any()):
            class_score[:, c] = probs[:, mask].max(axis=1)
    best_cls = class_score[:, 1:].argmax(axis=1) + 1            # (Q,) 1-based
    best_score = class_score[xp.arange(Q), best_cls]            # (Q,)
    keep = best_score > box_threshold
    b = boxes[keep]
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = xp.stack([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                     (cx + w / 2) * img_w, (cy + h / 2) * img_h], axis=1)
    return xyxy, best_cls[keep], best_score[keep]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_gdino_postprocess.py`
Expected: `2/2 passed`.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_gdino_postprocess.py
git commit -m "feat(tcn_artekmed): add lean GDINO post-process (fixed-prompt token->class)"
```

---

### Task 3: `GDinoTrtDetector` runtime class

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py`

**Interfaces:**
- Consumes: `gdino_postprocess` (from `langsam_helpers`).
- Produces: `GDinoTrtDetector(engine_path, text_npz, prompts, device, box_threshold=0.3, hw=(512,672))`
  with `detect(rgb_gpu) -> (boxes_xyxy_gpu, class_ids_list, scores_gpu)`, where `rgb_gpu` is an
  `(H0,W0,3)` uint8 CUDA tensor and returned boxes are pixel-xyxy in the ORIGINAL camera resolution.

- [ ] **Step 1: Implement** (loads engine + text tensors once; per-call: GPU resize+normalize
  → set tensor addresses → `execute_async_v3` → `gdino_postprocess` on GPU with `xp=cp`).

```python
# in langsam_common.py  (import tensorrt as trt inside __init__ to keep it optional)
class GDinoTrtDetector:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, engine_path, text_npz, prompts, device,
                 box_threshold=0.3, hw=(512, 672)):
        import tensorrt as trt
        self.trt = trt
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self.H, self.W = int(hw[0]), int(hw[1])
        self.box_threshold = float(box_threshold)
        self.num_classes = len(prompts)
        data = np.load(text_npz, allow_pickle=True)
        saved_prompts = [str(p) for p in list(data["prompts"])]
        if saved_prompts != [str(p) for p in prompts]:
            raise ValueError(f"engine text prompts {saved_prompts} != configured {prompts}; re-export")
        self.token_class_ids = cp.asarray(data["token_class_ids"])
        with torch.cuda.device(self.device):
            logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            # constant text inputs -> persistent GPU tensors
            self._text = {}
            for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"):
                want = trt.nptype(self.engine.get_tensor_dtype(n))
                t = torch.as_tensor(np.ascontiguousarray(data[n])).to(self.device)
                t = t.to(self._torch_dtype(want)).contiguous()
                self._text[n] = t
            self._mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def _torch_dtype(np_t):
        import numpy as _np
        return {_np.int32: torch.int32, _np.int64: torch.int64, _np.float32: torch.float32,
                _np.float16: torch.float16, _np.bool_: torch.bool}[np_t]

    def detect(self, rgb_gpu):
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            H0, W0 = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
            img = rgb_gpu.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div(255.0)
            img = torch.nn.functional.interpolate(img, size=(self.H, self.W),
                                                  mode="bilinear", align_corners=False, antialias=True)
            img = ((img - self._mean) / self._std).contiguous()
            self.ctx.set_input_shape("img", tuple(img.shape))
            self.ctx.set_tensor_address("img", img.data_ptr())
            for n, t in self._text.items():
                self.ctx.set_input_shape(n, tuple(t.shape)); self.ctx.set_tensor_address(n, t.data_ptr())
            outs = {}
            for i in range(self.engine.num_io_tensors):
                n = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.OUTPUT:
                    outs[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), device=self.device,
                                          dtype=torch.float32)
                    self.ctx.set_tensor_address(n, outs[n].data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            logits = cp.from_dlpack(outs["logits"][0]); boxes = cp.from_dlpack(outs["boxes"][0])
            xyxy, cls, sc = gdino_postprocess(logits, boxes, self.token_class_ids,
                                              self.num_classes, self.box_threshold, H0, W0, xp=cp)
            return xyxy, [int(c) for c in cls.get()], sc
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile langsam_common.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py
git commit -m "feat(tcn_artekmed): add GDinoTrtDetector (prebuilt engine runtime)"
```

---

### Task 4: Wire `gdino_backend` toggle into `LangSamBatchOp`

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py`

**Interfaces:**
- Consumes: `GDinoTrtDetector`.
- Produces: `LangSamBatchOp` gains `gdino_backend` ("pytorch"|"trt"); when "trt" it builds a
  `GDinoTrtDetector` and uses `detect()` per camera; the SAM call and panoptic output are unchanged.

- [ ] **Step 1: In `LangSamBatchOp.__init__`**, read `gdino_backend`, `gdino_trt_engine`,
  `gdino_trt_text`, `gdino_input_size` (reused as the engine H if square-ish; else the export
  `hw`) from `langsam_cfg`. When `"trt"`, construct `self.gdino_trt = GDinoTrtDetector(engine,
  text, self.prompts, self.device, self.box_threshold, hw)`. Otherwise keep building `self.gdino`
  as today.

- [ ] **Step 2: In `compute`**, replace the detection block:

```python
            torch.cuda.nvtx.range_push("gdino")
            if self.gdino_backend == "trt":
                sam_imgs, sam_boxes, sam_labels, sam_idx = [], [], [], []
                for i, im in enumerate(rgb_gpu):
                    bx, cls, sc = self.gdino_trt.detect(im)   # xyxy px (GPU), class ids, scores
                    if len(cls) > 0:
                        sam_imgs.append(im)
                        sam_boxes.append(bx)                  # GPU tensor -> SAM (_prep_prompts)
                        sam_labels.append([self.prompts[c - 1] for c in cls])
                        sam_scores_i = sc
                        sam_idx.append(i)
                        det_scores.append(sc)
            else:
                gres = self.gdino.predict_gpu_batch(rgb_gpu, self.prompts,
                                                    self.box_threshold, self.text_threshold, hw)
                # ... existing _extract_result loop ...
            torch.cuda.nvtx.range_pop()
```

  Keep the existing SAM + panoptic code. For the TRT branch, `build_panoptic_map` gets
  `sam_labels[k]` and the SAM `mscores[k]` (same as today) — the GDINO `sc` is only used to
  decide keep/drop. (The label list already carries class identity.)

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile langsam_multicam_fragment.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py
git commit -m "feat(tcn_artekmed): gdino_backend toggle (trt) in LangSamBatchOp"
```

---

### Task 5: YAML config + single-cam parity toggle

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml`
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam2operator.py`

- [ ] **Step 1: YAML** — add under `langsam_inference`:

```yaml
  gdino_backend: "trt"            # "pytorch" | "trt"
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_tf32.engine"
  gdino_trt_text: "/srv/models/active/groundingdino/gdino_swint_prompts.npz"
```

- [ ] **Step 2: Single-cam** — mirror the Task-4 toggle in `LangSAM2Operator` (build a
  `GDinoTrtDetector` when `gdino_backend=="trt"`; call `detect(rgb_gpu)` in place of
  `predict_gpu`/`predict_gpu_batch`; keep the pytorch path as fallback). Reuse the same config keys.

- [ ] **Step 3: Verify syntax + yaml**

Run: `python3 -m py_compile langsam2operator.py && python3 -c "import yaml; yaml.safe_load(open('tcn_shm_vlm_inference.yaml'))"`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add -u && git commit -m "feat(tcn_artekmed): wire gdino_backend config + single-cam TRT path"
```

---

### Task 6: Container validation + profiling

**Files:** none (verification).

- [ ] **Step 1:** Put the ONNX + `.npz` + parity ref in the container-visible
  `/data/models/active/groundingdino/` (host `--stage export`), then build the engine **inside the
  container** with `--stage build`. A host-built engine will NOT load: TRT engines are
  version-locked (host TRT 11.2.1.2 / serialization 243 vs the container's 239), which is why the
  export tool is split into an export stage and a container build stage. See `gdino_trt_export.md`.
- [ ] **Step 2:** Run single-GPU (`langsam_multicam` one worker) with `gdino_backend: pytorch`,
  then `trt`; confirm the tiled masks and panoptic output match visually (same detections/colors).
- [ ] **Step 3:** With `timing`/nsys, confirm the `gdino` NVTX range drops from ~230 ms toward
  ~40 ms×N and GPU util rises.
- [ ] **Step 4:** Re-enable the **2-GPU split** and profile — verify it now scales (GDINO no
  longer GIL-bound), per the spec's key hypothesis.
- [ ] **Step 5:** If mask quality diverges from the pytorch backend, re-check the parity gate and
  the token→class map; fall back to `gdino_backend: pytorch` while investigating.

---

## Self-Review notes

- **Spec coverage:** offline export+parity (Task 1), lean post-process (Task 2), runtime detector
  (Task 3), backend toggle + integration (Tasks 4–5), config + fallback (Task 5), validation +
  split re-test (Task 6). All spec sections covered.
- **Type consistency:** `gdino_postprocess` signature identical in Task 2 (def), Task 3 (call), and
  the test. `GDinoTrtDetector.detect` returns `(boxes_xyxy_gpu, class_ids_list, scores_gpu)` used
  consistently in Task 4. `.npz` keys match between Task 1 (produced) and Task 3 (consumed).
- **Known container-only verification risks:** TRT-11 engine deserialization + `execute_async_v3`
  binding; GPU resize/normalize matching the export preprocessing (affects parity); dtype of the
  baked text tensors vs engine expectation; feeding GPU box tensors to SAM `_prep_prompts` in the
  TRT branch (already proven for the pytorch branch). These need the container (no TRT/engine on host).
