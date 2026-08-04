# Batched Grounding DINO inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue one Grounding DINO TensorRT execution and two device syncs per Holoscan tick instead of one execution and ~5 syncs *per camera*, cutting the `gdino` stage on the bottleneck worker.

**Architecture:** Rebuild the engine with a batch-dynamic optimization profile (container `--stage build` only — the ONNX is already batch-dynamic), add `GDinoTrtDetector.detect_batch` that preprocesses all of a worker's frames into one `(N,3,H,W)` tensor for a single `execute_async_v3`, and replace the per-image post-process with a sync-free batched one whose results reach the host in a single transfer. The prompt→class mapping is made re-derivable so a subset/reordering of the baked prompts adapts at runtime.

**Tech Stack:** Python 3.12, TensorRT (in-container), PyTorch, CuPy, Holoscan SDK 3.7, numpy.

**Spec:** [`../specs/2026-08-04-gdino-batched-inference-design.md`](../specs/2026-08-04-gdino-batched-inference-design.md)

## Global Constraints

- **No pytest.** Tests in `python/tests/` are plain scripts with a `__main__` runner that prints `PASS`/`FAIL` and exits non-zero. Run them with `python3 tests/test_<name>.py` from `applications/tcn_artekmed/tcn_shm_vlm_inference/python`.
- **Host tests must be numpy-only** — no torch, cupy, holoscan, or tensorrt imports — so they run on the host without the container. `langsam_helpers.py` is the numpy-only module that exists for exactly this reason; put pure logic there.
- **Container work** runs inside the holohub `tcn_shm_receiver` container: repo at `/workspace/holohub`, models at `/srv/models/active/groundingdino` (host `/data/models/active/groundingdino`). Start it with `./run_tcn_shm_receiver.sh`.
- **No host re-export.** The ONNX already declares `batch_size` dynamic on all six inputs and both outputs (`docs/gdino_trt_export.py:90-99`). Only `--stage build` re-runs.
- **YAML keys unchanged:** `gdino_trt_engine`, `gdino_trt_text`, `gdino_trt_hw` keep their current names and values; one engine file serves both workers.
- **Boxes must never leave the GPU.** SAM's `_prep_prompts` receives GPU tensors; a D2H+H2D round-trip there was removed deliberately in earlier work. Only class ids and scores may be transferred.
- **Batch profile:** min 1, opt 3, max 5.
- **Commit messages:** `feat(tcn_artekmed): ...` / `test(tcn_artekmed): ...`, ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **Current baseline to beat** (2026-08-04 nsys, 588.3 s, 5072 ticks): `gdino` stage 118.0 ms on worker B (3 cameras, 39.3 ms/camera) and 96.5 ms on worker A (2 cameras, 48.2 ms/camera); tick period 221.8 ms; TRT engine execution itself only ~11.9 ms/camera.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `python/langsam_helpers.py` | numpy/cupy-agnostic pure logic: batched decode, class-token masks, prompt remap | 1, 2 |
| `python/tests/test_gdino_postprocess.py` | host tests for the decode, incl. batch-vs-per-image equivalence | 1 |
| `python/tests/test_prompt_remap.py` | host tests for the prompt remap | 2 |
| `docs/gdino_trt_export.py` | batch-dynamic profile + batch-consistency gate | 3 |
| `docs/gdino_trt_export.md` | document `--batch` and the new gate | 3 |
| `python/langsam_common.py` | `GDinoTrtDetector`: `set_prompts`, batched text cache, `detect_batch` | 4 |
| `python/langsam_multicam_fragment.py` | call `detect_batch`; keep `_cmap`/labels consistent with the active prompt set | 5 |

---

### Task 1: Batched decode + hoisted class-token masks

Pure numpy/cupy logic with a host test proving it is numerically identical to the existing per-image path. This is the correctness gate for the whole plan and needs no GPU.

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py` (append after `gdino_postprocess`, which ends at line 128)
- Test: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_gdino_postprocess.py` (append; keep the existing two tests)

**Interfaces:**
- Consumes: existing `gdino_postprocess(logits, boxes, token_class_ids, num_classes, box_threshold, img_h, img_w, xp)` as the reference oracle.
- Produces:
  - `build_class_token_masks(token_class_ids, num_classes, xp=np) -> (C,256) bool`
  - `gdino_postprocess_batch(logits, boxes, class_masks, img_hw, xp=np) -> (xyxy (N,Q,4) float, best_cls (N,Q) int, best_score (N,Q) float)` — **no thresholding**, no boolean indexing, no host transfer. Task 4 thresholds after one transfer.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_gdino_postprocess.py` (above the `if __name__` block):

```python
from langsam_helpers import build_class_token_masks, gdino_postprocess_batch


def test_build_class_token_masks_marks_each_class():
    tcid = np.zeros(256, np.int64); tcid[1] = 1; tcid[2] = 1; tcid[4] = 2
    m = build_class_token_masks(tcid, num_classes=2, xp=np)
    assert m.shape == (2, 256)
    assert list(np.nonzero(m[0])[0]) == [1, 2]      # class 1 -> tokens 1,2
    assert list(np.nonzero(m[1])[0]) == [4]         # class 2 -> token 4
    assert m.dtype == np.bool_


def test_build_class_token_masks_empty_class_is_all_false():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    m = build_class_token_masks(tcid, num_classes=2, xp=np)
    assert m[0].any() and not m[1].any()


def _random_case(rng, n, q=40, c=2):
    tcid = np.zeros(256, np.int64)
    tcid[1] = 1; tcid[2] = 1
    if c >= 2:
        tcid[4] = 2
    logits = rng.normal(0, 3, size=(n, q, 256)).astype(np.float32)
    boxes = rng.uniform(0.15, 0.85, size=(n, q, 4)).astype(np.float32)
    boxes[..., 2:] *= 0.2                      # keep w/h small so boxes stay in frame
    return tcid, logits, boxes


def test_batch_matches_per_image_reference():
    """The batched decode + threshold must equal looping the per-image gdino_postprocess."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5):
        for thr in (0.3, 0.5, 0.9):
            tcid, logits, boxes = _random_case(rng, n)
            hw = [(100 + 7 * i, 200 + 11 * i) for i in range(n)]      # distinct per camera
            masks = build_class_token_masks(tcid, 2, xp=np)
            xyxy, bcls, bscore = gdino_postprocess_batch(logits, boxes, masks, hw, xp=np)
            for i in range(n):
                exp_bx, exp_cls, exp_sc = gdino_postprocess(
                    logits[i], boxes[i], tcid, 2, thr, hw[i][0], hw[i][1], xp=np)
                keep = np.nonzero(bscore[i] > thr)[0]
                assert list(bcls[i][keep]) == list(exp_cls), (n, thr, i)
                assert np.allclose(bscore[i][keep], exp_sc, atol=1e-6), (n, thr, i)
                assert np.allclose(xyxy[i][keep], exp_bx, atol=1e-3), (n, thr, i)


def test_batch_handles_zero_detections():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    logits = np.full((2, 5, 256), _sig_inv(0.05), np.float32)
    boxes = np.tile(np.array([0.5, 0.5, 0.1, 0.1], np.float32), (2, 5, 1))
    masks = build_class_token_masks(tcid, 1, xp=np)
    xyxy, bcls, bscore = gdino_postprocess_batch(logits, boxes, masks, [(100, 100)] * 2, xp=np)
    assert xyxy.shape == (2, 5, 4) and bcls.shape == (2, 5)
    assert not (bscore > 0.3).any()


def test_batch_empty_class_scores_zero_like_reference():
    """A prompt with no tokens (e.g. dropped by a remap) must score 0, never win argmax."""
    tcid = np.zeros(256, np.int64); tcid[1] = 1          # class 2 has no tokens
    logits = np.full((1, 3, 256), _sig_inv(0.02), np.float32)
    logits[0, 0, 1] = _sig_inv(0.95)
    masks = build_class_token_masks(tcid, 2, xp=np)
    _, bcls, bscore = gdino_postprocess_batch(logits, boxes=np.full((1, 3, 4), 0.5, np.float32),
                                              class_masks=masks, img_hw=[(10, 10)], xp=np)
    assert bcls[0, 0] == 1 and abs(float(bscore[0, 0]) - 0.95) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gdino_postprocess.py
```
Expected: `ImportError: cannot import name 'build_class_token_masks' from 'langsam_helpers'`.

- [ ] **Step 3: Write minimal implementation**

Append to `python/langsam_helpers.py`:

```python
def build_class_token_masks(token_class_ids, num_classes, xp=np):
    """(num_classes, 256) bool; row c-1 marks the token slots belonging to class c.

    Hoisted out of the per-frame path. `token_class_ids` is fixed for a given prompt set, so
    these masks -- and the per-class `.any()` check the per-image path ran for every camera on
    every frame, each a device->host sync -- are computed once, when the active prompt set
    changes. See GDinoTrtDetector.set_prompts.
    """
    tcid = xp.asarray(token_class_ids)
    return xp.stack([(tcid == c) for c in range(1, int(num_classes) + 1)], axis=0)


def gdino_postprocess_batch(logits, boxes, class_masks, img_hw, xp=np):
    """Batched Grounding DINO decode -- adds NO synchronisation.

    logits (N,Q,256), boxes (N,Q,4) cxcywh in [0,1], class_masks (C,256) bool from
    build_class_token_masks, img_hw a list of N (h,w) giving each camera's ORIGINAL pixel size.

    Returns (xyxy (N,Q,4) in pixels, best_cls (N,Q) 1-based, best_score (N,Q)) for ALL queries;
    the caller thresholds on `best_score` after a single device->host transfer. Deliberately
    returns unfiltered arrays: boolean indexing would force cupy to size the output on the
    host, which is one of the syncs this whole change exists to remove.
    """
    probs = 1.0 / (1.0 + xp.exp(-logits))                          # (N,Q,256)
    # xp.where(mask, probs, 0.0) instead of probs[..., mask]: sigmoids are strictly > 0, so
    # masking with 0 yields the same maximum as selecting the class's columns, while an empty
    # class scores exactly 0 -- matching gdino_postprocess -- and neither indexes nor syncs.
    per_class = xp.stack(
        [xp.where(class_masks[c], probs, 0.0).max(axis=-1) for c in range(class_masks.shape[0])],
        axis=-1)                                                   # (N,Q,C)
    best_idx = per_class.argmax(axis=-1)                           # (N,Q) 0-based
    best_cls = best_idx + 1                                        # 1-based; 0 is background
    best_score = xp.take_along_axis(per_class, best_idx[..., None], axis=-1)[..., 0]

    h = xp.asarray([float(a) for a, _ in img_hw]).reshape(-1, 1)
    w = xp.asarray([float(b) for _, b in img_hw]).reshape(-1, 1)
    cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = xp.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], axis=-1)   # (N,Q,4) pixels
    return xyxy, best_cls, best_score
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gdino_postprocess.py
```
Expected: `7/7 passed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_gdino_postprocess.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): batched, sync-free GDINO decode + hoisted class-token masks

gdino_postprocess_batch decodes all of a worker's cameras in one pass and returns
unfiltered arrays, so it neither boolean-indexes nor transfers -- the caller
thresholds after a single device->host copy. build_class_token_masks hoists the
per-class token masks, whose .any() cost a sync per class per camera per frame.

Tested against the existing per-image gdino_postprocess as an oracle over random
logits for N in 1..5 and three thresholds, including the zero-detection and
empty-class cases.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Prompt remap

Lets a runtime prompt change adapt when it is a subset and/or reordering of the baked prompts, and fail loudly when it is not. Pure numpy, host-testable.

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py` (append)
- Create: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_prompt_remap.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `build_prompt_remap(baked_prompts, active_prompts) -> np.ndarray int64 of length len(baked)+1`. Applied as `remap[token_class_ids]`. Raises `ValueError` on an unbaked term or a duplicate.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_prompt_remap.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Host tests for the baked->active Grounding DINO prompt remap (numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import build_prompt_remap


def _tcid():
    """token_class_ids for baked prompts ["floor", "person"]: class 1 -> tokens 1,2; class 2 -> 4."""
    t = np.zeros(256, np.int64); t[1] = 1; t[2] = 1; t[4] = 2
    return t


def test_identity_is_a_noop():
    remap = build_prompt_remap(["floor", "person"], ["floor", "person"])
    assert list(remap) == [0, 1, 2]
    assert np.array_equal(remap[_tcid()], _tcid())


def test_normalisation_ignores_case_and_whitespace():
    remap = build_prompt_remap(["floor", "person"], ["  Floor ", "PERSON"])
    assert list(remap) == [0, 1, 2]


def test_reorder_permutes_class_ids():
    remap = build_prompt_remap(["floor", "person"], ["person", "floor"])
    assert list(remap) == [0, 2, 1]           # baked floor(1) -> 2, baked person(2) -> 1
    out = remap[_tcid()]
    assert list(np.nonzero(out == 2)[0]) == [1, 2]   # floor tokens now class 2
    assert list(np.nonzero(out == 1)[0]) == [4]      # person tokens now class 1


def test_subset_zeroes_the_dropped_class_tokens():
    remap = build_prompt_remap(["floor", "person"], ["person"])
    assert list(remap) == [0, 0, 1]
    out = remap[_tcid()]
    assert not (out == 2).any()                       # no class 2 left
    assert list(np.nonzero(out == 1)[0]) == [4]       # person became class 1
    assert out[1] == 0 and out[2] == 0                # floor tokens dropped to background


def test_unbaked_term_raises_and_names_the_baked_set():
    try:
        build_prompt_remap(["floor", "person"], ["floor", "robot"])
    except ValueError as e:
        assert "robot" in str(e) and "floor" in str(e)
        return
    raise AssertionError("expected ValueError for an unbaked prompt")


def test_duplicate_after_normalisation_raises():
    try:
        build_prompt_remap(["floor", "person"], ["floor", "Floor"])
    except ValueError as e:
        assert "duplicate" in str(e).lower()
        return
    raise AssertionError("expected ValueError for duplicate prompts")


def test_empty_active_set_raises():
    try:
        build_prompt_remap(["floor", "person"], [])
    except ValueError as e:
        assert "empty" in str(e).lower()
        return
    raise AssertionError("expected ValueError for an empty prompt set")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, repr(e))
    print(f"{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_prompt_remap.py
```
Expected: `ImportError: cannot import name 'build_prompt_remap'`.

- [ ] **Step 3: Write minimal implementation**

Append to `python/langsam_helpers.py`:

```python
def build_prompt_remap(baked_prompts, active_prompts):
    """Baked class ids -> active class ids, for a prompt set the engine can already express.

    The TRT engine bakes the prompt TOKENS (input_ids, text_token_mask, fixed L), but the
    prompt->class mapping lives entirely in token_class_ids, where class i+1 is baked prompt i
    (see docs/gdino_trt_export.py build_text). So any subset and/or reordering of the baked
    prompts is expressible by renumbering alone -- no tokenizer, no re-export.

    Returns an int64 array of length len(baked_prompts)+1: index 0 (background) maps to 0, and
    index i+1 maps to the 1-based position of baked prompt i in active_prompts, or 0 if that
    prompt was dropped. Apply as `remap[token_class_ids]`.

    Raises ValueError if active_prompts is empty, contains duplicates after normalisation, or
    contains a term that is not baked into the engine -- that term's tokens simply are not in
    the engine's input_ids, so it requires a re-export + rebuild.
    """
    def _norm(p):
        return str(p).strip().lower()

    baked = [_norm(p) for p in baked_prompts]
    active = [_norm(p) for p in active_prompts]
    if not active:
        raise ValueError("active prompt set is empty; at least one prompt is required")
    if len(set(active)) != len(active):
        raise ValueError(f"duplicate prompts after normalisation: {active}")
    unknown = [p for p in active if p not in baked]
    if unknown:
        raise ValueError(
            f"prompts {unknown} are not baked into the GDINO TRT engine (baked: {baked}). "
            f"Only a subset or reordering of the baked prompts can be applied at runtime; a new "
            f"term needs a re-export and rebuild:\n"
            f"  host:      python3 gdino_trt_export.py --stage export --prompts {' '.join(active)} ...\n"
            f"  container: python3 gdino_trt_export.py --stage build --out /srv/models/active/groundingdino")
    remap = np.zeros(len(baked) + 1, np.int64)
    for i, p in enumerate(baked):
        remap[i + 1] = active.index(p) + 1 if p in active else 0
    return remap
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py
```
Expected: `7/7 passed` then `7/7 passed`, both exit 0.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_prompt_remap.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): derive a baked->active GDINO prompt remap

The engine bakes prompt tokens but the prompt->class mapping lives in
token_class_ids, where class i+1 is baked prompt i. So a subset and/or reordering
of the baked prompts is expressible by renumbering alone, with no re-export; a
genuinely new term is not, and raises with the exact export+build commands.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Batch-dynamic engine profile + batch-consistency gate

Container-side build change. The gate is the detector for this plan's primary risk: the ONNX was traced at batch 1, so a reshape may have baked a literal batch dimension despite the dynamic axes.

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py` (`build_engine`, `stage_build`, argparse)
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md`

**Interfaces:**
- Consumes: existing `_run_engine(ser, feed_cpu)`, `_top_box(logits, boxes)`, `_iou(a, b)`, `load_text_npz`, `load_parity_ref`.
- Produces: `build_engine(..., batch=(1,3,5))` and `batch_consistency_gate(ser, text, img, s_ref, b_ref, batch, min_iou, max_score_delta)`. An engine whose `img` profile max batch is 5.

- [ ] **Step 1: Parameterise the profile**

In `docs/gdino_trt_export.py`, change the `build_engine` signature and its profile block. Replace:

```python
def build_engine(onnx_path, engine_path, H, W, L, fp16=False):
```
with:
```python
def build_engine(onnx_path, engine_path, H, W, L, fp16=False, batch=(1, 3, 5)):
```

and replace these four lines:

```python
    prof = builder.create_optimization_profile()
    prof.set_shape("img", (1, 3, H, W), (1, 3, H, W), (1, 3, H, W))
    for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        prof.set_shape(n, (1, L), (1, L), (1, L))
    prof.set_shape("text_token_mask", (1, L, L), (1, L, L), (1, L, L))
```
with:
```python
    prof = builder.create_optimization_profile()
    # Batch-dynamic: one engine serves every worker. The ONNX already declares batch_size
    # dynamic on all six inputs (see export_onnx's dynamic_axes); only this profile used to
    # pin it to 1, which forced one execute + one stream sync PER CAMERA at runtime.
    bmin, bopt, bmax = (int(b) for b in batch)
    if not 1 <= bmin <= bopt <= bmax:
        raise SystemExit(f"--batch must satisfy 1 <= min <= opt <= max, got {batch}")
    prof.set_shape("img", (bmin, 3, H, W), (bopt, 3, H, W), (bmax, 3, H, W))
    for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        prof.set_shape(n, (bmin, L), (bopt, L), (bmax, L))
    prof.set_shape("text_token_mask", (bmin, L, L), (bopt, L, L), (bmax, L, L))
```

Also change the final print in `build_engine` from:
```python
    print(f"engine written: {engine_path}")
```
to:
```python
    print(f"engine written: {engine_path}  (batch {bmin}/{bopt}/{bmax}, "
          f"{os.path.getsize(engine_path) / 2**20:.0f} MiB)")
```

- [ ] **Step 2: Add the batch-consistency gate**

Insert immediately after the `parity_gate` function in `docs/gdino_trt_export.py`:

```python
def batch_consistency_gate(ser, text, img, s_ref, b_ref, batch, min_iou=0.99,
                           max_score_delta=0.01):
    """Every slice of a batched run must agree with the batch-1 result.

    The ONNX was TRACED at batch 1. GroundingDINO is full of reshape/view ops that can bake a
    literal batch dimension even though dynamic_axes marks it dynamic; the failure mode is
    silently wrong output for slices 1..N-1, which would look like random detection dropouts on
    some cameras. So replicate the parity image to the profile's opt batch and check each slice.

    Bit-exactness is NOT required -- batch>1 legitimately selects different kernels -- so this
    reuses the batch-1 parity criterion: top-box IoU and top-score agreement.
    """
    b = int(batch[1])
    if b < 2:
        print("batch-consistency gate: opt batch < 2, nothing to check")
        return
    feed = {"img": img.repeat(b, 1, 1, 1)}
    for k, v in text.items():
        feed[k] = v.repeat(*([b] + [1] * (v.dim() - 1)))
    o = _run_engine(ser, feed)
    worst_iou, worst_ds = 1.0, 0.0
    for i in range(b):
        s_i, b_i = _top_box(o["logits"][i:i + 1], o["boxes"][i:i + 1])
        iou = _iou(b_ref, b_i)
        ds = abs(float(s_i) - float(s_ref))
        worst_iou = min(worst_iou, iou)
        worst_ds = max(worst_ds, ds)
        if iou < min_iou or ds > max_score_delta:
            raise SystemExit(
                f"BATCH CONSISTENCY GATE FAILED at slice {i}/{b}: IoU {iou:.4f} (need "
                f">= {min_iou}), |score delta| {ds:.4f} (need <= {max_score_delta}).\n"
                f"The ONNX was traced at batch 1 and appears to have baked that batch "
                f"dimension, so batching is NOT safe with this ONNX. Re-export on the host "
                f"with a batch>1 dummy input, then rebuild.")
    print(f"batch-consistency gate OK (batch {b}: worst IoU {worst_iou:.4f}, "
          f"worst |score delta| {worst_ds:.4f})")
```

- [ ] **Step 3: Wire it into the build stage and the CLI**

In `stage_build`, replace:
```python
    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16)
    parity_gate(ser, text, ref_img, s_pt, b_pt, ref_image_path,
                min_iou=args.min_iou, min_detect=args.min_detect)
```
with:
```python
    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16, batch=args.batch)
    parity_gate(ser, text, ref_img, s_pt, b_pt, ref_image_path,
                min_iou=args.min_iou, min_detect=args.min_detect)
    batch_consistency_gate(ser, text, ref_img, s_pt, b_pt, args.batch, min_iou=args.min_iou)
```

and add to the argparse block, next to `--fp16`:
```python
    ap.add_argument("--batch", nargs=3, type=int, default=[1, 3, 5],
                    metavar=("MIN", "OPT", "MAX"),
                    help="build-stage optimization profile batch range. OPT should match the "
                         "busiest worker's camera count, MAX the largest split you want to run "
                         "without rebuilding (default 1 3 5)")
```

- [ ] **Step 4: Verify the tool still parses and guards**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/docs && python3 -m py_compile gdino_trt_export.py && echo "compile OK"
```
Expected: `compile OK`. (The tool needs tensorrt/torch to run, which the host export venv has: `~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python`. Verifying `--help` shows `--batch` is enough here; the real build happens in Task 6.)

```bash
~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python gdino_trt_export.py --stage build --help 2>&1 | grep -A2 -- "--batch"
```
Expected: the `--batch MIN OPT MAX` help text.

- [ ] **Step 5: Document it**

In `docs/gdino_trt_export.md`, in the "Stage 2 — build (inside the runtime container)" section, replace the command block with:

````markdown
```bash
./run_tcn_shm_receiver.sh          # drops into the tcn_shm_receiver container
python3 /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
  --stage build --hw 512 672 --out /srv/models/active/groundingdino \
  --batch 1 3 5
```

`--batch MIN OPT MAX` (default `1 3 5`) sets the optimization profile's batch range so one
engine can serve a whole worker's cameras in a single execution. Set `OPT` to the busiest
worker's camera count and `MAX` to the largest split you want to run without rebuilding. The
ONNX is already batch-dynamic, so this needs no host re-export.

The build then runs two gates: the batch-1 **parity gate** against the stage-1 PyTorch
reference, and a **batch-consistency gate** that replays the same image at batch `OPT` and
requires every slice to match batch 1 (top-box IoU >= `--min-iou`, top-score delta <= 0.01).
The second exists because the ONNX was traced at batch 1: GroundingDINO's reshape ops can bake
that dimension despite the dynamic axes, and the failure mode is silently wrong output on
slices 1..N-1. If it fails, batching needs a host re-export with a batch>1 dummy.
````

- [ ] **Step 6: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): batch-dynamic GDINO engine profile + consistency gate

--batch MIN OPT MAX (default 1 3 5) widens the optimization profile so one engine
serves a whole worker's cameras in a single execution. The ONNX was always
batch-dynamic; only the profile pinned batch=1, forcing one execute and one stream
sync per camera.

Adds a batch-consistency gate: the ONNX was traced at batch 1, so a reshape may have
baked that dimension despite dynamic_axes, silently corrupting slices 1..N-1. The
gate replays the parity image at the opt batch and checks every slice by the same
IoU/score criterion as the batch-1 gate.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `GDinoTrtDetector.detect_batch` + `set_prompts`

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py` — import block at lines 27-36, `GDinoTrtDetector.__init__` (lines 532-569), and `detect` (lines 576-603)

**Interfaces:**
- Consumes: `build_class_token_masks`, `gdino_postprocess_batch` (Task 1), `build_prompt_remap` (Task 2), and an engine with a batch-dynamic profile (Task 3).
- Produces:
  - `GDinoTrtDetector.set_prompts(active: list[str]) -> None`
  - `GDinoTrtDetector.detect_batch(frames: list[torch.Tensor]) -> list[tuple[cupy xyxy, list[int], cupy scores]]`
  - `GDinoTrtDetector.detect(rgb_gpu)` unchanged in signature, now `detect_batch([rgb_gpu])[0]`
  - attribute `max_batch: int`

- [ ] **Step 1: Extend the helper import**

In `python/langsam_common.py`, in the `from langsam_helpers import (  # noqa: F401` block (lines 27-36), add three names after `gdino_postprocess,`:

```python
    gdino_postprocess_batch,
    build_class_token_masks,
    build_prompt_remap,
```

- [ ] **Step 2: Replace the prompt handling in `__init__`**

Replace lines 539-545, currently:

```python
        self.num_classes = len(prompts)
        data = np.load(text_npz, allow_pickle=True)
        saved = [str(p) for p in list(data["prompts"])]
        if saved != [str(p) for p in prompts]:
            raise ValueError(
                f"GDINO TRT engine text prompts {saved} != configured {list(prompts)}; re-export the engine")
        self.token_class_ids = cp.asarray(data["token_class_ids"])
```
with:
```python
        data = np.load(text_npz, allow_pickle=True)
        self._baked_prompts = [str(p) for p in list(data["prompts"])]
        self._token_class_ids_baked = cp.asarray(data["token_class_ids"])
        self._prompt_key = None
        self.prompts = None
        self.num_classes = 0
        self.token_class_ids = None
        self._class_masks = None
        self._text_batched = {}
```

Then, at the very end of `__init__` (after the `self._std = ...` line, still inside the `with torch.cuda.device(self.device):` block), append:

```python
            # Profile max batch: how many cameras one execution can cover. Read from the engine
            # so a stale batch-1 engine is reported clearly instead of failing deep in TRT.
            try:
                self.max_batch = int(self.engine.get_tensor_profile_shape("img", 0)[2][0])
            except Exception:
                self.max_batch = 1
            self.set_prompts(prompts)
```

- [ ] **Step 3: Add `set_prompts` and the batched text cache**

Insert after the `_torch_dtype` staticmethod (after line 574):

```python
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

    def _text_for_batch(self, n):
        """Baked text tensors replicated to batch n, cached per n (they never change)."""
        t = self._text_batched.get(n)
        if t is None:
            t = {k: (v if n == 1 else v.repeat(*([n] + [1] * (v.dim() - 1)))).contiguous()
                 for k, v in self._text.items()}
            self._text_batched[n] = t
        return t
```

- [ ] **Step 4: Replace `detect` with `detect_batch`**

Replace the whole `detect` method (lines 576-603) with:

```python
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
        if n > self.max_batch:
            raise ValueError(
                f"{n} cameras requested but the GDINO engine's profile allows batch <= "
                f"{self.max_batch}. Rebuild it INSIDE the container with a wider profile:\n"
                f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                f"gdino_trt_export.py --stage build --hw {self.H} {self.W} "
                f"--batch 1 {n} {max(n, 5)} --out /srv/models/active/groundingdino")
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            hw0 = [(int(f.shape[0]), int(f.shape[1])) for f in frames]
            chw = [torch.nn.functional.interpolate(
                       f.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div(255.0),
                       size=(self.H, self.W), mode="bilinear", align_corners=False,
                       antialias=True)
                   for f in frames]
            img = ((torch.cat(chw, dim=0) - self._mean) / self._std).contiguous()
            self.ctx.set_input_shape("img", tuple(img.shape))
            self.ctx.set_tensor_address("img", img.data_ptr())
            for k, t in self._text_for_batch(n).items():
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
            logits = cp.from_dlpack(outs["logits"])            # (N,900,256)
            boxes = cp.from_dlpack(outs["boxes"])              # (N,900,4) cxcywh
            xyxy, best_cls, best_score = gdino_postprocess_batch(
                logits, boxes, self._class_masks, hw0, xp=cp)
            # sync 2 of 2: one copy for the whole batch. Stacked so it is a single transfer;
            # class ids are small ints, exact in float32.
            head = cp.asnumpy(cp.stack([best_cls.astype(cp.float32), best_score]))  # (2,N,Q)
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
```

- [ ] **Step 5: Verify it compiles and the helper names resolve**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_common.py && echo "compile OK"
grep -n "gdino_postprocess_batch\|build_class_token_masks\|build_prompt_remap" langsam_common.py | head
```
Expected: `compile OK`, and each of the three names appearing in both the import block and the body.

- [ ] **Step 6: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): batched GDINO detect_batch + adaptive prompt set

detect_batch runs all of a worker's cameras in one execute_async_v3 and reaches the
host once, so a tick costs 2 syncs instead of ~5 per camera. Text tensors are cached
per batch size; detect() is now a one-element wrapper.

set_prompts caches on the normalised prompt tuple -- which is what makes hoisting the
class-token masks out of the per-frame path safe -- and adapts a subset/reordering of
the baked prompts by renumbering token_class_ids. It subsumes the old exact-match
check, relaxing it from "must match" to "must be expressible".

detect_batch fails fast with the exact --batch rebuild command if asked for more
cameras than the engine's profile allows.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Call `detect_batch` from the operator

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py` — `__init__` (lines 41-75) and `compute` (lines 120-128)

**Interfaces:**
- Consumes: `GDinoTrtDetector.detect_batch` and `.set_prompts` (Task 4), existing `class_id_map`.
- Produces: no new public API; `LangSamBatchOp._apply_prompts(prompts)` is the single place that keeps the detector, `self.prompts` and `self._cmap` consistent.

- [ ] **Step 1: Add the one place that keeps prompt-derived state consistent**

In `langsam_multicam_fragment.py`, insert this method immediately after `__init__` (before `def setup`):

```python
    def _apply_prompts(self, prompts):
        """Single point of truth for prompt-derived state.

        Three things are derived from the prompt list and MUST move together, or class ids and
        mask colours silently disagree with the detections: the detector's token->class map,
        `self.prompts` (used to label boxes for SAM), and `self._cmap` (used to build the
        panoptic map). The TRT detector accepts any subset/reordering of its baked prompts and
        raises otherwise; the pytorch backend tokenises per call and accepts anything.
        """
        self.prompts = list(prompts)
        self._cmap = class_id_map(self.prompts)
        if self.gdino_trt is not None:
            self.gdino_trt.set_prompts(self.prompts)
```

Then delete line 45, which `_apply_prompts` now owns:

```python
        self._cmap = class_id_map(self.prompts)
```

(`self.prompts = list(prompts)` on line 44 stays — `_apply_prompts` re-assigns it, but the
attribute must exist before `super().__init__()` triggers `setup()`.)

Finally, at the end of `__init__` (after the `if/else` that builds `self.gdino_trt` /
`self.gdino`, still inside the `with torch.cuda.device(self.device):` block), append:

```python
            self._apply_prompts(self.prompts)
```

`self.gdino_trt` is set to `None` before that `if/else`, so `_apply_prompts` is safe on the
pytorch path too.

- [ ] **Step 2: Replace the per-camera detection loop**

In `compute`, replace lines 121-128, currently:

```python
            if self.gdino_backend == "trt":
                for i, im in enumerate(rgb_gpu):
                    boxes, cls, _ = self.gdino_trt.detect(im)   # xyxy px (GPU), class ids, scores
                    if len(cls) > 0:
                        sam_imgs.append(im)
                        sam_boxes.append(boxes)
                        sam_labels.append([self.prompts[c - 1] for c in cls])
                        sam_idx.append(i)
```
with:
```python
            if self.gdino_backend == "trt":
                # One engine execution for every camera on this worker (see detect_batch).
                for i, (boxes, cls, _) in enumerate(self.gdino_trt.detect_batch(rgb_gpu)):
                    if len(cls) > 0:
                        sam_imgs.append(rgb_gpu[i])
                        sam_boxes.append(boxes)
                        sam_labels.append([self.prompts[c - 1] for c in cls])
                        sam_idx.append(i)
```

- [ ] **Step 3: Verify it compiles and no caller of the old loop remains**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_multicam_fragment.py && echo "compile OK"
grep -n "gdino_trt.detect\|_apply_prompts\|_cmap =" langsam_multicam_fragment.py
```
Expected: `compile OK`; `detect_batch` is the only `gdino_trt.detect*` call; `self._cmap =` appears only inside `_apply_prompts`.

- [ ] **Step 4: Re-run the host tests (nothing should have regressed)**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gdino_postprocess.py && python3 tests/test_prompt_remap.py && python3 tests/test_langsam_multicam.py
```
Expected: all three report all-passed and exit 0.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): drive GDINO through detect_batch in LangSamBatchOp

One engine execution per tick instead of one per camera. _apply_prompts becomes the
single point where prompt-derived state moves together -- the detector's token->class
map, the label list, and the panoptic colour map -- so they cannot drift apart if a
prompt update is ever wired to this path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Rebuild, validate, and measure in the container

Verification only — no code changes unless a gate fails.

**Files:** none.

**Interfaces:** consumes everything from Tasks 1-5.

- [ ] **Step 1: Rebuild the engine with the batch profile**

```bash
./run_tcn_shm_receiver.sh
# inside the container:
python3 /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
  --stage build --hw 512 672 --out /srv/models/active/groundingdino --batch 1 3 5
```
Expected: `parity gate OK` with IoU ~0.999, then `batch-consistency gate OK (batch 3: ...)`, then `DONE. Engine: ...`.

**If the batch-consistency gate fails**, stop: the ONNX baked its batch-1 trace. Do not work around it. The fallback is a host re-export with a batch>1 dummy (`--stage export` in the wingdzero GroundingDINO checkout), which is outside this plan — report it and stop.

Record the engine size printed by `build_engine`; if it grew materially versus the previous ~703 MB, note it for Step 4.

- [ ] **Step 2: Run the app and confirm masks are unchanged**

```bash
./holohub run tcn_shm_vlm_inference --run-args='--tracking'
```
Expected: starts, all five cameras produce masks, and detections/colours look the same as before the change. A wrong prompt→class mapping shows up as swapped mask colours; wrong batching shows up as some cameras intermittently losing all detections.

- [ ] **Step 3: Profile and compare against the baseline**

```bash
./holohub run tcn_shm_vlm_inference --run-args='--tracking' ... profile     # profile mode
```
Then, on the host, against `/tmp/tcn/vlm_inference_profile.nsys-rep`:

```bash
cd /tmp/tcn && /opt/nvidia/nsight-systems/2026.4.1/bin/nsys stats \
  --report nvtx_pushpop_sum --format table vlm_inference_profile.nsys-rep > /dev/null
python3 - <<'EOF'
import sqlite3, statistics as st
c = sqlite3.connect("file:vlm_inference_profile.sqlite?mode=ro", uri=True)
def rows(q): return list(c.execute(q))
t0 = rows("SELECT MIN(start) FROM NVTX_EVENTS WHERE text='gdino'")[0][0]
CUT = t0 + 600*10**9          # ignore stray post-run ranges
for stage in ("gdino", "sam", "panoptic"):
    for (tid,) in rows(f"SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE text='{stage}'"):
        d = sorted((e-s)/1e6 for s, e in rows(
            f"SELECT start,end FROM NVTX_EVENTS WHERE text='{stage}' AND globalTid={tid} "
            f"AND end IS NOT NULL AND start<{CUT}"))
        if d:
            print(f"{stage:9} tid ...{tid%1000}: n={len(d):5} mean={st.mean(d):7.1f} "
                  f"p50={d[len(d)//2]:7.1f} ms")
EOF
```

Compare against the baseline in Global Constraints: `gdino` 118.0 ms (3-camera worker) and 96.5 ms (2-camera worker). The engine floor is ~11.9 ms/camera (~35.7 ms and ~23.8 ms), so the win is however much of the ~27 ms and ~36 ms per camera of non-engine time was removed.

- [ ] **Step 4: Record the outcome**

Append a short results note to `docs/specs/2026-08-04-gdino-batched-inference-design.md` under a new `## Results` heading: measured `gdino` stage before/after per worker, resulting tick period and fps, engine size, and whether GPU utilisation moved from the 51.2% / 62.4% baseline. Commit:

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-04-gdino-batched-inference-design.md
git commit -m "$(cat <<'EOF'
docs(tcn_artekmed): record measured results of batched GDINO inference

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review notes

- **Spec coverage.** Section 1 (batch-dynamic profile) → Task 3 Step 1. Section 2 (batch-consistency gate) → Task 3 Step 2. Section 3 (`detect_batch`, text cache, max-batch validation, `detect` wrapper, per-camera `img_h/img_w`) → Task 4 Steps 3-4. Section 4 (hoisted masks, batched scoring, one D2H, boxes stay on GPU) → Tasks 1 and 4. Section 5 (prompt updates, `set_prompts`, caller consistency) → Tasks 2, 4 Step 3, 5 Step 1. Section 6 (caller) → Task 5 Step 2. Testing table → Task 1 Step 1, Task 2 Step 1, Task 3 Step 1, Task 6 Steps 2-3. Risks → Task 6 Step 1 (stop condition) and Step 4 (engine size). All covered.
- **Type consistency.** `build_class_token_masks(token_class_ids, num_classes, xp)` and `gdino_postprocess_batch(logits, boxes, class_masks, img_hw, xp)` are called with exactly those names/orders in Task 4. `build_prompt_remap(baked_prompts, active_prompts)` returns a length-`len(baked)+1` array, applied as `remap[token_class_ids]` in Task 4 Step 3. `detect_batch` returns `list[(xyxy, list[int], scores)]`, unpacked as `(boxes, cls, _)` in Task 5 Step 2. `self._class_masks`, `self._text_batched`, `self._prompt_key`, `self._baked_prompts`, `self._token_class_ids_baked`, `self.max_batch` are all initialised in Task 4 Step 2 before first use in Steps 3-4.
- **Ordering.** Tasks 1-2 are pure and independent; 4 depends on both; 5 depends on 4; 3 is independent of 1/2/4/5 but must land before Task 6 Step 1. Only Task 6 needs the container.
