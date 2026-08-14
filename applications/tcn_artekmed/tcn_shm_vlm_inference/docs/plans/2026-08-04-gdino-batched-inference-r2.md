# Batched Grounding DINO inference — Revision 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the GDINO TensorRT engine at exactly N = the largest worker's camera count, and have every worker pad its frame list to N, so one engine execution covers a whole worker.

**Architecture:** The engine's batch is baked at ONNX **trace** time (a `Where` in the encoder fusion attention is only broadcast-conformable at the traced batch), so `--batch N` becomes a single integer used by both stages: export traces at N, build pins the profile to N/N/N, and both artifacts carry `_b<N>` in their filename. At run time the detector pads to N and returns only the real slices.

**Tech Stack:** Python 3.12, TensorRT 10.9 (container), PyTorch, CuPy, Holoscan SDK 3.7, numpy.

**Spec:** [`../specs/2026-08-04-gdino-batched-inference-design.md`](../specs/2026-08-04-gdino-batched-inference-design.md) (revision 2)

**Supersedes:** parts of [`2026-08-04-gdino-batched-inference.md`](./2026-08-04-gdino-batched-inference.md) — its Task 3 (`--batch MIN OPT MAX`, `batch_consistency_gate`) is replaced here. Its Tasks 1, 2, 4, 5 remain in force and must not be undone.

## Global Constraints

- **No pytest.** Host tests are plain scripts with a `__main__` PASS/FAIL runner, run as `python3 tests/test_<name>.py` from `applications/tcn_artekmed/tcn_shm_vlm_inference/python`.
- **Host tests must be numpy-only** — no torch, cupy, holoscan or tensorrt imports.
- **Boxes must never leave the GPU**; only class ids and scores may be transferred.
- Commit style `feat(tcn_artekmed): ...` / `fix(tcn_artekmed): ...`, ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- The working tree has unrelated uncommitted changes (`metadata.json`, `tcn_all.py`, `tcn_all.yaml`). Never `git add -A`; stage only the files a task names.
- **Measured facts to preserve** (do not "optimise" against them): batch-1 engine 38.64 ms/exec; batch-3 engine 81.03 ms/exec; slices agree at IoU 0.999611; every container TRT 10.9 engine deviates from PyTorch at IoU ~0.9637 with depressed scores.

---

### Task R1: Export tool — single `--batch N`, batch-tagged filenames, revised gates

**Files:**
- Modify: `../../../tcn_all/docs/gdino_trt_export.py`
- Modify: `../../../tcn_all/docs/gdino_trt_export.md`

**Interfaces produced:** `export_onnx(model, text, H, W, onnx_path, batch=1)`; `build_engine(onnx_path, engine_path, H, W, L, fp16=False, batch=1)`; `run_at_batch(ser, text, img, B)`; `slice_consistency_gate(slices, min_iou=0.999, max_score_delta=0.01)`; `fidelity_report(slice0, s_ref, b_ref, image_path, min_iou=0.99, min_detect=0.30, strict=False)`. Artifact names `gdino_swint_<H>x<W>_b<N>_tf32.{onnx,engine}`.

- [ ] **Step 1: Trace at N in `export_onnx`**

Change the signature `def export_onnx(model, text, H, W, onnx_path):` to `def export_onnx(model, text, H, W, onnx_path, batch=1):` and replace the `dummy = (...)` line with:

```python
    # The traced batch IS the engine's batch. dynamic_axes below declares batch_size symbolic
    # and TensorRT's parser reports -1, but a Where in the encoder fusion attention bakes a
    # broadcast that is only conformable at the traced batch: forcing a different batch fails
    # the build ("broadcast dimensions must be conformable"), and a multi-size profile silently
    # specialises to a static shape. So trace at exactly the batch the workers will run.
    B = int(batch)
    rep = lambda v: v if B == 1 else v.repeat(*([B] + [1] * (v.dim() - 1)))
    dummy = (torch.randn(B, 3, H, W), rep(text["input_ids"]), rep(text["attention_mask"]),
             rep(text["position_ids"]), rep(text["token_type_ids"]), rep(text["text_token_mask"]))
```

Keep `dynamic_axes` exactly as it is — it is still required to stop the tracer constant-folding `img` out of the graph (see the existing comment).

- [ ] **Step 2: Pin the build profile to N**

Change `def build_engine(onnx_path, engine_path, H, W, L, fp16=False, batch=(1, 3, 5)):` to `... batch=1):`, and replace the whole profile block (the `bmin, bopt, bmax = ...` line through `prof.set_shape("text_token_mask", ...)`) with:

```python
    # min = opt = max = N: the ONNX was traced at N and the engine can only run at N.
    B = int(batch)
    if B < 1:
        raise SystemExit(f"--batch must be >= 1, got {B}")
    prof.set_shape("img", (B, 3, H, W), (B, 3, H, W), (B, 3, H, W))
    for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        prof.set_shape(n, (B, L), (B, L), (B, L))
    prof.set_shape("text_token_mask", (B, L, L), (B, L, L), (B, L, L))
```

and change the closing print to:

```python
    print(f"engine written: {engine_path}  (batch {B}, "
          f"{os.path.getsize(engine_path) / 2**20:.0f} MiB)")
```

- [ ] **Step 3: Replace the gates**

Delete the entire `batch_consistency_gate` function and the entire `parity_gate` function, and put these three in their place:

```python
def run_at_batch(ser, text, img, B):
    """Run the engine once at batch B with the parity image replicated; per-slice (score, box)."""
    feed = {"img": img.repeat(B, 1, 1, 1)}
    for k, v in text.items():
        feed[k] = v.repeat(*([B] + [1] * (v.dim() - 1)))
    o = _run_engine(ser, feed)
    return [_top_box(o["logits"][i:i + 1], o["boxes"][i:i + 1]) for i in range(B)]


def slice_consistency_gate(slices, min_iou=0.999, max_score_delta=0.01):
    """BLOCKING. Every slice of a batched run must agree with slice 0.

    This is the check that proves batching is correct. The engine's batch is baked at trace
    time, and the failure mode of getting that wrong is silently wrong output on slices
    1..N-1 -- which looks like random per-camera detection dropouts, not like a crash.
    Identical inputs, so the bar is tight (measured 0.999611 on a good engine).
    """
    if len(slices) < 2:
        print("slice-consistency gate: batch 1, nothing to compare")
        return
    s0, b0 = slices[0]
    worst_iou, worst_ds = 1.0, 0.0
    for i, (s, b) in enumerate(slices[1:], start=1):
        iou = _iou(b0, b)
        ds = abs(float(s) - float(s0))
        worst_iou, worst_ds = min(worst_iou, iou), max(worst_ds, ds)
        if iou < min_iou or ds > max_score_delta:
            raise SystemExit(
                f"SLICE CONSISTENCY GATE FAILED at slice {i}/{len(slices)}: IoU {iou:.6f} "
                f"(need >= {min_iou}), |score delta| {ds:.6f} (need <= {max_score_delta}).\n"
                f"The engine gives different answers for identical inputs across the batch, so "
                f"batching is NOT safe with it. Re-export with --batch {len(slices)} and rebuild.")
    print(f"slice-consistency gate OK (batch {len(slices)}: worst IoU {worst_iou:.6f}, "
          f"worst |score delta| {worst_ds:.6f})")


def fidelity_report(slice0, s_ref, b_ref, image_path, min_iou=0.99, min_detect=0.30,
                    strict=False):
    """PyTorch fidelity: always REPORTED, fatal only with --strict-parity.

    Non-blocking by default because container TRT 10.9 engines currently deviate from PyTorch
    for reasons unrelated to batching -- boxes stay close (IoU ~0.9637) but confidence scores
    are depressed (0.52-0.82 vs 0.889). A batch-1 engine deviates identically, and the 0.9994
    recorded on 2026-08-03 came from a host TRT 11.2 build. Blocking by default would block
    every build for a pre-existing, separately-tracked problem, so it is printed loudly instead.
    """
    if s_ref < min_detect:
        raise SystemExit(
            f"PARITY IMAGE UNUSABLE: PyTorch top score {s_ref:.3f} < {min_detect} -- the parity "
            f"image '{image_path}' does not contain the prompted classes, so top-box IoU is not "
            f"a valid faithfulness check. Re-run --stage export with an image that clearly shows "
            f"the prompted classes.")
    s, b = slice0
    iou = _iou(b_ref, b)
    ds = abs(float(s) - float(s_ref))
    ok = iou >= min_iou and ds <= 0.01
    print(f"pytorch fidelity [{'OK' if ok else 'DEVIATION'}]: pytorch score={s_ref:.3f} vs "
          f"trt {s:.3f} (|d|={ds:.3f}) | top-box IoU={iou:.4f} (want >= {min_iou}) on {image_path}")
    if not ok:
        print("  ^ NOT blocking. Known open issue: container TRT 10.9 depresses confidence "
              "scores while keeping boxes close; a batch-1 engine deviates identically, so this "
              "is not caused by batching. Pass --strict-parity to make it fatal.")
        if strict:
            raise SystemExit("PARITY GATE FAILED (--strict-parity)")
```

- [ ] **Step 4: Rewire the stages, the CLI and the filenames**

In `stage_export`, change the `export_onnx(model, text, H, W, onnx_path)` call to `export_onnx(model, text, H, W, onnx_path, batch=args.batch)`, and change its final print to mention the batch:

```python
    print(f"\nDONE (export, batch {args.batch}). Now build the engine INSIDE the runtime "
          f"container:\n  python3 gdino_trt_export.py --stage build --hw {H} {W} "
          f"--batch {args.batch} --out <container path>")
```

In `stage_build`, replace the three lines from `ser = build_engine(...)` through the `batch_consistency_gate(...)` call with:

```python
    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16, batch=args.batch)
    slices = run_at_batch(ser, text, ref_img, int(args.batch))
    slice_consistency_gate(slices)
    fidelity_report(slices[0], s_pt, b_pt, ref_image_path, min_iou=args.min_iou,
                    min_detect=args.min_detect, strict=args.strict_parity)
```

In the argparse block, replace the existing three-value `--batch` argument with:

```python
    ap.add_argument("--batch", type=int, default=1,
                    help="the engine's batch = the largest LangSAM worker's camera count. Both "
                         "stages need the SAME value: export traces at it (the traced batch is "
                         "baked into the graph) and build pins the profile to it. Artifacts are "
                         "named _b<N>_ so a mismatched pair cannot be combined by accident.")
    ap.add_argument("--strict-parity", action="store_true",
                    help="make the PyTorch fidelity deviation fatal (default: reported only)")
```

Finally, put the batch in the artifact names — change the `tag = ...` line to:

```python
    tag = f"gdino_swint_{H}x{W}_b{int(args.batch)}_{'fp16' if args.fp16 else 'tf32'}"
```

Leave `npz_path` and `ref_path` exactly as they are: the prompt tensors and the batch-1 PyTorch reference are batch-independent and shared by every batch size.

- [ ] **Step 5: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/docs && python3 -m py_compile gdino_trt_export.py && echo "compile OK"
~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python gdino_trt_export.py --stage build --help 2>&1 | grep -A4 -- "--batch"
grep -c "batch_consistency_gate\|parity_gate" gdino_trt_export.py
```
Expected: `compile OK`; help shows the single-integer `--batch` and `--strict-parity`; the grep returns **0** (both old functions gone, no dangling callers).

- [ ] **Step 6: Update the doc**

In `gdino_trt_export.md`, rewrite the `--batch` explanation in the Stage 2 section and the stage-1 command block so both stages pass the same `--batch N`, and replace the two-gate description. State plainly: the traced batch is baked into the graph, so a camera-count change needs a **re-export**, not just a rebuild; artifacts carry `_b<N>`; the slice-consistency gate blocks and the PyTorch fidelity check only reports (with the TRT 10.9 reason). Keep every existing section that is still true.

- [ ] **Step 7: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): build the GDINO engine at one fixed batch N

The engine's batch is baked at ONNX trace time: a Where in the encoder fusion
attention is only broadcast-conformable at the traced batch, so forcing another
batch fails the build and a multi-size profile silently specialises to a static
shape. --batch is therefore a single integer used by BOTH stages -- export traces
at it, build pins min=opt=max to it -- and artifacts carry _b<N> so a mismatched
pair cannot be combined.

Gates split by what they answer: slice-consistency (blocking, IoU >= 0.999 across
slices of one batched run) proves batching is correct; PyTorch fidelity is reported
but non-blocking, because every container TRT 10.9 engine deviates from PyTorch
identically whether batched or not -- a pre-existing issue that would otherwise
block every build.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task R2: Runtime — pad to the engine's fixed batch

**Files:**
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py` (append)
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_prompt_remap.py` (append tests)
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py`
- Modify: `applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py`

**Interfaces consumed:** `gdino_postprocess_batch`, `build_class_token_masks`, `build_prompt_remap` (already present and unchanged).
**Interfaces produced:** `plan_batch_padding(n_frames, engine_batch) -> int`; `GDinoTrtDetector.engine_batch`; `GDinoTrtDetector.batch_error(n)`.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_prompt_remap.py`, above the `if __name__` runner:

```python
from langsam_helpers import plan_batch_padding


def test_padding_exact_fit_needs_no_padding():
    assert plan_batch_padding(3, 3) == 0


def test_padding_pads_a_smaller_worker():
    assert plan_batch_padding(2, 3) == 1
    assert plan_batch_padding(1, 3) == 2


def test_padding_rejects_more_frames_than_the_engine_batch():
    try:
        plan_batch_padding(4, 3)
    except ValueError as e:
        assert "4" in str(e) and "3" in str(e)
        return
    raise AssertionError("expected ValueError when frames exceed the engine batch")


def test_padding_rejects_empty_and_bad_engine_batch():
    for args in ((0, 3), (3, 0)):
        try:
            plan_batch_padding(*args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % (args,))
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_prompt_remap.py
```
Expected: `ImportError: cannot import name 'plan_batch_padding'`.

- [ ] **Step 3: Implement the helper**

Append to `python/langsam_helpers.py`:

```python
def plan_batch_padding(n_frames, engine_batch):
    """Dummy slices needed to fill a fixed-batch GDINO engine.

    The engine's batch is baked at ONNX trace time, so it runs at exactly `engine_batch`
    images -- never fewer, never more. A worker with fewer cameras pads; the padded slices are
    computed and discarded. Returns the number of pad slices.

    Raises ValueError if there are no frames, if the engine batch is nonsensical, or if there
    are more frames than the engine can take -- the last needs a re-export at the new batch,
    not a runtime workaround.
    """
    n, b = int(n_frames), int(engine_batch)
    if n < 1:
        raise ValueError("no frames to detect")
    if b < 1:
        raise ValueError(f"engine batch must be >= 1, got {b}")
    if n > b:
        raise ValueError(f"{n} frames but the engine is built for batch {b}")
    return b - n
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py && python3 tests/test_langsam_multicam.py
```
Expected: `11/11 passed`, then `7/7 passed`, then `9/9 passed`.

- [ ] **Step 5: Read the engine's fixed batch**

In `python/langsam_common.py`, import `plan_batch_padding` by adding it to the existing `from langsam_helpers import (` block.

Replace the `try/except` that sets `self.max_batch` with:

```python
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
```

- [ ] **Step 6: Rename the error helper and pad in `detect_batch`**

Rename `max_batch_error` to `batch_error` and replace its body's message with one that says a re-export is needed:

```python
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
```

Store `self._engine_path = engine_path` in `__init__` (next to where the engine is opened) so the message can name the directory.

In `detect_batch`, replace the `if n > self.max_batch: raise ValueError(self.max_batch_error(n))` guard with a padding plan, and pad the stacked tensor:

```python
        try:
            pad = plan_batch_padding(n, self.engine_batch)
        except ValueError as e:
            raise ValueError(f"{e}\n{self.batch_error(n)}") from None
```

immediately after `n = len(frames)`. Then, after the line that builds `img` from `torch.cat(chw, dim=0)`, insert the padding:

```python
            if pad:
                # Fixed-batch engine: fill the unused slices. Their outputs are discarded, so
                # the content does not matter -- zeros avoid copying a real frame.
                img = torch.cat([img, torch.zeros((pad,) + tuple(img.shape[1:]),
                                                  device=img.device, dtype=img.dtype)], dim=0)
```

Change the text-tensor lookup from `self._text_for_batch(n)` to `self._text_for_batch(self.engine_batch)`, and change the result loop `for i in range(n)` so it still iterates only the real frames (it already uses `n`, so confirm it does — the padded slices must never be returned).

- [ ] **Step 7: Update the operator guard**

In `python/langsam_multicam_fragment.py`, change the construction-time check to use the new names:

```python
                if len(self.cameras) > self.gdino_trt.engine_batch:
                    raise ValueError(self.gdino_trt.batch_error(len(self.cameras)))
```

- [ ] **Step 8: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_helpers.py langsam_common.py langsam_multicam_fragment.py && echo "compile OK"
grep -rn "max_batch" langsam_common.py langsam_multicam_fragment.py | grep -v "engine_batch" || echo "no stale max_batch references"
python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py && python3 tests/test_langsam_multicam.py
```
Expected: `compile OK`; no stale `max_batch`; `11/11`, `7/7`, `9/9`.

- [ ] **Step 9: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_prompt_remap.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): pad to the GDINO engine's fixed batch

The engine runs at exactly one batch size, baked at ONNX trace time, so a worker
with fewer cameras pads the stacked image tensor and discards the extra slices.
max_batch (an upper bound) becomes engine_batch (an exact value), and the error
now says a re-export is required, not just a rebuild.

plan_batch_padding is a pure numpy-free helper so the arithmetic and its failure
cases are host-testable without TensorRT.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task R3: Produce the artifacts, validate, measure

Verification only. Driven by the controller (needs the host GroundingDINO venv and the running container).

- [ ] **Step 1: Export at batch 3 (host)**

```bash
cd ~/develop/vision/GroundingDINO-TensorRT-and-ONNX-Inference
cp /home/ecku/develop/holoscan/holohub-tcn/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py .
PYTHONPATH=$PWD ~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python gdino_trt_export.py \
  --stage export --prompts floor person --hw 512 672 --batch 3 \
  --out /data/models/active/groundingdino --parity-image images/in/person.jpg
```
Expected: `gdino_swint_512x672_b3_tf32.onnx` written, plus the unchanged npz and parity ref.

- [ ] **Step 2: Build in the container**

```bash
python3 /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
  --stage build --hw 512 672 --batch 3 --out /srv/models/active/groundingdino
```
Expected: `engine written: ..._b3_tf32.engine (batch 3, ~670 MiB)`, `slice-consistency gate OK` with worst IoU ≈ 0.9996, then `pytorch fidelity [DEVIATION]` printed but not fatal.

**STOP if the slice-consistency gate fails** — that means the engine gives different answers for identical inputs, and batching is unsafe. Report and stop.

- [ ] **Step 3: Point the YAML at the new engine**

In `python/tcn_shm_vlm_inference.yaml`, set `gdino_trt_engine` to `/srv/models/active/groundingdino/gdino_swint_512x672_b3_tf32.engine`. Leave `gdino_trt_text` and `gdino_trt_hw` unchanged.

- [ ] **Step 4: Run and confirm masks**

```bash
./holohub run tcn_shm_vlm_inference --run-args='--tracking'
```
Expected: starts (no batch error), all five cameras produce masks, detections/colours unchanged.

- [ ] **Step 5: Profile and compare**

Run profile mode, then on the host against `/tmp/tcn/vlm_inference_profile.nsys-rep` use the per-worker NVTX query from the revision-1 plan's Task 6 Step 3. Compare `gdino` against the baseline **118.0 ms (worker B, 3 cameras)** and **96.5 ms (worker A, 2 cameras)**; expect worker B ≈ 81 ms plus overhead, worker A slightly above its 96.5 ms only if padding dominates.

- [ ] **Step 6: Record results in the spec**

Append a `## Results` section to the revision-2 spec: measured `gdino` per worker before/after, tick period and fps, engine size, and GPU utilisation vs the 51.2% / 62.4% baseline. Commit.

- [ ] **Step 7: Clean up the experiment artifacts**

```bash
rm -f /data/models/active/groundingdino/EXPERIMENT_*
```

---

## Self-Review notes

- **Spec coverage:** design §1 (engine at N, workers pad) → R2 Steps 3-7 + R3; §2 (`--trace-batch`, folded into a single `--batch`) → R1 Steps 1, 4; §3 (single-integer build batch) → R1 Step 2; §4 (split gates) → R1 Step 3; §5 (runtime padding, `engine_batch`) → R2; §6 (carry-over) → untouched by construction. Testing table → R2 Steps 1-4, R3 Steps 2, 4, 5. Risks (re-export on camera-count change; padding waste) → R1 Step 6 doc + R2 Step 6 error text.
- **Naming consistency:** `--batch` is a single int everywhere; `engine_batch` replaces `max_batch` in both files; `batch_error` replaces `max_batch_error`; artifacts are `_b<N>_`. R2 Step 8's grep enforces no stale `max_batch`.
- **Ordering:** R1 and R2 touch disjoint files and could run in either order, but R3 needs both.
