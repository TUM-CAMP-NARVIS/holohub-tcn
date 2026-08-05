# Deferred findings — LangSAM / VLM inference

Known issues that code review raised and we consciously chose **not** to fix at the time. None is
a known-broken behaviour; they are risks, rough edges and dead code. Recorded here because the
review ledgers they came from live in git-ignored scratch and would otherwise be lost.

Every entry below was **re-verified against the code on 2026-08-05** — items that had since been
fixed were dropped rather than carried forward. If you fix one, delete its entry.

Not repeated here (they have their own homes):

- The **TRT 10.9 confidence-score depression** — boxes right (IoU ~0.964), scores 0.52 vs
  PyTorch's 0.889. The likeliest cause of corner-case detection misses. See
  [`specs/2026-08-04-gdino-batched-inference-design.md`](./specs/2026-08-04-gdino-batched-inference-design.md).
- The remaining **performance levers** (stage pipelining, the SAM decode loop) — see
  [`optimization-playbook.md`](./optimization-playbook.md) §4 and
  [`specs/2026-08-05-per-worker-engines-design.md`](./specs/2026-08-05-per-worker-engines-design.md).

---

## 1. Correctness risk — low probability, real consequence

### 1.1 A legacy range-profile engine silently degrades to batch 1

`python/langsam_common.py`: `engine_batch` is read from the profile **minimum**. An engine built
before the fixed-batch work (e.g. with `--batch 1 3 5`) therefore reports `engine_batch = 1` and
rejects every multi-camera worker at construction, with only a printed warning explaining why.
Intended behaviour, but surprising for any pre-existing engine.

*Fix (optional):* warn loudly when profile min != max, naming both.

### 1.2 No minimum-batch guard

`docs/gdino_trt_export.py` accepts `--batch` values that make `min > 1`, but the runtime only
reads the profile's min as an exact batch. An engine built that way would fail inside TensorRT
with a cryptic shape error for any worker below that batch.

*Fix:* read both ends of the profile and check `n < min_batch` alongside the existing max check.

### 1.3 An unrecognised backend value silently runs the eager path

Both `sam_backend` and `gdino_backend` compare against the literal `"trt"`, so a typo
(`"tensorrt"`) quietly gives you the PyTorch path — you would only notice from the framerate.

*Fix:* validate against the known set and raise on anything else.

### 1.4 `set_prompts` mutates before committing its cache key

`python/langsam_common.py`: `token_class_ids` / `num_classes` / `_class_masks` are assigned before
`_prompt_key`. If `build_class_token_masks` raised, a retry with the old prompt set would
early-return on the stale key and leave `_class_masks` inconsistent with `num_classes`. The
realistic path is safe — `build_prompt_remap` validates and raises *before* any mutation — so the
only window is a GPU allocation failure, which is unrecoverable anyway.

*Fix:* assign to locals, commit all four at the end.

---

## 2. Error quality and ergonomics

### 2.1 `gdino_trt_export.py --help` still describes the old single "parity gate"

The `ArgumentParser(description=...)` and the `--stage` help text predate the split into a
blocking `slice_consistency_gate` plus a non-blocking `fidelity_report` (and now also
`image_independence_gate`). `--help` does not mention the blocking checks at all.

### 2.2 A swallowed TensorRT exception yields a misleading message

`python/langsam_common.py`: the `except Exception` around `get_tensor_profile_shape` now logs the
cause, but still falls back to `engine_batch = 1`, after which the user is told "the engine is
built for batch 1" — false, and pointing at a rebuild that will not help.

### 2.3 Hard-coded engine output names give a bare `KeyError`

`SamTrtEncoder.encode` discovers output tensors dynamically for allocation but then indexes three
literal names. A mis-named engine fails with `KeyError` instead of the actionable rebuild message
the class otherwise invests in.

### 2.4 `SAM(device=None)` plus `sam_backend: "trt"` raises a bare `TypeError`

Unreachable from `langsam_multicam_fragment.py`, which always passes a device.
`GDinoTrtDetector` has the same shape, so at least they are consistent.

---

## 3. Dead code and cosmetics

- **`GDinoTrtDetector.detect()` has no caller** — kept deliberately as a single-frame convenience
  wrapper, but it is currently API for nobody.
- **`self.token_class_ids` / `self.num_classes`** are written in `set_prompts` and never read
  outside it; `detect_batch` uses `_class_masks` only.
- **`gdino_postprocess` is re-exported from `langsam_common`** with no runtime consumer — the
  tests import it directly from `langsam_helpers`, and the PyTorch backend uses HF's
  `post_process_grounded_object_detection`. It is now the *oracle* for the batched version's
  equivalence test, which is worth saying in its docstring.
- **`model = p.model`** is dead in the TRT branch of `SAM._set_image_batch_gpu`.
- **`worker_batch` is imported but unused** in `langsam_multicam_fragment.py` (the file computes
  `self.batch = len(self.cameras)` inline).
- **`plan_batch_padding`'s `n < 1` branch is unreachable** from `detect_batch`, which returns
  early on an empty frame list. Harmless as a general-purpose guard.
- **`python/tests/test_prompt_remap.py`** has a second `from langsam_helpers import ...` in the
  middle of the file; fold it into the top import block on a future touch. Also,
  `test_padding_rejects_empty_and_bad_engine_batch` catches a bare `ValueError` for both cases
  without distinguishing the messages, so a helper raising the wrong one would still pass.
- **`self.prompts` on the detector keeps first-seen spelling** — a second `set_prompts` differing
  only in case or whitespace hits the normalised cache key and early-returns. Cosmetic; the
  operator uses its own list for labels.

---

## 4. Performance, measured and deliberately not pursued

- **`cp.asarray(keep)` is one host→device copy per detecting camera** in `detect_batch`. Async on
  the CuPy stream, so it does not break the two-synchronisation budget, but the whole batch's
  indices could go in one transfer with a concat plus offsets. Only worth it if it shows on a
  profile.
- **`build_prompt_remap` is O(n²)** — `p in active` followed by `active.index(p)` per baked
  prompt. Runs once per prompt change over a handful of classes.
- **The operator's `hw` is the last camera's shape** (`langsam_multicam_fragment.py`) and is then
  used for *every* camera's panoptic map. **Pre-existing**, and harmless while all five cameras
  are 2048×1536 — but `detect_batch` now carefully carries per-frame `(H0, W0)` through to pixel
  scaling, so the per-camera contract holds right up to the operator boundary and then collapses.
  Worth a comment at minimum, so nobody assumes the operator is resolution-agnostic.
