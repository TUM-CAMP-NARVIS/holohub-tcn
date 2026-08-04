# Batched Grounding DINO inference (design)

Run the Grounding DINO TensorRT engine once per camera **worker** instead of once per camera.

> **Revision 2 (2026-08-04, after hardware validation).** Revision 1 was written from a profile
> reading that turned out to be wrong, and proposed an engine shape TensorRT will not build for
> this model. Both are corrected below; the sections that changed are marked. Revision 1's
> Tasks 1-5 are implemented, reviewed and committed — they remain correct, but they are no
> longer where the win comes from.

## What revision 1 got wrong

**1. The overhead estimate.** Revision 1 read the nsys `myelinGraphExecute` range (5.96 ms x 2
per camera = 11.9 ms) as the engine's total execution time, and concluded that the remaining
~27 ms of the 39.3 ms per-camera `gdino` stage was removable overhead from ~5 GPU->CPU
synchronisations. That was wrong: `myelinGraphExecute` covers only the Myelin-fused subgraphs.
Direct benchmarking (median of 20, container TRT 10.9.0.34) measures **38.64 ms per batch-1
execution** — essentially the entire stage. There is very little sync overhead to remove.

The sync-elimination work (batched decode, hoisted class-token masks, single device->host
transfer) is correct and harmless, and it still removes ~5 syncs per camera. It is simply not
worth the ~34 ms/tick that revision 1 attributed to it.

**2. The engine shape.** Revision 1 specified one batch-dynamic engine, `min 1 / opt 3 / max 5`.
TensorRT will not produce that for this model. Measured:

| ONNX traced at | build result | engine accepts |
|---|---|---|
| batch 1 | succeeds, silently specialises | **only** batch 1 |
| batch 1, forced `min=opt=max=2` | **fails**: `IISelectLayer /transformer/encoder/fusion_layers.0/attn/Where_1: broadcast dimensions must be conformable` | — |
| batch 3 | succeeds | **only** batch 3 (`set_input_shape` returns False for 1, 2, 5) |

`torch.onnx.export`'s `dynamic_axes` declares `batch_size` symbolic — the ONNX and the parsed
TensorRT network both show `(-1, 3, -1, -1)` — but a `Where` in the text-image fusion attention
bakes a broadcast that is only conformable at the traced batch. **The trace's batch is the
engine's batch.** A profile spanning several batch sizes is not achievable; TensorRT either
fails the build or silently specialises to a static shape.

## What the win actually is

Same benchmark, both engines, pure engine execution:

| | per execution | 3 cameras |
|---|---|---|
| batch-1 engine | 38.64 ms | **115.91 ms** (3 executions) |
| batch-3 engine | 81.03 ms | **81.03 ms** (1 execution) |

**34.88 ms/tick saved on a 3-camera worker (30% of engine time)** — because the batched engine
is more efficient per camera (27.0 vs 38.6 ms), not because syncs were removed.

Worker B (3 cameras, GPU 1) sets the frame rate at 97% of a 221.8 ms tick, so this is roughly
**4.5 -> 5.3 fps, about +19%**.

Batching is numerically sound. All three slices of a batch-3 run are identical to each other and
match the batch-1 engine at **IoU 0.999611** (score delta 0.0055) on the parity image.

## Design (revised)

### 1. One engine, built at exactly N = the largest worker's camera count

`N = max(len(w.cameras) for w in langsam_multicam.workers)` — 3 for the current 2/3 split.
Every worker uses the same engine and **pads** its frame list to N.

Padding costs the smaller worker a little: worker A runs 3 slices for 2 cameras, 81.0 ms instead
of today's 77.3 ms. It sits at 73.6% busy and does not set the frame rate, so the system still
gains. The alternative — one engine per distinct camera count — is optimal for both workers but
doubles exports, builds and artifacts (~2.7 GB); rejected as not worth it for ~4 ms on a
non-bottleneck worker.

### 2. Export stage: `--trace-batch N`

The host export stage traces the dummy inputs at batch N (image `(N,3,H,W)`, text tensors
repeated to N). Everything else is unchanged, including the requirement to call
`unset_image_tensor()` before tracing.

Filenames encode the batch so a mismatched pair cannot be combined silently:

```
gdino_swint_<H>x<W>_b<N>_tf32.onnx
gdino_swint_<H>x<W>_b<N>_tf32.engine
gdino_swint_<H>x<W>_parity_ref.npz     (unchanged — the reference is batch-1 PyTorch)
gdino_swint_prompts.npz                (unchanged)
```

### 3. Build stage: `--batch N` (a single integer, replacing `MIN OPT MAX`)

Profile `min = opt = max = N`, matching the ONNX's traced batch. Revision 1's three-value
`--batch` and its "widen the profile" rationale are removed — they describe a shape TensorRT
will not build.

### 4. Gates (revised)

Two checks, because they answer different questions and have different thresholds:

- **Slice-consistency gate (strict, blocking).** Run the parity image replicated to N; every
  slice must agree with slice 0 to IoU >= 0.999 and score delta <= 0.01. This is what proves
  batching is correct. Measured 0.999611 / 0.0055, so it passes with margin.
- **PyTorch fidelity check (reported, non-blocking by default).** Compare slice 0 against the
  stage-1 PyTorch reference and print the IoU and score delta. `--strict-parity` makes it fatal.

The fidelity check is non-blocking because it currently fails for reasons unrelated to
batching: **every container-built (TRT 10.9) engine has depressed confidence scores** — IoU
~0.9637 and score 0.52-0.82 against PyTorch's 0.889, while boxes stay close. A batch-1 engine
fails it identically, and the 0.9994 recorded on 2026-08-03 came from a **host TRT 11.2** build.
Making it blocking by default would block every build for a pre-existing, separately-tracked
problem. It is printed loudly on every build so it cannot be forgotten.

> Open issue, out of scope here: why TRT 10.9 depresses scores. Boxes remain accurate, so strong
> detections clear `box_threshold: 0.3` comfortably and masks look correct, but marginal
> detections may be lost. First experiments to try: build with TF32 disabled, and check whether
> the INT64 inputs TensorRT warns about (`input_ids`, `position_ids`, `token_type_ids`) are
> being truncated to INT32.

### 5. Runtime: pad to the engine's exact batch

`GDinoTrtDetector` reads the engine's batch from its profile (min == max == N) and exposes it as
`engine_batch`. `detect_batch(frames)`:

- `len(frames) > engine_batch` -> raise, naming the rebuild command (as today).
- `len(frames) < engine_batch` -> pad the stacked image tensor to N, run, and **return only the
  first `len(frames)` results**. Padded slices are never surfaced.
- Text tensors are cached at N only (one entry, not per batch size).

`LangSamBatchOp.__init__` keeps its construction-time guard, now against `engine_batch`.

### 6. What carries over unchanged from revision 1

`gdino_postprocess_batch`, `build_class_token_masks`, `build_prompt_remap`, `set_prompts`, the
single device->host transfer, `_apply_prompts`, and the operator calling `detect_batch` once per
tick. All implemented, reviewed and committed; all still correct. Section 5 of revision 1
(prompt updates) is unaffected and remains in force.

## Testing

| test | where | gate |
|---|---|---|
| existing host suites (`test_gdino_postprocess`, `test_prompt_remap`, `test_langsam_multicam`) | host, numpy | must stay 7/7, 7/7, 9/9 |
| padding logic: N-camera engine with fewer frames returns exactly `len(frames)` results, in order | host, numpy-only fake | new |
| slice-consistency gate | container, `--stage build` | IoU >= 0.999 across slices |
| end-to-end masks | container | detections/colours unchanged |
| `gdino` NVTX stage | container, nsys | vs 118.0 ms (worker B) / 96.5 ms (worker A) |

## Risks

- **A camera-count change requires a re-export, not just a rebuild**, because the batch is baked
  at trace time. The filename encodes N and the runtime guard fails fast, so the failure mode is
  loud rather than silent.
- **Padding wastes compute on smaller workers.** Bounded and measured (~4 ms on worker A).
- The fidelity gap above remains open and is deliberately not blocking.

## Out of scope

- The TRT 10.9 score-depression investigation.
- Reclaiming `sam_compile` via fixed-batch padding (~18 ms/tick).
- Per-worker engines.
- Adding a `text_prompts` port to the multicam path.
