# Deterministic replay harness (design)

A reusable dataset-replay operator that impersonates `TcnShmSubscriberOp`, plus a harness that
turns "does this change alter the output?" from an eyeball judgement into a script that returns
pass/fail.

## Motivation

Every optimisation from here on is gated on a correctness question the live pipeline cannot answer:

- **4a (batched SAM decode)** is measured at +9.7% but still `false` by default, because its gate is
  per-mask IoU ≥ 0.999 and there is no way to compare masks between two runs of a live stream.
- **CUDA graphs** has a *byte-identity* gate — strictly impossible against a live source.
- **A1 (GDINO FP16)** is the largest remaining win (−20 to −24 ms of GPU work) and its entire risk
  is a **quality** regression. Quality is exactly what we currently cannot measure.
- Any change below ~5% is indistinguishable from scene drift.

The pipelining refactor's correctness was never formally verified either. This is the blocker.

## Dataset facts (verified on the host, 2026-08-10)

`~/develop/artekmed/artekmed_test_data/data/` — in the container at
`/workspace/volumes/artekmed_test_data/data/`.

| | `orbbec_capture` | `k4a_capture` |
|---|---|---|
| resolution | 1920×1080 | **2048×1536** |
| cameras | camera01–camera04 | camera01–camera04 |
| frames | 6 | 6 |
| colour size on disk | 6.7 MB | 14 MB |
| format | JPEG, RGB | JPEG, RGB |

Consequences that shape the design, and that must not be glossed over:

1. **`k4a_capture` matches the live camera resolution (2048×1536); `orbbec_capture` does not.** For
   perf-representative runs use k4a. Either is fine for correctness A/B. The path is a parameter, so
   this is a default, not a constraint.
2. **4 cameras, but production runs 5** (`camera01..camera05_colorimage`, split 2 on dev0 / 3 on
   dev1). The harness therefore runs a **2+2 split**, which needs batch-2 engines on both workers
   instead of b2+b3. **Absolute harness timings are NOT comparable to the live 2/3 numbers** — the
   period baselines (171.0 ms etc.) stay live-measured. The harness measures *relative* change under
   identical conditions, which is all a gate needs.
3. **6 frames is enough for a correctness gate but not for steady state**, so the replayer loops.
   Looping preloaded frames gives arbitrarily long deterministic runs.
4. The reader returns **RGB** (`load_color_image` → `Image.convert("RGB")`) but the pipeline's
   consumers do `[..., [2,1,0]]` because the SHM subscriber delivers **BGR**. See §1.3 — this is the
   single detail most likely to silently invalidate everything.

## Part 1 — `TcnDatasetReplayerOp`

New reusable operator: `operators/tcn_artekmed/tcn_dataset_replayer/`.

**Python, not C++**, because `artekmed_dataset_reader` is a pure-Python package. This follows the
existing precedent in that tree (`tcn_util`, `tcn_processing`, `tcn_shm_io` are all Python packages
with an `__init__.py` and a `metadata.json`, imported as
`from operators.tcn_artekmed.tcn_dataset_replayer import TcnDatasetReplayerOp` — no CMake).

### 1.1 It must be drop-in for the subscriber

`TcnShmSubscriberOp` (`shm_subscriber_op.cpp`) emits:

| port | contents |
|---|---|
| `color_outputs` | one entity, N tensors named `<camera_id>_colorimage`, **uint8**, shape `(H, W, C)`, device memory |
| `depth_outputs` | one entity, N tensors named `<camera_id>_depthimage`, **uint16**, shape `(H, W, C)`, device memory |

Consumers do `msg.get("camera01_colorimage")` then `cp.asarray(t)`, so each value must be a device
tensor exposing `__cuda_array_interface__`. Emitting a `dict` of cupy arrays from a Python operator
produces the equivalent TensorMap.

The replayer takes the same `allocator` and `cuda_stream_pool` arguments even where it does not need
them, so swapping it in is a one-line change in `compose`.

### 1.2 Preload, then upload

Decode cost must not enter the measurement. PIL decoding four 2048×1536 JPEGs per tick would
dominate everything.

- At `start()`, decode **every** selected frame for **every** selected camera into **pinned** host
  memory (`cp.cuda.alloc_pinned_memory` or `torch.empty(..., pin_memory=True)`).
- Per `compute()`, issue one H2D copy per camera into freshly allocated device tensors.
- Budget: 6 frames × 4 cameras × 2048×1536×3 = **226 MiB** pinned for k4a. Trivial. A `max_frames`
  cap guards against someone pointing this at a large export.

Pinned memory also matches the subscriber's H2D pattern, so the harness exercises a realistic
upload path rather than an unrealistically cheap one.

### 1.3 Channel order — the correctness trap

The reader returns RGB; the subscriber delivers BGR; consumers swap BGR→RGB themselves.

**The replayer must therefore emit BGR** (i.e. reverse the reader's channels once, at preload time,
for free). Emitting RGB would make every consumer swap it to BGR, and GDINO would detect on
channel-swapped images — plausible-looking output, quietly wrong, and it would corrupt every gate
built on top of it.

Expose `channel_order: "bgr" | "rgb"`, default **`"bgr"`**, and state in the docstring that `bgr`
is what makes it faithful to the subscriber. Assert the array is 3-channel before swapping.

### 1.4 Playback control

Two modes, because the harness and interactive debugging want different things:

- **`playback: "auto"`** (default) — emit one frame group per `compute()`. The app bounds the run
  with `CountCondition(app, count=N)`. This is what the harness uses.
- **`playback: "manual"`** — hold an `AsynchronousCondition` (the same mechanism the subscriber
  uses) and emit only when `request_next()` is called, for step-through debugging.

Both are deterministic: frame order is a fixed list, and nothing depends on wall-clock time.

### 1.5 Parameters

| param | default | meaning |
|---|---|---|
| `dataset_path` | — | passed to `open_dataset` |
| `cameras` | all | camera ids, **in order**; the emitted tensor name is `<id>_colorimage` |
| `start_frame` / `frame_count` / `frame_step` | 0 / all / 1 | frame selection |
| `loop` | `True` | cycle the preloaded frames indefinitely |
| `emit_depth` | `False` | the VLM app only consumes colour |
| `channel_order` | `"bgr"` | §1.3 |
| `playback` | `"auto"` | §1.4 |
| `max_preload_frames` | 64 | guard against huge exports |

`frame_index` and `loop_count` are exposed as read-only attributes so a harness can label dumps
with the true source frame number rather than a tick counter — essential when `loop` is on.

### 1.6 What is host-testable

The reader needs `pyarrow`/`tifffile`/`PIL`, and emitting needs cupy — so the operator itself is
container-only. But the **pure frame-plan arithmetic** is host-testable and is where off-by-one bugs
live: `plan_frame_sequence(frame_numbers, start, count, step, loop, ticks)` → the exact list of
source frame numbers a run will emit. Put it in a `_planning.py` beside the operator with numpy-only
tests, mirroring how `langsam_helpers.py` is split out from `langsam_common.py`.

## Part 2 — the harness

### 2.1 Replayer source in the app

Add `source: "shm" | "dataset"` to the app config (default `"shm"`, so nothing changes for live
runs) plus a `dataset_source` block (`path`, `cameras`, `frame_count`, `loop`). In `compose`, build
either the subscriber or the replayer and wire the same two edges.

This mirrors the existing `gdino_backend` / `sam_backend` / `pipelined` / `sam_batched_decode`
toggle precedent: opt-in, one config line, instantly revertible.

### 2.2 Mask dump

A small `MaskDumpOp` in the app (not in `operators/` — it is harness-specific) that takes the
collector's mask map and writes `<out_dir>/frame<NNNNNN>_<camera>.npy` per camera per tick, labelled
by **arrival index** — the order in which mask maps actually reach `MaskDumpOp`, not the replayer's
live `frame_index`. See §Results for what this corrected and why it mattered;
`index_manifest.tsv` in `<out_dir>` records the arrival-index -> source-frame mapping.

Panoptic maps are packed integers (`class << 8 | instance`), so `.npy` round-trips them exactly and
comparison needs no tolerance on the container side.

Guard it behind `mask_dump_dir` being set; unset means the operator is not built at all, so a live
run pays nothing.

### 2.3 Compare script

`docs/compare_mask_dumps.py A_dir B_dir`, numpy-only, host-runnable:

- **exact mode** (default): report the fraction of differing pixels per frame/camera; exit non-zero
  on any difference. This is the gate for CUDA graphs and for any refactor claiming byte-identity.
- **`--iou-gate 0.999`**: decompose each packed map into per-class masks, report per-class IoU and
  the min across all frames/cameras; exit non-zero if any falls below the threshold. This is the
  gate for 4a and for A1 (FP16).
- Always report: frames compared, cameras compared, per-class instance counts on both sides, and
  **which** frame/camera/class was worst. A gate that says only "FAIL" wastes the next hour.
- Fail loudly on a shape/frame-set mismatch rather than silently comparing an intersection.

### 2.4 Procedure the harness enables

```
run A: source=dataset, <feature>=false, mask_dump_dir=/tmp/A   -> N frames, exits
run B: source=dataset, <feature>=true,  mask_dump_dir=/tmp/B   -> N frames, exits
python3 docs/compare_mask_dumps.py /tmp/A /tmp/B [--iou-gate 0.999]
```

Deterministic on both sides, so a difference means the change did it.

## Testing

| test | where | gate |
|---|---|---|
| `plan_frame_sequence` | host, numpy | new; start/count/step/loop and the wrap boundary, against an independent oracle |
| existing host suites | host | 8/8, 11/11, 7/7, 19/19 unchanged |
| `compare_mask_dumps.py` | host | synthetic arrays: identical → pass; one flipped pixel → fail exact, pass IoU; a whole class missing → fail both |
| `source: "shm"` unchanged | container | live path untouched |
| replayer emits BGR | container | dump one frame, compare against the reader's array reversed on axis −1 |
| replay is deterministic | container | two identical runs → `compare_mask_dumps` exact-equal. **This is the harness's own gate: if it cannot reproduce itself, it cannot gate anything.** |

## Results

**Self-consistency (2026-08-10).** Two identical dataset runs, 16 dumps each, compared with
`compare_mask_dumps.py` in exact mode: **0 of 50,331,648 pixels differing, instance counts
identical.** This is the check §Testing calls "the harness's own gate: if it cannot reproduce
itself, it cannot gate anything" — it passed. The harness can be used to gate other changes.

Determinism held under looping too: with `loop: true`, the same source frame on its second pass
through produced bit-identical output — `frame000000` and `frame000006` showed identical diff
counts throughout the comparison.

**MaskDumpOp arrival-index correction.** §2.2's dumps were originally labelled with the replayer's
live `frame_index` attribute. That is wrong: `frame_index` advances as the replayer *emits* frames,
but a mask map reaches `MaskDumpOp` several ticks later, after gdino/sam/panoptic have processed
it — the replayer's live counter runs roughly 2 frames ahead of the mask that is actually arriving.
So a file named `frame000006_camera01.npy` was not frame 6's mask; it was an earlier frame's mask
carrying the replayer's *current* position as its label.

This is dangerous rather than cosmetic: the offset **equals pipeline latency**, and pipeline latency
is not a constant — it varies with configuration (monolithic vs pipelined, batch size, stage count).
Two runs of the *same* configuration can get the same offset by construction and appear to agree,
hiding the bug; two runs of *different* configurations would silently compare mismatched frames
while the matching filenames imply alignment.

The fix: dumps are now labelled by **arrival index** — the order in which mask maps actually reach
`MaskDumpOp` — and `index_manifest.tsv` is written alongside them recording the arrival-index ->
source-frame mapping, so the true frame number stays recoverable without the comparison depending
on it.

## Risks

- **Non-determinism elsewhere in the pipeline** would defeat the harness. §Testing's
  self-consistency check is what detects it. Known suspects if it fails: `EventBasedScheduler`
  thread interleaving affecting reduction order, and any use of uninitialised memory. If two
  identical runs differ, **stop and investigate** — do not proceed to gate other work.
- **4 cameras / 2+2 split** means new batch-2 engines may be needed before the harness runs at all;
  check which engines exist before building anything.
- **Looping 6 frames** is not a varied scene. Detection counts stay realistic but cache behaviour is
  friendlier than live. Acceptable for gates; not a substitute for live perf numbers.
- **Pinned allocation of 226 MiB** at startup, held for the run.

## Out of scope

- Replacing live perf measurement. Absolute baselines stay live-measured (§Dataset facts 2).
- Depth-consuming apps: `emit_depth` exists and is faithful, but only the colour path is exercised.
- Automating the A1/B2 work itself.
