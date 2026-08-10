# TCN Dataset Replayer

`TcnDatasetReplayerOp` replays a preloaded `artekmed_dataset_reader` export in place of the live
SHM subscriber (`TcnShmSubscriberOp`), so pipeline output becomes deterministic and reproducible
across runs -- the prerequisite for any correctness gate (mask IoU thresholds, byte-identity
diffs) that a live camera stream cannot provide.

See the design doc for the full rationale:
`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-10-replay-harness-design.md`
(Part 1 is this operator; Part 2, the harness that wires it into the app, is a later task).

## Overview

Plain Python package (no CMakeLists), imported as:

```python
from operators.tcn_artekmed.tcn_dataset_replayer import TcnDatasetReplayerOp
```

At `start()` it decodes every selected frame for every selected camera into **pinned host
memory**, once. Each `compute()` then does a single H2D copy per camera into a fresh device
array -- no JPEG decoding happens on the hot path, so decode cost never pollutes a timing
measurement.

It emits through the exact same two ports as `TcnShmSubscriberOp`
(`operators/tcn_artekmed/tcn_shm_subscriber/shm_subscriber_op.cpp`), with the same tensor names,
dtypes, and shapes, so it is a drop-in replacement at a `compose()` call site.

## The BGR note (read this before changing channel_order)

`artekmed_dataset_reader`'s `color_image()` decodes JPEGs and returns **RGB**. The live SHM
subscriber instead delivers **BGR** (the raw camera wire format), and every downstream consumer
(`GdinoOp.compute` in `langsam_pipelined.py` / `langsam_multicam_fragment.py`, etc.) does
`[..., [2, 1, 0]]` itself to recover RGB before running inference.

`TcnDatasetReplayerOp` therefore defaults to `channel_order="bgr"` and reverses the reader's
channel axis **once, at preload time**, so its output is byte-identical to what the subscriber
would have produced for the same frame.

**Do not "fix" this to emit RGB.** Emitting RGB directly would make every consumer's
BGR->RGB swap turn an already-correct image into a channel-swapped one: plausible-looking,
silently wrong, and it would invalidate every gate built on top of this operator's output. If you
ever need the reader's raw RGB output unmodified (e.g. to compare against the on-disk JPEGs
directly), pass `channel_order="rgb"` explicitly and be aware the pipeline under test will then
see the wrong channel order.

## Ports

| port | contents |
|---|---|
| `color_outputs` | dict of `{f"{camera_id}_colorimage": <device uint8, shape (H, W, 3)>}` |
| `depth_outputs` | dict of `{f"{camera_id}_depthimage": <device uint16, shape (H, W, 1)>}` when `emit_depth`, else an empty dict (kept wired so a downstream operator that always reads it does not break) |

## Parameters

| param | default | meaning |
|---|---|---|
| `dataset_path` | — (required) | passed to `artekmed_dataset_reader.open_dataset` |
| `cameras` | all cameras in the dataset, sorted | camera ids, in order; the emitted tensor name is `<id>_colorimage` / `<id>_depthimage` |
| `start_frame` | `0` | **index** into the dataset's ascending frame-number list (not a frame number) |
| `frame_count` | `None` (all) | number of frames to select, applied after `frame_step` |
| `frame_step` | `1` | stride over the dataset's frame-number list |
| `loop` | `True` | cycle the preloaded selection indefinitely across `compute()` calls |
| `emit_depth` | `False` | the VLM app only consumes colour; set `True` to also preload/emit depth |
| `channel_order` | `"bgr"` | see above -- `"bgr"` is what makes the operator faithful to the subscriber |
| `playback` | `"auto"` | `"auto"`: emit one frame group per `compute()`, bounded externally by e.g. `CountCondition`. `"manual"`: hold the `async_condition` and only emit after `request_next()` is called (step-through debugging) |
| `max_preload_frames` | `64` | guard against pointing this at a large export; raises `ValueError` if the selection is larger |
| `async_condition` | `None` | a `holoscan.conditions.AsynchronousCondition`, required (and only used) when `playback="manual"` |
| `allocator`, `cuda_stream_pool` | `None` | accepted and **ignored** -- present only so this operator is a drop-in replacement at `TcnShmSubscriberOp`'s construction site; pinned/device memory is managed directly via cupy |

`frame_index` (the source dataset frame **number** most recently emitted) and `loop_count` (how
many times the preloaded selection has fully wrapped) are exposed as read-only properties, so a
harness can label dumps with the true source frame number rather than a raw tick counter --
essential once `loop=True` and the run outlives one pass through the dataset.

## Frame selection

The exact list of source frame numbers a run selects (before looping) is computed by the pure,
host-testable `plan_frame_sequence` in `_planning.py` -- see `tests/test_planning.py`. `start` is
always an **index** into the dataset's frame-number list, never a frame number itself; the tests
deliberately use a non-contiguous frame-number list (`[10, 20, 30, 40]`) so an index/frame-number
mix-up fails loudly instead of silently passing on a dataset where index and frame number happen
to coincide.
