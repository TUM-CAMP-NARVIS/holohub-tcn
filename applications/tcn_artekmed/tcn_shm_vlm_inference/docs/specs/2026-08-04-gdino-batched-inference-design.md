# Batched Grounding DINO inference (design)

Batch the Grounding DINO TensorRT engine across a worker's cameras, and batch the
post-process with it, so one Holoscan tick issues **one** engine execution and **two** device
syncs instead of one execution and ~5 syncs *per camera*.

Follows [2026-08-02-gdino-trt-integration-design.md](./2026-08-02-gdino-trt-integration-design.md),
which brought the engine up. This spec is about the time *around* the engine.

## Motivation (measured)

From the 2026-08-04 nsys trace (588.3 s steady state, 5072 ticks, 2-GPU split with
2 cameras on GPU0 / 3 on GPU1, `sam_compile: false`):

| observation | value |
|---|---|
| pipeline | 5 cameras @ ~4.3-4.5 fps, tick period 221.8 ms |
| bottleneck | worker B (3 cam, GPU1): 215.2 ms of stage work per 221.9 ms period = **97% saturated** |
| `gdino` stage | 118.0 ms (B) = **39.3 ms/camera**; 96.5 ms (A) = 48.2 ms/camera |
| TRT engine execution | `myelinGraphExecute` 5.96 ms x 2 per camera = **~11.9 ms/camera** |
| **non-engine time** | **~27 ms/camera on B, ~36 ms on A** — the target of this spec |
| GPU utilisation | GPU0 51.2%, GPU1 62.4% — neither saturated, so this is CPU/latency-bound |

The 2-GPU split itself is healthy (99.8% overlap between workers; GDINO in TRT is no longer
GIL-bound), and the engine hit its predicted ~38 ms/camera. The remaining cost is per-camera
CPU round-trips, not compute.

### Where the syncs are, per camera, today

1. `langsam_common.py:597` — explicit `torch.cuda.current_stream().synchronize()` after
   `execute_async_v3`.
2. `langsam_helpers.py:119` — `bool(mask.any())` once **per class**, a D2H on
   `token_class_ids`, which is a compile-time constant that never changes.
3. `langsam_helpers.py:124` — `boxes[keep]`; cupy boolean indexing must size the output, so it
   syncs.
4. `langsam_common.py:603` — `cls.get()`.

On the 3-camera worker that is ~15 syncs per tick.

## Design

### 1. Export tool: batch-dynamic optimization profile

`docs/gdino_trt_export.py`, build stage only.

`build_engine` takes a `batch=(min, opt, max)` triple, exposed as `--batch MIN OPT MAX`
(default `1 3 5`), and applies it to every binding:

```
img              (1,3,H,W) / (3,3,H,W) / (5,3,H,W)
input_ids        (1,L)     / (3,L)     / (5,L)
attention_mask   (1,L)     / (3,L)     / (5,L)
position_ids     (1,L)     / (3,L)     / (5,L)
token_type_ids   (1,L)     / (3,L)     / (5,L)
text_token_mask  (1,L,L)   / (3,L,L)   / (5,L,L)
```

`opt=3` matches the bottleneck worker; `max=5` covers any split of the five cameras, so
changing the worker assignment needs no rebuild. One engine file, so the YAML keys
(`gdino_trt_engine`, `gdino_trt_text`, `gdino_trt_hw`) are unchanged.

**No host re-export.** The ONNX already declares `batch_size` dynamic on all six inputs and
both outputs (`gdino_trt_export.py:90-99`); only the engine profile pinned batch=1. This is a
container-side `--stage build` re-run.

### 2. Export tool: batch-consistency gate

After the existing batch-1 parity gate, run the same parity image replicated to batch `opt`
and require **every** output slice to agree with the batch-1 result. Batch>1 may select
different TRT kernels, so bit-exactness is not a valid requirement; the gate reuses the same
criterion as the batch-1 parity gate instead:

- top-box IoU per slice ≥ `--min-iou` (0.99), via the existing `_top_box` / `_iou` helpers
- top-score delta per slice ≤ 0.01

Abort the build on any slice failing either. This is the detector for the primary risk below,
so it must run on every build, not behind a flag.

This exists because of the primary risk below. It is cheap and reuses the stage-1 reference.

### 3. Runtime: `GDinoTrtDetector.detect_batch`

```python
detect_batch(frames: list[uint8 CUDA (H0,W0,3)]) -> list[(xyxy_gpu, class_ids: list[int], scores_gpu)]
```

- resize + normalize each frame to `(3,H,W)`, `torch.stack` → `(N,3,H,W)`
- text tensors expanded to batch `N`, cached per `N` (`self._text_batched[N]`) so the expand
  and `.contiguous()` happen once per batch size, not once per tick
- one `set_input_shape` pass, one `execute_async_v3`, **one** `synchronize()`
- outputs `(N,900,256)` logits and `(N,900,4)` boxes

`detect(frame)` is retained as `detect_batch([frame])[0]`.

Each frame's own `(H0,W0)` is carried through so box→pixel scaling stays per camera (frames
are the same size today, but the per-camera contract is kept).

Fail fast in `__init__` if the engine's profile max batch is smaller than the worker's camera
count, naming the `--batch` rebuild command — the same ergonomics as the existing TRT version
mismatch message.

### 4. Runtime: batched post-process

New `gdino_postprocess_batch(logits, boxes, ...)` in `langsam_helpers.py` — the numpy/cupy
agnostic module, so it unit-tests on the host.

- **Constant per-class token masks hoisted** into `GDinoTrtDetector.__init__`: the
  `tcid == c` masks and their `.any()` are computed once at construction, removing sync (2)
  entirely.
- Scores, per-class max, argmax and threshold computed for all `N` images in one set of GPU
  ops.
- **One D2H per tick**: transfer `best_cls` and `best_score`, both `(N,900)` (~14 KB), and do
  the keep-index arithmetic on the host; then gather boxes on the GPU by index. This replaces
  syncs (3) and (4). Boxes never leave the device, so SAM's `_prep_prompts` contract is
  unchanged.

The existing per-image `gdino_postprocess` stays (it is the reference the batched version is
tested against, and the pytorch backend path may still use it).

### 5. Caller

`langsam_multicam_fragment.py:121-128`: the per-camera `for i, im in enumerate(rgb_gpu)` loop
becomes a single `detect_batch(rgb_gpu)` call; the existing partition-into-SAM logic is
unchanged.

## Testing

| test | where | gate |
|---|---|---|
| `gdino_postprocess_batch` vs looping `gdino_postprocess` | host, numpy, no GPU | numerically identical for random logits/boxes, several N and threshold values, including the zero-detection case |
| batch-consistency | container, `--stage build` | batch-`opt` slices match batch-1 |
| end-to-end masks | container | detections/colours visually unchanged vs the current engine |
| `gdino` NVTX stage | container, nsys | compare against today's 118.0 ms (B) / 96.5 ms (A) |

## Risks

**Primary: the ONNX was traced at batch 1.** GroundingDINO contains many reshape/view ops that
can bake a literal batch dimension despite `dynamic_axes`. The failure mode is either a build
error or — worse — silently wrong results for slices 1..N-1. The batch-consistency gate exists
to catch exactly this. If it trips, the fallback is re-running `--stage export` on the host
with a batch>1 dummy, which requires the wingdzero GroundingDINO checkout again; that is a
host-side change and would extend this work.

**Secondary: activation memory.** A max-batch-5 profile reserves more workspace on both GPUs.
Report the engine size and build-time workspace after the build; drop `max` to 3 if it is
material.

**Non-risk (measured):** TRT Myelin graph load/unload churn was suspected but totals 1.5 s over
588 s (0.25%) — not worth addressing.

## Out of scope

- Reclaiming `sam_compile` via fixed-batch padding (~18 ms/tick, ~+9% fps). Deferred until this
  lever lands, being several times smaller.
- Batching GDINO *across* workers — they are on different GPUs by design.
- Any change to the SAM path.
