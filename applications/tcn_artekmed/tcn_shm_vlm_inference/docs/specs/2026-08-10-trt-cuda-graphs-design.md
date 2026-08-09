# CUDA graph capture for the TRT engines (design)

Replay each TensorRT engine execution from a captured CUDA graph instead of re-issuing every
kernel launch each tick.

## Motivation — measured

From the 2026-08-10 batched-decode traces (`sam_batched_decode: true`, monolithic), launches
attributed to the NVTX stage they were issued inside, and to the kernel each one launched:

| stage | launcher | kernel family | calls/tick |
|---|---|---|---|
| gdino | `cuLaunchKernel` | TRT/myelin | **1027.9** |
| gdino | `cuLaunchKernelEx` | TRT/cublas | **815.4** |
| gdino | `cuLaunchKernel` | cupy/cub | 101.8 |
| gdino | `cudaLaunchKernel` | torch | 26.0 |
| sam | `cudaLaunchKernel` | torch | **672.0** |
| sam | `cuLaunchKernel` | TRT/myelin | 170.0 |
| sam | `cuLaunchKernelEx` + other | TRT/cublas | 144.0 |

**~94% of `gdino`'s launches are the TRT engine itself** — one `execute_async_v3` fans out into
~1843 individual kernel launches. That costs ~8.8 ms/tick of launch time plus a share of the
11.0 ms/tick of non-API (Python/GIL) time, on a stage whose actual work is a single engine call.

A captured graph replays all ~1843 kernels from **one** launch.

The `sam` split is the useful boundary: its ~310 TRT launches are the encoder and are capturable;
its 672 torch launches are the mask decoder, whose batch dimension M is the per-tick box count and
therefore **not** a graph candidate without padding M to a fixed size. Those 672 belong to step 4b,
not here.

## Why this is cheaper than the alternatives

It needs no ONNX export, no model surgery, and no inlining of upstream internals. The engines are
already **fixed-batch** (from the earlier per-worker engine work), which is precisely the
precondition graph capture requires. And unlike step 4a, replaying a graph runs the identical
kernels on identical data, so this one **is** required to be byte-identical.

## Design

### 1. Persistent IO buffers — the actual prerequisite

Graph capture bakes device pointers into the graph. Today `GDinoTrtDetector.detect_batch`
re-allocates both sides every tick:

```python
img = ((torch.cat(chw, dim=0) - self._mean) / self._std).contiguous()   # new tensor each tick
...
outs[nm] = torch.empty(tuple(self.ctx.get_tensor_shape(nm)), ...)       # new tensors each tick
self.ctx.set_tensor_address(nm, outs[nm].data_ptr())
```

PyTorch's caching allocator often hands back the same addresses, but "often" is not a contract, and
a graph replayed against a stale pointer reads freed memory — silently, with plausible-looking
output. So this change is not optional and is most of the work:

- Allocate the input buffer `(engine_batch, 3, H, W)` and every output tensor **once**, at
  construction, and keep them for the object's lifetime.
- Preprocess **into** the input buffer (`buf[:n].copy_(...)` or `out=`) rather than producing a new
  tensor.
- Call `set_input_shape` / `set_tensor_address` **once** at construction, not per tick. Shapes are
  constant because the engine is fixed-batch.
- The pad rows (`buf[n:]`) are zeroed once at construction and never written again. Today they are
  re-zeroed every tick via `torch.cat`; with a persistent buffer they simply stay zero. Note `n` is
  constant in practice (a worker always submits all its cameras), so `pad` is normally 0 — but the
  buffer must still be correct if a camera drops out.

### 2. Capture off the default stream, replay on it

`cudaStreamBeginCapture` **fails on the legacy default stream**. But capture and replay streams are
independent: `cudaGraphLaunch(graph, stream)` accepts any stream, including the default one.

So:

- **At warmup**, on a private `torch.cuda.Stream`: run the engine once eagerly (TRT requires a warm
  execution before capture), then `BeginCapture` → `execute_async_v3(capture_stream)` →
  `EndCapture` → instantiate.
- **Per tick**, launch the instantiated graph on `torch.cuda.current_stream()` — the default stream.

This is deliberate: everything downstream of these engines depends on default-stream ordering, and
`langsam_pipelined.py`'s `SamOp`/`PanopticOp` guards actively enforce it. Capturing on a side stream
while replaying on the default stream leaves that invariant **completely untouched**. Do not be
tempted to also move the replay onto a side stream — that is the separate, larger change described
in the roadmap's step 3 retry.

### 3. Fallback, not a hard requirement

Not every engine is capturable — a layer that allocates, or does host-side work, during enqueue will
make `EndCapture` fail. Treat capture as an optimisation that may decline:

- Try to capture at construction. On any failure, log **once** at WARNING with the exception, set
  `self.graph = None`, and use the existing `execute_async_v3` path forever after.
- Never let a capture failure break the app. The eager path stays as the reference implementation,
  exactly as the decode loop did in 4a.

### 4. Opt-in

`langsam_inference.trt_cuda_graphs: true|false`, **default `false`**, following the
`gdino_backend` / `sam_backend` / `sam_batched_decode` / `pipelined` precedent. Applies to both the
GDINO detector and the SAM TRT encoder.

### 5. Scope

- **In:** `GDinoTrtDetector` (the ~1843-launch win) and `SamTrtEncoder` (~310 launches).
- **Out:** the SAM mask decoder's 672 torch launches — variable M. Padding M to a fixed size plus
  `torch.cuda.CUDAGraph` is conceivable but is a different design; step 4b (exporting the decoder to
  TRT with a padded box dimension) subsumes it.
- **Out:** the ~102 cupy/cub launches in `gdino` — that is `gdino_postprocess_batch`, whose work is
  data-dependent.

## Correctness gate: byte-identical

Graph replay issues the same kernels, in the same order, on the same data. Unlike 4a there is no
re-association and no shape change, so outputs must match **exactly**.

- Same scene, `trt_cuda_graphs: false` then `true` → identical masks, identical box counts,
  identical class ids.
- Any difference at all is a bug — most likely a stale pointer from §1 — and must be investigated,
  not tolerated. This is a stronger gate than 4a's IoU ≥ 0.999 and the difference is the point.

## Expected effect

`gdino` on dev1 spends ~8.8 ms/tick in launch calls plus 11.0 ms non-API. Replacing ~1843 launches
with one should remove most of the former and a meaningful part of the latter; the SAM encoder adds
~1–2 ms. Estimate **10–16 ms/tick**, i.e. period 171 → ~157 ms (≈6.4 fps, +9%).

Stated as an estimate, and note the 4a lesson: predict from **non-API + launch time**, not from
total stage wall time, because `gdino`'s 49.3 ms of `cudaStreamSynchronize` is GPU wait that will
not shrink and may simply relocate.

The GPU floor on dev1 is 117.5 ms/frame, so after this the remaining addressable overhead is
`sam`'s decoder (4b) and `panoptic` (step 2, the C++/CUDA operator).

## Testing

| test | where | gate |
|---|---|---|
| existing host suites | host, numpy | 8/8, 11/11, 7/7, 19/19 unchanged |
| module parses, flag threaded through both engine classes | host, `py_compile` + grep | the only host check possible |
| `trt_cuda_graphs: false` unchanged | container | proves the toggle is inert |
| false vs true, same scene | container | **byte-identical** masks, boxes and class ids |
| capture-failure path | container | force a failure (e.g. temporarily raise in the capture block) and confirm it logs once and still runs |
| launches/tick inside `gdino` | container, nsys | ~1843 TRT launches → ~1; this is the proof the graph is live |
| period and fps | container, nsys | vs 171.0 ms / 5.847 fps |

The engine classes import torch and TensorRT and **cannot be host-tested**; correctness comes from
the container A/B, as with 4a.

## Risks

- **Stale pointers (§1).** The failure mode is silent wrong output, not a crash. This is the main
  risk and the reason persistent buffers are a prerequisite rather than a follow-up.
- **Capture unsupported** for one of the engines — handled by §3's fallback.
- **Re-capture on change.** Verified: `_text_for_batch` memoises its result in `self._text_batched`
  per batch size, so the text tensors are allocated once and keep their addresses for the object's
  lifetime — a captured graph stays valid across ticks. But `set_prompts` rebuilds `_class_masks`
  and can change what the engine is asked to express, so **re-capture (or invalidate) the graph in
  `set_prompts`** rather than assuming the baked pointers still describe the right work. Cheap
  insurance: prompts change rarely, and the alternative failure is silent.
- **Memory.** Persistent IO buffers are held for the process lifetime instead of being recycled by
  the caching allocator. Small next to the engines, but no longer transient.
