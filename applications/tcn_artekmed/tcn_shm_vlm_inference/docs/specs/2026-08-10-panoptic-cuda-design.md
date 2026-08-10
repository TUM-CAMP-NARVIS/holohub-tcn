# Panoptic map as a C++/CUDA kernel (design)

Replace `build_panoptic_map`'s per-detection cupy boolean-mask assignment with one fused CUDA
kernel, exposed to Python as a pybind free function.

Implements step 2 of [`../dataflow-and-pipelining-roadmap.md`](../dataflow-and-pipelining-roadmap.md).

## Motivation — measured, and the cause is one line

From the 2026-08-10 batched-decode trace, dev1 `panoptic`: **28.52 ms wall over only 3.12 ms of
actual GPU work**, with 248 kernel launches and 44.9 `cudaStreamSynchronize` calls per tick.

The cause is a single statement in `langsam_helpers.build_panoptic_map`:

```python
pmap[masks[j] > 0] = v          # once per detection
```

cupy expands a boolean-mask assignment into *greater → nonzero → scatter*, and `nonzero` **must
synchronise** to size its output. The trace matches exactly:

| kernel | calls/tick | what it is |
|---|---|---|
| `cupy_scan_naive` | 89.6 | the `nonzero` scans (~3 per assignment) |
| `cupy_greater__uint8_uint_bool` | 29.9 | `masks[j] > 0` |
| `cupy_scatter_update_mask` | 29.9 | the masked store |
| `cudaStreamSynchronize` | 44.9 | forced by `nonzero` sizing |

~30 detections/tick → ~30 assignments → ~240 kernels and ~45 syncs, to paint a map whose real work
is 3.12 ms. This is exactly the hidden-sync trap already recorded in
[`../optimization-playbook.md`](../optimization-playbook.md), sitting in our own hot path.

There is also one `argsort(scores).tolist()` D2H per camera.

## Design

### 1. Reformulate: "highest score wins per pixel", not ordered painting

The current semantics are: number instances per class by **descending** score (instance 1 = most
confident), then paint in **ascending** score order so the most confident detection wins an overlap.

Painting in ascending order is equivalent to, for each pixel, selecting the covering detection with
the highest score. So the kernel needs no ordering and no atomics: one thread per pixel, loop over
the M detections, keep the best, write once.

Equivalence requires care with ties. Rather than compare scores (float ties are order-dependent),
the host passes a **priority** per detection: its index in the existing `argsort(scores)` order.
Those indices are unique, so "largest priority wins" reproduces "last painted in ascending order
wins" **exactly**, including for tied scores. This makes the kernel fully deterministic and lets the
gate be byte-identity rather than a tolerance.

### 2. Split of responsibility

Host (Python, unchanged logic) keeps everything involving labels and dicts:

- `argsort(scores)` — one small D2H per camera. Reduced from ~45 syncs to 1; eliminating it entirely
  would mean moving label→class-id lookup onto the device, which is a bigger change for a sync of a
  handful of floats. **Deliberately kept.**
- `class_id_for_label(labels[j], cmap)`, instance numbering, and the `(cid << 8) | inst` packing —
  produces a `values[j]` array (uint16, 0 for unknown labels, which the kernel then skips).
- `priorities[j]` per §1.

Device (the new kernel) does only the paint:

```
panoptic_paint(const uint8_t* masks,      // (M, H, W), row-major, contiguous
               const uint16_t* values,    // (M,) packed ids; 0 = skip this detection
               const int32_t* priorities, // (M,) unique; larger wins
               int M, int H, int W,
               uint16_t* out)             // (H, W), written in full (no pre-zeroing needed)
```

One thread per output pixel. Every pixel is written exactly once, so the kernel does not require a
zeroed output buffer — but it must write 0 where nothing covers, so the caller need not memset.

Cost estimate: M≈20 at 2048×1536 is ~63 MB of mask reads, memory-bound, ~0.15 ms. The existing GPU
work is 3.12 ms, so the kernel is not the point — removing ~240 launches and ~45 syncs is.

### 3. Packaging: a pybind free function, not an operator

`operators/tcn_artekmed/tcn_panoptic_map/`, following the tree's existing C++ operator layout
(`tcn_flatten_tensor` is the smallest reference) — `CMakeLists.txt` with
`project(... LANGUAGES CXX CUDA)`, a `.cu`, a `.hpp`, and `python/` using
`pybind11_add_holohub_module`.

**A free function, not an `Operator`.** This is the important structural choice. Panoptic runs
*inline* inside `LangSamBatchOp` in the monolithic path, which is the default and the only
configuration currently worth running. A C++ **operator** would only be usable in the pipelined
path, which measured −12%. A callable function is usable from **both** paths immediately, and the
same kernel can back a real C++ operator later if the pipelining retry ever happens.

Precedent for free functions in this tree: `discover_shm` in
`tcn_shm_subscriber/python/shm_subscriber_op.cpp` (`m.def("discover_shm", ...)`), imported by the
app from the inner module.

Note `pybind11_add_holohub_module`'s `CLASS_NAME` feeds a `configure_file` template that generates
`from ._<module> import <CLASS_NAME>` in the package `__init__.py` — it need not be a class, so set
it to the exported function name.

Signature exposed to Python takes raw device pointers as integers plus shapes and a stream handle,
which avoids any DLPack/`holoscan::Tensor` dependency in the binding:

```python
build_panoptic_map_cuda(masks_ptr, values_ptr, priorities_ptr, M, H, W, out_ptr, stream_ptr)
```

A thin Python wrapper in `langsam_helpers` (or `langsam_common`) extracts `.data.ptr` from the cupy
arrays and passes `cp.cuda.get_current_stream().ptr`, keeping the call site readable and the pointer
handling in one place.

### 4. Opt-in

`langsam_inference.panoptic_backend: "cupy" | "cuda"`, **default `"cupy"`**, following the
`gdino_backend` / `sam_backend` / `sam_batched_decode` / `pipelined` precedent. The existing cupy
implementation stays as the reference and the A/B oracle. If the extension is not built, fall back to
cupy with a single WARNING rather than failing.

## Correctness gate: byte-identical

Per §1 the kernel is deterministic and reproduces the existing tie behaviour exactly, so:

- `compare_mask_dumps.py A B` in **exact** mode, `panoptic_backend: cupy` vs `cuda`, must show zero
  differing pixels. Not an IoU gate — any difference is a bug.

This is a stronger gate than 4a's and is available precisely because this change moves no
floating-point arithmetic.

## Testing

| test | where | gate |
|---|---|---|
| host-side `values`/`priorities` computation | host, numpy | new; extract the label→class-id, instance-numbering and priority logic into a pure helper and test it against the current `build_panoptic_map` internals as oracle |
| numpy model of the kernel | host, numpy | implement "highest priority wins per pixel" in numpy and assert it equals the existing `build_panoptic_map` output for randomised masks/labels/scores, **including tied scores and unknown labels** |
| existing host suites | host | 8/8, 11/11, 7/7, 19/19 unchanged |
| `panoptic_backend: cupy` unchanged | container | proves the toggle is inert |
| cupy vs cuda | container | `compare_mask_dumps.py` exact — **the gate** |
| `panoptic` NVTX wall, launches, syncs | container, nsys | vs 28.52 ms, 248 launches, 44.9 syncs on dev1 |
| period and fps | container, nsys | vs 171.0 ms / 5.847 fps |

The numpy model of the kernel is the highest-value host test: it validates the *reformulation*
(§1), which is where a correctness error would actually come from. The CUDA code itself is a direct
transcription of it.

## Expected effect

panoptic 28.52 → ~5 ms, i.e. **~19 ms** saved. Note ~7.50 ms of the current sync time is absorbed
SAM wait (deferred work from `predict_batch_gpu`, which never synchronises at `timing=False`), and
that will **relocate** rather than vanish — per the 4a lesson, predict from non-API + launch time,
not from total stage wall time.

If it lands, period 171.0 → ~155 ms. Combined with A1 (which lowers the GPU floor from 117.5 to
~95 ms) the two are complementary: A1 removes GPU work, this removes overhead.

## Risks

- **Non-contiguous mask stacks.** The kernel assumes a contiguous `(M,H,W)` uint8 buffer. `SamOp`
  produces a *list* of per-camera `(K,H,W)` cupy arrays, so the wrapper must `cp.ascontiguousarray`
  / stack and must not silently pass a strided view. Assert contiguity in the wrapper.
- **Build integration.** A new `operators/tcn_artekmed/` subdirectory must be added to that tree's
  `CMakeLists.txt`; a missed entry means the module silently is not built, which the §4 fallback
  would then mask as a warning. Check the fallback logs loudly enough to notice.
- **`uint16` output dtype** must match exactly — the dumps are compared bitwise and the packed
  format is `(class << 8) | instance`, so a dtype change would break both the gate and downstream
  colourisation.
- **M = 0** (no detections on a camera): must return an all-zero map without launching.

## Out of scope

- Running panoptic as a true C++ `Operator` (only useful with pipelining, which measured −12%).
- Moving label→class-id resolution onto the device to eliminate the last `argsort` D2H.
- The `hw` quirk (the operator uses the *last* camera's shape for every camera's map) — recorded in
  [`../deferred-findings.md`](../deferred-findings.md) §4 and deliberately preserved here.
