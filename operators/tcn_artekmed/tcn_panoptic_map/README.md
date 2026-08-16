# TCN Panoptic Map

A fused CUDA kernel that replaces the per-detection cupy boolean-mask assignment in
`build_panoptic_map` (`applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py`)
with a single kernel launch. See
`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-10-panoptic-cuda-design.md`
for the motivation and the reformulation this kernel implements.

## Overview

`build_panoptic_map`'s cupy path paints one detection mask at a time:

```python
pmap[masks[j] > 0] = v          # once per detection
```

cupy expands that boolean-mask assignment into `nonzero()` + scatter, and `nonzero()` must
synchronise the device to size its output -- so ~30 detections per tick costs ~30 forced
`cudaStreamSynchronize` calls to paint a map whose actual GPU work is a few milliseconds.

This module does the same paint with one kernel launch instead: one thread per output pixel,
loop over the M detections, keep the one with the largest **priority** that covers the pixel and
has a non-zero packed value, write once. No atomics, no per-detection launches, no forced syncs.

## Reformulation

Painting each detection's mask in ascending-score order (so the most confident detection wins an
overlap) is equivalent to, per pixel, selecting the covering detection with the highest score.
Ties are handled by passing a **priority** per detection -- its index in the host's
`argsort(scores)` order -- rather than comparing scores on the device, so "largest priority wins"
reproduces "last painted in ascending order wins" exactly, including for tied scores.

## Build requirement (silent if unmet)

This operator has no model and no container requirement, but it **must be built**, and an unbuilt
module does not announce itself loudly.

`tcn_langsam` imports `holohub.tcn_panoptic_map` lazily and falls back to a cupy implementation with a
single warning line if the import fails, so a missing module costs performance rather than
correctness — which is exactly why it is easy to miss in a long log. Check for:

```
panoptic_backend=cuda requested but holohub.tcn_panoptic_map is unavailable; falling back to cupy
```

The module is only built when an **application lists it under `DEPENDS OPERATORS`**:

```cmake
add_holohub_application(<app> DEPENDS OPERATORS
        tcn_panoptic_map          # <- this is what sets OP_tcn_panoptic_map=ON
        ...)
```

Adding it to the application's `target_link_libraries` is **not** sufficient: that is a link-time
relationship resolved after configure, so the subdirectory is never added and the pybind module is
never produced. See the collection README.

To confirm it is enabled in an existing build tree:

```bash
grep OP_tcn_panoptic_map <build>/CMakeCache.txt      # expect :BOOL=ON
```

Reference usage: `applications/tcn_artekmed/tcn_shm_vlm_inference`, config key
`langsam_inference.panoptic_backend: cuda | cupy`.

## Usage

Not an `Operator`. Exposes one free function, taking raw CUDA device pointers as integers:

```python
from holohub.tcn_panoptic_map._tcn_panoptic_map import build_panoptic_map_cuda

build_panoptic_map_cuda(
    masks_ptr, values_ptr, priorities_ptr,  # M, H, W device buffers (see docstring)
    M, H, W,
    out_ptr, stream_ptr,
)
```

The host side (`values`/`priorities` computation, i.e. label -> class id, instance numbering,
and the priority permutation) lives in `plan_panoptic_paint`
(`applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py`), which is unit
tested against `build_panoptic_map` as an oracle in
`applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_panoptic_paint.py`. A thin
cupy-facing wrapper (pointer extraction, contiguity/dtype assertions, CUDA extension
availability fallback) lives in `langsam_common.py`, selected by the
`langsam_inference.panoptic_backend: "cupy" | "cuda"` config toggle (default `"cupy"`).

## M == 0

If there are no detections, `out` is zeroed via `cudaMemsetAsync` without launching the paint
kernel, so callers always get a valid all-zero map.

## Erosion — `erode_panoptic_map_cuda`

The second free function this module exposes, and the fix for **mask bleed at object edges**.

The packed map is consumed as a texture-lookup target: `tcn_label_sampler` samples it through the
texcoords `tcn_depthimage_backprojection` produces. So a mask that overshoots its object by a few
*colour* pixels labels **background depth pixels** as that object — and those points are real,
finite, and metres behind it. They are exactly what `tcn_instance_stats`' `trim_percentile` spends
its breakdown point on, so removing them here is strictly better than rejecting them downstream.

```python
from holohub.tcn_panoptic_map._tcn_panoptic_map import erode_panoptic_map_cuda

erode_panoptic_map_cuda(in_ptr, scratch_ptr, out_ptr, H, W, radius, stream_ptr)
```

A pixel keeps its packed label only if **every** pixel in the `(2r+1)²` window carries that same
label; otherwise it becomes 0. Because labels partition the image, that single test erodes every
instance at once — there is no per-label pass — and it also opens a seam between two instances that
touch, which is the same bleed problem seen from the other side.

**Separable.** The square window factors into a horizontal then a vertical pass: after the first
pass a pixel is already either its own label or 0, so requiring the vertical run to be all-equal
composes to exactly the square window. That equivalence is not assumed — it is pinned against a
brute-force square oracle (see Tests).

**Borders are clamped**, not treated as background, so an object running off the side of the frame
keeps its pixels there. At these camera angles most detections touch an edge, so the alternative
would shave nearly all of them.

`scratch` must not alias `in` or `out`; `out` **may** alias `in`. `out` is written in full,
including at `radius <= 0` (the identity), so it never needs pre-zeroing.

### Choosing a radius

The radius is in **colour** pixels (2048×1536), while the depth grid the labels land on is about 3×
coarser — so below ~3 it barely moves a depth pixel. An object thinner than `2r+1` disappears
entirely, which is what bounds it from above. Config key: `langsam_inference.mask_erosion_px`,
default `0` (off, and free — no allocation and no launch).

Measured at 1536×2048, one camera-frame:

| radius | time | labelled pixels kept |
|---|---|---|
| 2 | 78 µs | 97.6% |
| 4 | 106 µs | 95.3% |
| 6 | 128 µs | 93.0% |
| 9 | 165 µs | 89.5% |

### Fallback

`erode_panoptic_map_auto` (`tcn_langsam/models.py`) falls back to `erode_panoptic_map_cupy` —
`minimum_filter == maximum_filter` over the window, which is precisely "every pixel in the window
carries the same label" — if the compiled extension predates this function. Same result, more
launches. `mode="nearest"` there is exactly the kernel's clamped border: replicating the edge pixel
only adds values already inside the truncated window, so the min and max are unchanged by it.

## Tests

```bash
# host, numpy only -- the reference vs a brute-force square-window oracle
python3 ../tcn_langsam/tests/test_panoptic_erosion.py        # 9 cases

# in container -- the real kernel vs that reference, and vs the cupy fallback
PYTHONPATH=<build>/python/lib:/workspace/holohub python3 tests/test_panoptic_erode_cuda.py  # 7 cases
```

Between the two the chain is **brute-force square window == separable numpy reference == CUDA
kernel == cupy fallback**. The CUDA gate also covers a non-square map (a square one hides a row/
column swap), `out` aliasing `in`, an edge-touching object, and a full 1536×2048 map.

`test_panoptic_paint.py` (9 cases, host) still covers the paint reformulation.
