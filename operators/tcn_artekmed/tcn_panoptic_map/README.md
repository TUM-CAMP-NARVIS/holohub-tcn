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
