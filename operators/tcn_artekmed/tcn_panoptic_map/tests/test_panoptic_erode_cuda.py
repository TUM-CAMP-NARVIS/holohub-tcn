# SPDX-License-Identifier: Apache-2.0
"""In-container gate for `launch_panoptic_erode` -- the CUDA half of the mask-erosion path.

Runs the REAL kernel and asserts it is bit-identical to `erode_panoptic_np`
(operators/tcn_artekmed/tcn_langsam/helpers.py), which is itself pinned against a brute-force
square-window oracle by tcn_langsam/tests/test_panoptic_erosion.py. Between the two files the
chain is: brute-force square window == separable numpy reference == CUDA kernel.

Also checks the kernel against `erode_panoptic_map_cupy`, the fallback used when the compiled
extension is missing -- a fallback that silently disagrees with the kernel is worse than no
fallback, because the disagreement only shows up on machines where the build is broken.

Needs the toolchain, so it runs inside the container:

    docker exec <container> bash -lc '
      cd /workspace/holohub/operators/tcn_artekmed/tcn_panoptic_map/tests
      PYTHONPATH=/workspace/holohub/build/tcn_shm_vlm_inference/python/lib:/workspace/holohub \
        python3 test_panoptic_erode_cuda.py'
"""

import os
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, "/workspace/holohub")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tcn_langsam"))

from helpers import erode_panoptic_np                                    # noqa: E402
from holohub.tcn_panoptic_map._tcn_panoptic_map import (                 # noqa: E402
    erode_panoptic_map_cuda,
)


def _erode_kernel(pmap_np, radius):
    """Run the real kernel on a numpy map, return a numpy map."""
    src = cp.asarray(pmap_np, dtype=cp.uint16)
    scratch = cp.empty_like(src)
    out = cp.empty_like(src)
    h, w = src.shape
    erode_panoptic_map_cuda(
        int(src.data.ptr), int(scratch.data.ptr), int(out.data.ptr),
        int(h), int(w), int(radius), int(cp.cuda.get_current_stream().ptr),
    )
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out)


def _erode_cupy_fallback(pmap_np, radius):
    from cupyx.scipy.ndimage import maximum_filter, minimum_filter
    src = cp.asarray(pmap_np, dtype=cp.uint16)
    size = 2 * int(radius) + 1
    lo = minimum_filter(src, size=size, mode="nearest")
    hi = maximum_filter(src, size=size, mode="nearest")
    return cp.asnumpy(cp.where(lo == hi, src, cp.uint16(0)).astype(cp.uint16))


def _random_pmap(rng, h, w, n_labels=5, blob_radius=11):
    """Overlapping discs, so the map has real regions with real boundaries. Per-pixel noise would
    erode to nothing and would prove almost nothing."""
    pmap = np.zeros((h, w), dtype=np.uint16)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n_labels):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        label = np.uint16(((i % 5 + 1) << 8) | (i % 7))
        pmap[(yy - cy) ** 2 + (xx - cx) ** 2 <= blob_radius ** 2] = label
    return pmap


def case_kernel_matches_the_numpy_reference():
    rng = np.random.default_rng(101)
    for trial in range(10):
        h, w = int(rng.integers(24, 97)), int(rng.integers(24, 97))
        r = int(rng.integers(1, 7))
        pmap = _random_pmap(rng, h, w, n_labels=int(rng.integers(2, 7)))
        got, want = _erode_kernel(pmap, r), erode_panoptic_np(pmap, r)
        assert np.array_equal(got, want), (
            f"trial {trial}: h={h} w={w} r={r}, "
            f"{int(np.sum(got != want))} differing pixels of {pmap.size}")


def case_kernel_matches_the_cupy_fallback():
    """The fallback must agree with the kernel, or a machine with a stale build gets different
    masks from a machine with a fresh one."""
    rng = np.random.default_rng(103)
    for trial in range(6):
        h, w = int(rng.integers(24, 65)), int(rng.integers(24, 65))
        r = int(rng.integers(1, 6))
        pmap = _random_pmap(rng, h, w)
        got, want = _erode_kernel(pmap, r), _erode_cupy_fallback(pmap, r)
        assert np.array_equal(got, want), (
            f"trial {trial}: h={h} w={w} r={r}, "
            f"{int(np.sum(got != want))} differing pixels of {pmap.size}")


def case_non_square_map_is_not_transposed():
    """A square test map hides a row/column swap in the flat indexing. This one cannot."""
    rng = np.random.default_rng(107)
    pmap = _random_pmap(rng, 31, 97, n_labels=4, blob_radius=9)
    got, want = _erode_kernel(pmap, 3), erode_panoptic_np(pmap, 3)
    assert np.array_equal(got, want), f"{int(np.sum(got != want))} differing pixels"


def case_radius_zero_is_the_identity():
    rng = np.random.default_rng(109)
    pmap = _random_pmap(rng, 40, 40)
    assert np.array_equal(_erode_kernel(pmap, 0), pmap), "radius 0 changed the map"


def case_out_may_alias_in():
    """The header promises in-place is safe (after the horizontal pass `in` is never read again).
    A caller relying on that must not silently get a corrupt map."""
    rng = np.random.default_rng(113)
    pmap = _random_pmap(rng, 48, 48)
    want = erode_panoptic_np(pmap, 3)

    buf = cp.asarray(pmap, dtype=cp.uint16)
    scratch = cp.empty_like(buf)
    h, w = buf.shape
    erode_panoptic_map_cuda(
        int(buf.data.ptr), int(scratch.data.ptr), int(buf.data.ptr),   # out aliases in
        int(h), int(w), 3, int(cp.cuda.get_current_stream().ptr),
    )
    cp.cuda.get_current_stream().synchronize()
    got = cp.asnumpy(buf)
    assert np.array_equal(got, want), f"{int(np.sum(got != want))} differing pixels in place"


def case_edge_touching_object_survives_the_border():
    """Borders are clamped, not background. Getting this wrong shaves every detection that runs
    off the side of the frame -- which, at these camera angles, is most of them."""
    pmap = np.zeros((32, 32), dtype=np.uint16)
    pmap[8:24, 0:16] = np.uint16((2 << 8) | 1)
    r = 3
    got = _erode_kernel(pmap, r)
    assert np.array_equal(got, erode_panoptic_np(pmap, r))
    assert np.all(got[8 + r:24 - r, 0] != 0), "the left-edge column was eroded away"


def case_full_resolution_map():
    """The real thing: one camera's map at colour resolution, at a plausible radius. Guards the
    int32 flat index against overflow at 3.1 Mpx and confirms the cost is trivial."""
    rng = np.random.default_rng(127)
    pmap = _random_pmap(rng, 1536, 2048, n_labels=6, blob_radius=180)
    got, want = _erode_kernel(pmap, 5), erode_panoptic_np(pmap, 5)
    assert np.array_equal(got, want), (
        f"{int(np.sum(got != want))} differing pixels of {pmap.size}")
    assert np.count_nonzero(got) < np.count_nonzero(pmap), "nothing was eroded at all"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("case_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} cases passed")
    raise SystemExit(1 if failed else 0)
