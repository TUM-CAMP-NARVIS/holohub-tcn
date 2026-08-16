# SPDX-License-Identifier: Apache-2.0
"""Host-runnable tests for the panoptic-map erosion (`erode_panoptic_np`, mirrored by
`launch_panoptic_erode` in operators/tcn_artekmed/tcn_panoptic_map).

Why this exists: the packed panoptic map is sampled through the depth image's texcoords
(`tcn_label_sampler`), so a mask that overshoots the object by a few colour pixels labels
BACKGROUND depth pixels as the object. Those points are real, finite and metres away, and they
are what `tcn_instance_stats`' percentile trim spends its breakdown point on. Eroding the map
before the lookup removes the bleed at its source instead.

`_oracle_square` -- the (2r+1)^2 window applied at once, written the slow obvious way -- is the
ORACLE. `erode_panoptic_np` is separable (two 1-D passes) because the kernel is, so the highest
value test here is that the separable form matches the square one: that is the property the
kernel's structure depends on, not just a transcription detail.

Compatible with pytest; also runnable directly, matching the other tests in this directory.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import erode_panoptic_np


def _oracle_square(pmap, radius):
    """Brute-force (2r+1)^2 erosion, clamped at the borders. Deliberately naive."""
    if radius <= 0:
        return np.asarray(pmap).copy()
    h, w = pmap.shape
    out = np.zeros_like(pmap)
    for y in range(h):
        for x in range(w):
            v = pmap[y, x]
            y0, y1 = max(0, y - radius), min(h, y + radius + 1)
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            out[y, x] = v if np.all(pmap[y0:y1, x0:x1] == v) else 0
    return out


def _random_pmap(rng, h, w, n_labels=4, blob_radius=6):
    """A label map made of overlapping discs, so it has real regions with real boundaries --
    per-pixel noise would be eroded to nothing and would test almost nothing."""
    pmap = np.zeros((h, w), dtype=np.uint16)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n_labels):
        cy, cy_ = rng.integers(0, h), rng.integers(0, w)
        label = np.uint16(((i % 5 + 1) << 8) | (i % 7))
        disc = (yy - cy) ** 2 + (xx - cy_) ** 2 <= blob_radius ** 2
        pmap[disc] = label
    return pmap


# --- the property the kernel's separable structure rests on ------------------------------------

def test_separable_matches_square_window():
    rng = np.random.default_rng(7)
    for trial in range(12):
        h, w = int(rng.integers(12, 34)), int(rng.integers(12, 34))
        r = int(rng.integers(1, 5))
        pmap = _random_pmap(rng, h, w, n_labels=int(rng.integers(2, 6)))
        got = erode_panoptic_np(pmap, r)
        want = _oracle_square(pmap, r)
        assert np.array_equal(got, want), (
            f"trial {trial}: h={h} w={w} r={r}, "
            f"{int(np.sum(got != want))} differing pixels out of {pmap.size}")


def test_separable_matches_square_on_touching_regions():
    """Two labels sharing a straight seam. The seam must open by r on BOTH sides -- the case
    where 'erode each label independently' and 'erode the partition' could differ."""
    pmap = np.zeros((20, 20), dtype=np.uint16)
    pmap[:, :10] = np.uint16((1 << 8) | 0)
    pmap[:, 10:] = np.uint16((2 << 8) | 1)
    for r in (1, 2, 3):
        got = erode_panoptic_np(pmap, r)
        assert np.array_equal(got, _oracle_square(pmap, r)), f"r={r}"
        # The seam is at x=10; columns 10-r .. 9+r inclusive lose their label on both sides.
        assert np.all(got[:, 10 - r:10 + r] == 0), f"r={r}: seam did not open"
        assert np.all(got[:, :10 - r] == pmap[:, :10 - r]), f"r={r}: left side over-eroded"


# --- behaviour --------------------------------------------------------------------------------

def test_radius_zero_is_identity():
    rng = np.random.default_rng(11)
    pmap = _random_pmap(rng, 24, 24)
    for r in (0, -1):
        assert np.array_equal(erode_panoptic_np(pmap, r), pmap), f"r={r}"


def test_object_thinner_than_the_window_vanishes():
    """A stripe of width 2r or less has no pixel with a full window inside it, so it must
    disappear entirely -- this is what bounds a usable radius from above."""
    for r in (1, 2, 3):
        for width in range(1, 2 * r + 1):
            pmap = np.zeros((16, 16), dtype=np.uint16)
            pmap[:, 5:5 + width] = np.uint16((3 << 8) | 2)
            got = erode_panoptic_np(pmap, r)
            assert not np.any(got), (
                f"r={r}, stripe width {width} (<= 2r) survived erosion: "
                f"{int(np.count_nonzero(got))} pixels left")
        # ...and one pixel wider it must survive, or the test above would pass for a kernel
        # that erodes everything.
        pmap = np.zeros((16, 16), dtype=np.uint16)
        pmap[:, 5:5 + 2 * r + 1] = np.uint16((3 << 8) | 2)
        assert np.any(erode_panoptic_np(pmap, r)), f"r={r}: a (2r+1)-wide stripe was eaten"


def test_edge_touching_object_is_not_eaten_from_the_border():
    """Borders are clamped, not treated as background: an object running off the left edge keeps
    its pixels there. Treating outside as background would shave every frame-edge detection."""
    pmap = np.zeros((16, 16), dtype=np.uint16)
    pmap[4:12, 0:8] = np.uint16((1 << 8) | 3)
    r = 2
    got = erode_panoptic_np(pmap, r)
    assert np.array_equal(got, _oracle_square(pmap, r))
    # Column 0 is interior in x (clamped) but rows 4..5 / 10..11 still erode in y.
    assert np.all(got[4 + r:12 - r, 0] != 0), "left-edge column was eroded away"
    assert np.all(got[4, :] == 0), "top row of the object should still erode"


def test_background_stays_background():
    rng = np.random.default_rng(13)
    pmap = _random_pmap(rng, 24, 24)
    got = erode_panoptic_np(pmap, 2)
    assert np.all(got[pmap == 0] == 0), "erosion invented a label on a background pixel"


def test_erosion_only_removes_never_relabels():
    """Every surviving pixel must carry the label it already had. A pixel taking a NEIGHBOUR's
    label would corrupt instance identity, which is what the whole tracking chain keys on."""
    rng = np.random.default_rng(17)
    for r in (1, 2, 3):
        pmap = _random_pmap(rng, 28, 28, n_labels=5)
        got = erode_panoptic_np(pmap, r)
        surviving = got != 0
        assert np.array_equal(got[surviving], pmap[surviving]), f"r={r}: a label changed"
        assert np.count_nonzero(got) <= np.count_nonzero(pmap), f"r={r}: erosion added pixels"


def test_monotonic_in_radius():
    """A larger radius can only remove more. Guards against an off-by-one that makes r+1
    erode less than r somewhere."""
    rng = np.random.default_rng(19)
    pmap = _random_pmap(rng, 30, 30, n_labels=4, blob_radius=9)
    prev = erode_panoptic_np(pmap, 1)
    for r in (2, 3, 4):
        cur = erode_panoptic_np(pmap, r)
        kept = cur != 0
        assert np.array_equal(cur[kept], prev[kept]), f"r={r}: kept a pixel r-1 had dropped"
        assert np.count_nonzero(cur) <= np.count_nonzero(prev), f"r={r}: eroded less than r-1"
        prev = cur


def test_dtype_and_shape_preserved():
    rng = np.random.default_rng(23)
    pmap = _random_pmap(rng, 17, 29)
    got = erode_panoptic_np(pmap, 2)
    assert got.dtype == np.uint16, got.dtype
    assert got.shape == pmap.shape, got.shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
