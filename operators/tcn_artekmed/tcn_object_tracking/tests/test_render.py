"""Host tests for the box-to-line-segment conversion. Imports only numpy.

Run directly: `python3 operators/tcn_artekmed/tcn_object_tracking/tests/test_render.py`

`render.py` imports cupy and holoscan at module scope (it builds device tensors and Holoviz specs), so
the geometry helper is exercised through a copy of its constants rather than by importing the module.
Keeping the expectations independent is the point: they are derived from what a box's 12 edges ARE.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from association import UP_AXIS_Y, oriented_corners

# Import the pure function without pulling cupy/holoscan: exec just the parts we need. The names
# render.py takes from `association` are injected, because the exec'd slice skips its import block.
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "render.py")).read()
_ns = {"np": np, "math": __import__("math"),
       "UP_AXIS_Y": UP_AXIS_Y, "oriented_corners": oriented_corners}
_start = _src.index("_BOX_EDGES = (")
_end = _src.index("def box_input_specs(")
exec(_src[_start:_end], _ns)                      # noqa: S102 - deliberate, see the docstring
box_line_vertices = _ns["box_line_vertices"]
VERTICES_PER_BOX = _ns["VERTICES_PER_BOX"]
_BOX_EDGES = _ns["_BOX_EDGES"]


def test_no_boxes_gives_one_nan_segment():
    v = box_line_vertices([])
    assert v.shape == (1, 2, 3), v.shape
    assert np.all(np.isnan(v)), "an empty class must be unrenderable, not at the origin"


def test_shape_scales_with_box_count():
    for n in (1, 2, 7):
        boxes = [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))] * n
        assert box_line_vertices(boxes).shape == (1, n * VERTICES_PER_BOX, 3)


def test_twelve_distinct_edges_of_unit_cube():
    """A cube has 12 edges; every emitted pair must be an actual edge (differ on exactly one axis)."""
    v = box_line_vertices([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])[0]
    assert v.shape[0] == 24
    edges = set()
    for i in range(0, 24, 2):
        a, b = tuple(v[i]), tuple(v[i + 1])
        differing = sum(1 for d in range(3) if a[d] != b[d])
        assert differing == 1, f"pair {a}->{b} differs on {differing} axes; not a box edge"
        edges.add(frozenset((a, b)))
    assert len(edges) == 12, f"expected 12 distinct edges, got {len(edges)}"


def test_vertices_are_exactly_the_eight_corners():
    lo, hi = (-1.0, 2.0, -3.0), (4.0, 5.0, 6.0)
    v = box_line_vertices([(lo, hi)])[0]
    want = {tuple(c) for c in itertools.product(*zip(lo, hi))}
    assert {tuple(p) for p in v} == want, "emitted vertices are not the box corners"


def test_every_corner_has_degree_three():
    """Each cube corner joins exactly 3 edges -- catches a duplicated or missing edge."""
    v = box_line_vertices([((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))])[0]
    degree = {}
    for i in range(0, v.shape[0], 2):
        for p in (tuple(v[i]), tuple(v[i + 1])):
            degree[p] = degree.get(p, 0) + 1
    assert sorted(degree.values()) == [3] * 8, degree


def test_flat_box_still_produces_segments():
    """A degenerate box (zero extent on one axis) must not crash or drop vertices."""
    v = box_line_vertices([((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))])
    assert v.shape == (1, VERTICES_PER_BOX, 3)
    assert np.all(np.isfinite(v))


def test_output_is_float32_for_holoviz():
    v = box_line_vertices([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    assert v.dtype == np.float32, v.dtype


# ── oriented boxes ────────────────────────────────────────────────────────────────────────────────

oriented_line_vertices = _ns["oriented_line_vertices"]


def test_zero_yaw_oriented_box_matches_an_aabb():
    """With yaw 0 the oriented box must be exactly the axis-aligned one -- the degenerate case."""
    corners = oriented_corners((1.0, 2.0, 3.0), (0.4, 0.6, 1.8), 0.0, up_axis=1)
    xs = sorted({round(c[0], 6) for c in corners})
    ys = sorted({round(c[1], 6) for c in corners})
    zs = sorted({round(c[2], 6) for c in corners})
    assert xs == [0.8, 1.2], xs          # 0.4 along u=x
    assert ys == [1.1, 2.9], ys          # 1.8 vertical (up_axis=1)
    assert zs == [2.7, 3.3], zs          # 0.6 along v=z


def test_yaw_rotates_only_the_horizontal_plane():
    """A 90 degree yaw swaps the two horizontal extents and leaves the vertical one alone."""
    corners = oriented_corners((0.0, 0.0, 0.0), (2.0, 1.0, 1.6), np.pi / 2, up_axis=1)
    xs = max(c[0] for c in corners) - min(c[0] for c in corners)
    ys = max(c[1] for c in corners) - min(c[1] for c in corners)
    zs = max(c[2] for c in corners) - min(c[2] for c in corners)
    assert abs(xs - 1.0) < 1e-6, xs      # the 2.0 extent now lies along z
    assert abs(zs - 2.0) < 1e-6, zs
    assert abs(ys - 1.6) < 1e-6, ys      # vertical untouched


def test_oriented_box_has_twelve_edges_and_degree_three_corners():
    objs = [{"bbox_min": (0, 0, 0), "bbox_max": (1, 1, 1), "centroid": (0.5, 0.5, 0.5),
             "yaw": 0.3, "oriented_extent": (1.0, 0.5, 1.8), "oriented_center": (0.5, 0.9, 0.5)}]
    v = oriented_line_vertices(objs)[0]
    assert v.shape[0] == 24
    degree = {}
    for i in range(0, 24, 2):
        a, b = tuple(np.round(v[i], 6)), tuple(np.round(v[i + 1], 6))
        assert a != b, "degenerate edge"
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    assert sorted(degree.values()) == [3] * 8, degree


def test_falls_back_to_the_aabb_without_a_usable_orientation():
    """A near-circular footprint yields zero oriented extents; drawing that would be worse."""
    objs = [{"bbox_min": (0, 0, 0), "bbox_max": (1, 2, 3), "centroid": (0.5, 1, 1.5),
             "yaw": 0.0, "oriented_extent": (0.0, 0.0, 0.0), "oriented_center": (0, 0, 0)}]
    v = oriented_line_vertices(objs)[0]
    assert np.isfinite(v).all()
    assert abs(v[:, 1].max() - 2.0) < 1e-6, "did not fall back to the AABB extents"


def test_no_objects_still_gives_one_nan_segment():
    v = oriented_line_vertices([])
    assert v.shape == (1, 2, 3) and np.all(np.isnan(v))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, e)
        except Exception as e:
            bad += 1; print("ERROR", fn.__name__, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
