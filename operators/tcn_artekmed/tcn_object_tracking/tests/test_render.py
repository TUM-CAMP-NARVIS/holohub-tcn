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

# Import the pure function without pulling cupy/holoscan: exec just the parts we need.
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "render.py")).read()
_ns = {"np": np}
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
