"""Host tests for association.py (pure geometry and fusion). No holoscan, cupy or numpy.

Run directly: `python3 operators/tcn_artekmed/tcn_object_tracking/tests/test_association.py`

Expectations are hand-computed volumes and distances, never a restatement of the implementation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from association import (UP_AXIS_Y, UP_AXIS_Z, Observation, box_containment,
                         box_intersection, box_iou, box_union, box_volume,
                         centroid_distance, footprint_iou, fuse_observations,
                         merge_observations, match_detections_to_tracks)


def box(cx, cy, cz, sx, sy, sz):
    """Centred box with the given full extents -- how a person-sized object is described."""
    return ((cx - sx / 2, cy - sy / 2, cz - sz / 2), (cx + sx / 2, cy + sy / 2, cz + sz / 2))


def obs(class_id, inst, cam, n, cx, cy, cz, sx=0.5, sy=0.5, sz=1.8):
    return Observation(class_id, inst, cam, n, (cx, cy, cz), box(cx, cy, cz, sx, sy, sz))


# ── geometry ──────────────────────────────────────────────────────────────────────────────────────

def test_volume_and_disjoint_boxes():
    assert box_volume(((0, 0, 0), (2, 3, 4))) == 24.0
    assert box_volume(((0, 0, 0), (0, 3, 4))) == 0.0             # degenerate
    assert box_intersection(((0, 0, 0), (1, 1, 1)), ((2, 2, 2), (3, 3, 3))) is None
    assert box_iou(((0, 0, 0), (1, 1, 1)), ((2, 2, 2), (3, 3, 3))) == 0.0


def test_iou_hand_computed():
    # Two unit cubes offset by 0.5 on x: intersection 0.5, union 2 - 0.5 = 1.5.
    a, b = ((0, 0, 0), (1, 1, 1)), ((0.5, 0, 0), (1.5, 1, 1))
    assert abs(box_iou(a, b) - (0.5 / 1.5)) < 1e-12


def test_touching_faces_do_not_overlap():
    # Shared face has zero volume; treating it as overlap would fuse objects standing side by side.
    assert box_intersection(((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1))) is None


def test_containment_catches_the_partial_view_case():
    """A small box inside a large one: low IoU, high containment. This is the reason it exists."""
    full = box(0, 0, 0.9, 0.5, 0.5, 1.8)          # whole person
    torso = box(0, 0, 1.2, 0.4, 0.4, 0.5)         # one camera sees only the torso
    assert box_iou(full, torso) < 0.25
    assert box_containment(full, torso) > 0.9


def test_union_covers_both():
    u = box_union(((0, 0, 0), (1, 1, 1)), ((-1, 0.5, 0), (0.5, 2, 3)))
    assert u == ((-1, 0, 0), (1, 2, 3))


def test_centroid_distance():
    assert abs(centroid_distance((0, 0, 0), (3, 4, 0)) - 5.0) < 1e-12


# ── merging ───────────────────────────────────────────────────────────────────────────────────────

def test_merge_weights_centroid_by_point_count():
    """A camera seeing a sliver must not pull the centre as hard as one seeing the whole object."""
    a = obs(1, 1, 0, n=9000, cx=0.0, cy=0.0, cz=0.9)
    b = obs(1, 1, 1, n=1000, cx=1.0, cy=0.0, cz=0.9)
    d = merge_observations([a, b])
    assert abs(d.centroid[0] - 0.1) < 1e-9, d.centroid      # 9000/10000 weight on x=0
    assert d.num_points == 10000
    assert d.cameras == (0, 1)


def test_merge_box_is_the_union():
    a = Observation(1, 1, 0, 100, (0, 0, 0), ((0, 0, 0), (1, 1, 1)))
    b = Observation(1, 1, 1, 100, (0, 0, 0), ((-1, -1, -1), (0.5, 0.5, 0.5)))
    assert merge_observations([a, b]).box == ((-1, -1, -1), (1, 1, 1))


# ── fusion ────────────────────────────────────────────────────────────────────────────────────────

def test_two_cameras_seeing_one_person_fuse_to_one():
    o = [obs(1, 1, 0, 50000, 2.0, 3.0, 0.9), obs(1, 1, 1, 40000, 2.05, 2.98, 0.9)]
    d = fuse_observations(o)
    assert len(d) == 1, d
    assert d[0].num_points == 90000
    assert d[0].cameras == (0, 1)


def test_two_separate_people_stay_separate():
    o = [obs(1, 1, 0, 50000, 0.0, 0.0, 0.9), obs(1, 2, 0, 50000, 3.0, 0.0, 0.9)]
    assert len(fuse_observations(o)) == 2


def test_fragments_within_one_camera_merge():
    """One physical object arriving as two instance ids from the SAME camera must become one object."""
    lower = Observation(1, 1, 0, 20000, (0, 0, 0.4), box(0, 0, 0.4, 0.5, 0.5, 0.8))
    upper = Observation(1, 2, 0, 20000, (0, 0, 1.2), box(0, 0, 1.2, 0.5, 0.5, 0.9))
    d = fuse_observations([lower, upper], up_axis=UP_AXIS_Z)   # fixture is built z-up
    assert len(d) == 1, f"fragments stayed split: {d}"
    assert d[0].num_points == 40000
    assert d[0].observations == 2


def test_vertically_split_fragments_merge_on_footprint():
    """A person split by an occluding table: same footprint, stacked, boxes barely touching."""
    legs = Observation(1, 1, 0, 15000, (0, 0, 0.4), box(0, 0, 0.4, 0.5, 0.5, 0.8))    # z 0.0..0.8
    torso = Observation(1, 2, 0, 25000, (0, 0, 1.3), box(0, 0, 1.3, 0.5, 0.5, 0.7))   # z 0.95..1.65
    d = fuse_observations([legs, torso], up_axis=UP_AXIS_Z)   # this fixture is built z-up
    assert len(d) == 1, f"stacked fragments on one footprint stayed split: {d}"
    assert d[0].num_points == 40000

    # The SAME geometry with the wrong up axis must not merge -- which is why up_axis is configured
    # rather than assumed: a wrong value degrades fusion silently.
    wrong = fuse_observations([legs, torso], up_axis=UP_AXIS_Y)
    assert len(wrong) == 2, "the footprint rule fired with the wrong vertical axis"


def test_footprint_rule_does_not_merge_people_side_by_side():
    """Both span the full height, so neither is above the other -- the vertical gap rule must not fire."""
    a = Observation(1, 1, 0, 20000, (0.0, 0, 0.9), box(0.0, 0, 0.9, 0.5, 0.5, 1.8))
    b = Observation(1, 2, 0, 20000, (0.55, 0, 0.9), box(0.55, 0, 0.9, 0.5, 0.5, 1.8))
    assert len(fuse_observations([a, b])) == 2, "adjacent people were fused"


def test_footprint_rule_does_not_merge_across_a_large_vertical_gap():
    low = Observation(1, 1, 0, 20000, (0, 0, 0.2), box(0, 0, 0.2, 0.5, 0.5, 0.4))     # z 0.0..0.4
    high = Observation(1, 2, 0, 20000, (0, 0, 2.5), box(0, 0, 2.5, 0.5, 0.5, 0.4))    # z 2.3..2.7
    assert len(fuse_observations([low, high], max_vertical_gap_m=0.5,
                                 up_axis=UP_AXIS_Z)) == 2


def test_footprint_iou_uses_the_configured_plane():
    """A box tall in z: its x-y footprint is small, its x-z footprint large. The axis decides."""
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 4.0))
    b = ((0.0, 0.0, 3.0), (1.0, 1.0, 7.0))          # stacked along z, same x-y footprint
    assert footprint_iou(a, b, UP_AXIS_Z) == 1.0    # z is height: identical ground footprints
    assert footprint_iou(a, b, UP_AXIS_Y) < 1.0     # y is height: the x-z footprints differ


def test_different_classes_never_fuse():
    o = [obs(1, 1, 0, 10000, 0.0, 0.0, 0.9), obs(2, 1, 0, 10000, 0.0, 0.0, 0.9)]
    d = fuse_observations(o)
    assert len(d) == 2 and {x.class_id for x in d} == {1, 2}


def test_distance_gate_stops_single_linkage_chaining():
    """Three overlapping boxes in a row must not chain into one object across a large distance."""
    # Extents chosen so NEIGHBOURS overlap above the IoU gate (0.7 m of 1.5 m) while the ends do
    # not: with sx=1.0 the neighbour IoU is only 0.111 and there would be no chain to break.
    o = [obs(1, 1, 0, 10000, 0.0, 0, 0.9, sx=1.5),
         obs(1, 2, 0, 10000, 0.8, 0, 0.9, sx=1.5),
         obs(1, 3, 0, 10000, 1.6, 0, 0.9, sx=1.5)]
    linked = fuse_observations(o, max_centroid_distance_m=1.0)
    assert len(linked) == 1                      # neighbours are within 1 m: one chain
    split = fuse_observations(o, max_centroid_distance_m=0.5)
    assert len(split) == 3, split                # gate below the spacing: no chaining


def test_background_and_empty_rows_are_ignored():
    o = [Observation(0, 0, 0, 5000, (0, 0, 0), ((0, 0, 0), (1, 1, 1))),   # background
         Observation(1, 1, 0, 0, (0, 0, 0), ((0, 0, 0), (0, 0, 0))),      # placeholder row
         obs(1, 1, 0, 10000, 5.0, 5.0, 0.9)]
    d = fuse_observations(o)
    assert len(d) == 1 and d[0].class_id == 1


def test_fusion_output_order_is_deterministic():
    a = obs(1, 1, 0, 10000, 0.0, 0.0, 0.9)
    b = obs(1, 2, 0, 30000, 5.0, 0.0, 0.9)
    first = [(d.class_id, d.num_points) for d in fuse_observations([a, b])]
    second = [(d.class_id, d.num_points) for d in fuse_observations([b, a])]
    assert first == second, "output depends on input order"
    assert first[0][1] == 30000, "largest object should come first"


# ── matching ──────────────────────────────────────────────────────────────────────────────────────

class FakeTrack:
    def __init__(self, class_id, centroid, box_):
        self.class_id, self.centroid, self.box = class_id, centroid, box_


def test_matching_pairs_the_nearest_and_leaves_the_rest():
    dets = fuse_observations([obs(1, 1, 0, 10000, 0.0, 0, 0.9), obs(1, 2, 0, 10000, 4.0, 0, 0.9)])
    tracks = [FakeTrack(1, (0.05, 0, 0.9), box(0.05, 0, 0.9, 0.5, 0.5, 1.8))]
    pairs, ud, ut = match_detections_to_tracks(dets, tracks)
    assert len(pairs) == 1 and pairs[0][1] == 0
    assert len(ud) == 1 and ut == []
    matched_det = dets[pairs[0][0]]
    assert abs(matched_det.centroid[0]) < 0.5, "matched the far detection instead of the near one"


def test_matching_respects_the_class():
    dets = fuse_observations([obs(2, 1, 0, 10000, 0.0, 0, 0.9)])
    tracks = [FakeTrack(1, (0.0, 0, 0.9), box(0, 0, 0.9, 0.5, 0.5, 1.8))]
    pairs, ud, ut = match_detections_to_tracks(dets, tracks)
    assert pairs == [] and ud == [0] and ut == [0]


def test_matching_gate_rejects_a_distant_pairing():
    dets = fuse_observations([obs(1, 1, 0, 10000, 0.0, 0, 0.9)])
    tracks = [FakeTrack(1, (9.0, 0, 0.9), box(9, 0, 0.9, 0.5, 0.5, 1.8))]
    pairs, ud, ut = match_detections_to_tracks(dets, tracks, max_centroid_distance_m=1.0)
    assert pairs == [], "a 9 m jump was accepted as the same object"


def test_two_objects_do_not_swap_when_close():
    """Symmetric crossing is the classic id-swap failure; each must take its own nearest track."""
    dets = fuse_observations([obs(1, 1, 0, 10000, 0.0, 0, 0.9), obs(1, 2, 0, 10000, 0.9, 0, 0.9)])
    tracks = [FakeTrack(1, (0.02, 0, 0.9), box(0.02, 0, 0.9, 0.5, 0.5, 1.8)),
              FakeTrack(1, (0.92, 0, 0.9), box(0.92, 0, 0.9, 0.5, 0.5, 1.8))]
    pairs, _, _ = match_detections_to_tracks(dets, tracks)
    assert len(pairs) == 2
    for di, ti in pairs:
        assert centroid_distance(dets[di].centroid, tracks[ti].centroid) < 0.3, \
            f"detection {dets[di].centroid} matched the far track {tracks[ti].centroid}"


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
