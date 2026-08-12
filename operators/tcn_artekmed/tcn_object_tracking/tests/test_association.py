"""Host tests for association.py (pure geometry and fusion). No holoscan, cupy or numpy.

Run directly: `python3 operators/tcn_artekmed/tcn_object_tracking/tests/test_association.py`

Expectations are hand-computed volumes and distances, never a restatement of the implementation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from association import (UP_AXIS_Y, UP_AXIS_Z, Observation, box_containment, box_extent,
                         filter_detections, filter_observations, suppress_aggregates,
                         box_intersection, box_iou, box_union, box_volume,
                         centroid_distance, footprint_iou, fuse_observations,
                         merge_observations, match_detections_to_tracks,
                         box_corners, oriented_corners, oriented_hull)


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


# ── small-object and aggregate rejection ──────────────────────────────────────────────────────────

def test_min_points_rejects_specks():
    keep = filter_observations([obs(1, 1, 0, 50, 0, 0, 0.9), obs(1, 2, 0, 5000, 3, 0, 0.9)],
                               min_points=500)
    assert [o.num_points for o in keep] == [5000]


def test_min_extent_rejects_a_flat_sliver():
    """A mask fringe produces a legitimately small, very flat box; no volume rule catches it."""
    sliver = Observation(1, 1, 0, 9000, (0, 0, 0.9), box(0, 0, 0.9, 0.27, 0.33, 0.03))
    person = obs(1, 2, 0, 9000, 3.0, 0, 0.9)
    keep = filter_observations([sliver, person], min_extent_m=0.10)
    assert len(keep) == 1 and keep[0].instance_id == 2, [box_extent(o.box) for o in keep]


def test_aggregate_containing_two_disjoint_objects_is_dropped():
    """A mask covering two people must not become one identity -- prefer the individuals."""
    a = obs(1, 1, 0, 20000, -0.6, 0, 0.9)                       # person A
    b = obs(1, 2, 0, 20000, +0.6, 0, 0.9)                       # person B, disjoint from A
    blob = Observation(1, 3, 1, 45000, (0.0, 0, 0.9), box(0.0, 0, 0.9, 2.0, 1.0, 2.0))
    keep = suppress_aggregates([a, b, blob])
    assert {o.instance_id for o in keep} == {1, 2}, [o.instance_id for o in keep]


def test_a_single_contained_partial_view_is_kept_and_merged():
    """The legitimate case the containment rule exists for: torso inside whole person -> one object."""
    person = obs(1, 1, 0, 40000, 0.0, 0, 0.9)
    torso = Observation(1, 1, 1, 12000, (0.0, 0, 1.2), box(0.0, 0, 1.2, 0.4, 0.4, 0.5))
    keep = suppress_aggregates([person, torso])
    assert len(keep) == 2, "a single-child container was mistaken for an aggregate"
    assert len(fuse_observations([person, torso])) == 1, "the partial view stopped merging"


def test_aggregate_of_similar_sized_boxes_is_not_dropped():
    """Two boxes of nearly equal size can each 'contain' the other; neither is an aggregate."""
    a = obs(1, 1, 0, 20000, 0.0, 0, 0.9)
    b = obs(1, 2, 1, 20000, 0.02, 0, 0.9)
    assert len(suppress_aggregates([a, b], min_volume_ratio=1.5)) == 2


def test_aggregate_containing_views_of_one_object_is_not_dropped():
    """Its children overlap each other, so they are views of one object, not distinct objects."""
    big = Observation(1, 1, 0, 40000, (0, 0, 0.9), box(0, 0, 0.9, 1.0, 1.0, 2.0))
    v1 = Observation(1, 2, 1, 10000, (0, 0, 0.9), box(0, 0, 0.9, 0.4, 0.4, 1.0))
    v2 = Observation(1, 3, 2, 10000, (0.05, 0, 0.95), box(0.05, 0, 0.95, 0.4, 0.4, 1.0))
    keep = suppress_aggregates([big, v1, v2])
    assert len(keep) == 3, "overlapping child views were treated as distinct objects"


def test_fusion_end_to_end_prefers_individuals_over_the_blob():
    a = obs(1, 1, 0, 20000, -0.6, 0, 0.9)
    b = obs(1, 2, 0, 20000, +0.6, 0, 0.9)
    blob = Observation(1, 3, 1, 45000, (0.0, 0, 0.9), box(0.0, 0, 0.9, 2.0, 1.0, 2.0))
    d = fuse_observations([a, b, blob], max_centroid_distance_m=1.0)
    assert len(d) == 2, f"expected two identities, got {len(d)}: {d}"
    assert all(x.num_points == 20000 for x in d), [x.num_points for x in d]


def test_min_cameras_drops_uncorroborated_detections():
    """An object only one camera ever saw is usually fringe; two cameras is corroboration."""
    seen_by_two = [obs(1, 1, 0, 20000, 0.0, 0, 0.9), obs(1, 1, 1, 18000, 0.03, 0, 0.9)]
    seen_by_one = [obs(1, 2, 2, 3000, 5.0, 0, 0.9)]
    d = fuse_observations(seen_by_two + seen_by_one, min_cameras=2)
    assert len(d) == 1, [x.cameras for x in d]
    assert d[0].cameras == (0, 1)
    # ...and with the default it survives, because a one-camera object can be real.
    assert len(fuse_observations(seen_by_two + seen_by_one)) == 2


def test_detection_gates_apply_to_the_FUSED_object():
    """Two partial views, each below the point threshold, together pass it."""
    partial = [obs(1, 1, 0, 600, 0.0, 0, 0.9), obs(1, 1, 1, 600, 0.02, 0, 0.9)]
    assert len(fuse_observations(partial, min_detection_points=1000)) == 1
    assert len(fuse_observations(partial, min_detection_points=2000)) == 0


def test_filter_detections_is_independent_of_fusion():
    from association import Detection
    d = [Detection(1, 5000, (0, 0, 0.9), box(0, 0, 0.9, 0.5, 0.5, 1.8), cameras=[0]),
         Detection(1, 5000, (3, 0, 0.9), box(3, 0, 0.9, 0.5, 0.5, 1.8), cameras=[0, 2])]
    assert len(filter_detections(d, min_cameras=2)) == 1
    assert len(filter_detections(d, min_points=6000)) == 0
    assert len(filter_detections(d, min_extent_m=1.0)) == 0     # 0.5 m on two axes


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


# ── oriented boxes ────────────────────────────────────────────────────────────────────────────────

def test_oriented_hull_of_an_axis_aligned_cloud_at_zero_yaw_is_its_aabb():
    b = box(1.0, 2.0, 3.0, 0.4, 1.8, 0.6)
    center, extent = oriented_hull(box_corners(b), 0.0, UP_AXIS_Y)
    assert max(abs(center[d] - (1.0, 2.0, 3.0)[d]) for d in range(3)) < 1e-9, center
    # extent is (along u, along v, vertical) = (x, z, y) for up=y
    assert max(abs(extent[d] - (0.4, 0.6, 1.8)[d]) for d in range(3)) < 1e-9, extent


def test_oriented_hull_recovers_the_box_it_was_generated_from():
    """corners -> hull is the identity at the generating yaw. Catches a sign error in either direction."""
    center0, extent0, yaw = (0.5, 1.0, -2.0), (1.2, 0.4, 1.7), 0.7
    center, extent = oriented_hull(oriented_corners(center0, extent0, yaw, UP_AXIS_Y), yaw, UP_AXIS_Y)
    assert max(abs(center[d] - center0[d]) for d in range(3)) < 1e-9, center
    assert max(abs(extent[d] - extent0[d]) for d in range(3)) < 1e-9, extent


def test_oriented_hull_of_a_rotated_box_at_zero_yaw_is_larger():
    """A 45-degree box measured on the world axes must be wider than itself -- the AABB inflation that
    motivates oriented boxes at all."""
    corners = oriented_corners((0, 0, 0), (2.0, 0.2, 1.0), 3.14159265358979 / 4, UP_AXIS_Y)
    _, extent = oriented_hull(corners, 0.0, UP_AXIS_Y)
    assert extent[0] > 1.5, extent      # sqrt(2)*(2+0.2)/2 = 1.556
    assert extent[2] == 1.0 or abs(extent[2] - 1.0) < 1e-9, extent   # vertical is unaffected by yaw


def test_merged_oriented_box_covers_every_view_not_just_the_largest():
    """The invariant that makes the reported oriented box trustworthy: it must not be smaller than the
    object. Taking the extent from the best-observed view alone reports a partial view as a tight fit.
    """
    a = Observation(1, 1, 0, 50000, (0.0, 0.9, 0.0), box(0.0, 0.9, 0.0, 0.5, 1.8, 0.5),
                    yaw=0.0, oriented_extent=(0.5, 0.5, 1.8), oriented_center=(0.0, 0.9, 0.0))
    b = Observation(1, 1, 1, 5000, (1.0, 0.9, 0.0), box(1.0, 0.9, 0.0, 0.5, 1.8, 0.5),
                    yaw=0.0, oriented_extent=(0.5, 0.5, 1.8), oriented_center=(1.0, 0.9, 0.0))
    d = merge_observations([a, b], UP_AXIS_Y)
    # the two views are 1 m apart on x, so the union spans 1.5 m there
    assert abs(d.oriented_extent[0] - 1.5) < 1e-9, d.oriented_extent
    assert abs(d.oriented_center[0] - 0.5) < 1e-9, d.oriented_center


def test_merged_oriented_volume_never_exceeds_the_aabb_volume():
    """An oriented box is the AABB refined, so it can only be tighter -- at equal yaw, identical."""
    a = Observation(1, 1, 0, 9000, (0.0, 0.9, 0.0), box(0.0, 0.9, 0.0, 1.4, 1.8, 1.4),
                    yaw=0.6, oriented_extent=(1.2, 0.4, 1.8), oriented_center=(0.0, 0.9, 0.0))
    b = Observation(1, 1, 1, 3000, (0.2, 0.9, 0.1), box(0.2, 0.9, 0.1, 1.4, 1.8, 1.4),
                    yaw=0.55, oriented_extent=(1.2, 0.4, 1.8), oriented_center=(0.2, 0.9, 0.1))
    d = merge_observations([a, b], UP_AXIS_Y)
    oriented_volume = d.oriented_extent[0] * d.oriented_extent[1] * d.oriented_extent[2]
    assert oriented_volume <= box_volume(d.box) + 1e-9, (oriented_volume, box_volume(d.box))


def test_a_view_whose_oriented_box_pokes_outside_its_aabb_cannot_inflate_the_result():
    """The reason the axis-aligned box is an explicit candidate rather than just `yaw = 0`.

    A circumscribing rectangle at any angle but the minimum-area one sticks out past the AABB at its
    corners, so the hull of those corners exceeds the axis-aligned union -- here a 0.6 m square
    footprint claimed at 45 degrees needs 0.85 m of room on each world axis. The search must then fall
    back to the AABB rather than reporting the inflated rotated box.
    """
    a = Observation(1, 1, 0, 50000, (0, 0.9, 0), box(0, 0.9, 0, 0.6, 1.8, 0.6),
                    yaw=0.785398, oriented_extent=(0.6, 0.6, 1.8), oriented_center=(0, 0.9, 0))
    d = merge_observations([a], UP_AXIS_Y)
    assert d.yaw == 0.0, f"yaw {d.yaw} was kept although it inflates the box"
    assert abs(d.oriented_extent[0] - 0.6) < 1e-6, d.oriented_extent
    assert abs(d.oriented_extent[1] - 0.6) < 1e-6, d.oriented_extent


def test_a_tight_single_view_oriented_box_is_kept():
    """The other side of that rule: a genuinely tighter rotated box must survive, or the min-area search
    would have silently degraded every object to its AABB."""
    a = Observation(1, 1, 0, 50000, (0, 0.9, 0), box(0, 0.9, 0, 1.1, 1.8, 1.1),
                    yaw=0.785398, oriented_extent=(1.5, 0.2, 1.8), oriented_center=(0, 0.9, 0))
    d = merge_observations([a], UP_AXIS_Y)
    assert abs(d.yaw - 0.785398) < 1e-9, d.yaw
    assert abs(d.oriented_extent[0] - 1.5) < 1e-6, d.oriented_extent


def test_merge_picks_the_yaw_that_gives_the_tightest_footprint():
    """Both views are 40-degree plates, so expressing the hull at 40 degrees is much tighter than at
    zero -- the candidate search has to find that rather than defaulting to no rotation."""
    a = Observation(1, 1, 0, 50000, (0, 0.9, 0), box(0, 0.9, 0, 1.2, 1.8, 1.2),
                    yaw=0.7, oriented_extent=(1.4, 0.2, 1.8), oriented_center=(0, 0.9, 0))
    b = Observation(1, 1, 1, 20000, (0, 0.9, 0), box(0, 0.9, 0, 1.2, 1.8, 1.2),
                    yaw=0.7, oriented_extent=(1.4, 0.2, 1.8), oriented_center=(0, 0.9, 0))
    d = merge_observations([a, b], UP_AXIS_Y)
    assert abs(d.yaw - 0.7) < 1e-9, d.yaw
    assert abs(d.oriented_extent[0] - 1.4) < 1e-9, d.oriented_extent
    assert abs(d.oriented_extent[1] - 0.2) < 1e-9, d.oriented_extent


def test_merged_oriented_footprint_never_exceeds_the_axis_aligned_one():
    """The invariant that makes an oriented box safe to report: it is the AABB REFINED, never inflated.

    Adopting the best-observed view's yaw outright breaks this -- two views 2 m apart hulled at 81
    degrees came out at 10.7 m2 against the axis-aligned 6.9 m2 on live data.
    """
    a = Observation(1, 1, 0, 50000, (0.0, 0.9, 0.0), box(0.0, 0.9, 0.0, 0.6, 1.8, 0.6),
                    yaw=1.41, oriented_extent=(0.6, 0.4, 1.8), oriented_center=(0.0, 0.9, 0.0))
    b = Observation(1, 1, 1, 30000, (2.0, 0.9, 0.3), box(2.0, 0.9, 0.3, 0.6, 1.8, 0.6),
                    yaw=-0.2, oriented_extent=(0.6, 0.4, 1.8), oriented_center=(2.0, 0.9, 0.3))
    d = merge_observations([a, b], UP_AXIS_Y)
    footprint = d.oriented_extent[0] * d.oriented_extent[1]
    aabb_footprint = box_extent(d.box)[0] * box_extent(d.box)[2]
    assert footprint <= aabb_footprint + 1e-9, (footprint, aabb_footprint)


def test_unoriented_view_contributes_its_aabb():
    """A near-circular footprint has no usable orientation, so its AABB is all it may claim -- and the
    merge must still cover it rather than dropping it."""
    a = Observation(1, 1, 0, 50000, (0, 0.9, 0), box(0, 0.9, 0, 0.5, 1.8, 0.5),
                    yaw=0.0, oriented_extent=(0.5, 0.5, 1.8), oriented_center=(0, 0.9, 0))
    b = Observation(1, 1, 1, 900, (2.0, 0.9, 0), box(2.0, 0.9, 0, 0.5, 1.8, 0.5))  # no oriented box
    d = merge_observations([a, b], UP_AXIS_Y)
    assert abs(d.oriented_extent[0] - 2.5) < 1e-9, d.oriented_extent


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
