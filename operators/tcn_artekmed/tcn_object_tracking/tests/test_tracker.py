"""Host tests for tracker.py (identity over time). No holoscan, cupy or numpy.

Run directly: `python3 operators/tcn_artekmed/tcn_object_tracking/tests/test_tracker.py`

These are the tests that matter most: identity bugs -- swapped ids, churn, resurrection -- are
invisible in a rendered scene and obvious here.
"""
import os
import sys

# Import through the package: tracker.py imports association relatively, which is correct for
# package use and means it cannot be loaded as a loose module. The package __init__ is lazy, so this
# pulls in no holoscan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcn_object_tracking.association import Detection, fuse_observations, Observation
from tcn_object_tracking.tracker import ObjectTracker


def box(cx, cy, cz, sx=0.5, sy=0.5, sz=1.8):
    return ((cx - sx / 2, cy - sy / 2, cz - sz / 2), (cx + sx / 2, cy + sy / 2, cz + sz / 2))


def det(class_id, cx, cy=0.0, cz=0.9, n=10000):
    return Detection(class_id, n, (cx, cy, cz), box(cx, cy, cz), cameras=[0])


def ids(tracks):
    return [t.track_id for t in tracks]


# ── lifecycle ─────────────────────────────────────────────────────────────────────────────────────

def test_a_new_object_is_not_reported_until_min_hits():
    t = ObjectTracker(min_hits=3, max_age=8)
    assert ids(t.update([det(1, 0.0)])) == []          # tentative
    assert ids(t.update([det(1, 0.01)])) == []
    assert ids(t.update([det(1, 0.02)])) == [1]        # confirmed on the third hit


def test_a_one_frame_false_positive_never_gets_an_id():
    t = ObjectTracker(min_hits=3, max_age=2)
    t.update([det(1, 0.0)])
    for _ in range(4):
        assert ids(t.update([])) == []
    assert t.tracks == [], "the tentative track was never retired"


def test_identity_is_stable_while_observed():
    t = ObjectTracker(min_hits=2, max_age=8)
    seen = []
    for i in range(12):
        seen.append(ids(t.update([det(1, 0.05 * i)])))   # walking slowly along x
    assert seen[1:] == [[1]] * 11, seen


def test_occlusion_shorter_than_max_age_keeps_the_id():
    t = ObjectTracker(min_hits=2, max_age=5)
    t.update([det(1, 0.0)]); t.update([det(1, 0.0)])
    assert ids(t.confirmed()) == [1]
    for _ in range(4):                                   # 4 missed frames, under max_age
        t.update([])
    assert ids(t.update([det(1, 0.05)])) == [1], "id changed across a short occlusion"


def test_occlusion_longer_than_max_age_gives_a_new_id():
    t = ObjectTracker(min_hits=1, max_age=3)
    assert ids(t.update([det(1, 0.0)])) == [1]
    for _ in range(5):                                   # beyond max_age: retired
        t.update([])
    assert ids(t.update([det(1, 0.0)])) == [2], "a retired track was resurrected"


def test_ids_are_never_reused():
    t = ObjectTracker(min_hits=1, max_age=1)
    first = ids(t.update([det(1, 0.0)]))
    for _ in range(4):
        t.update([])
    second = ids(t.update([det(1, 0.0)]))
    third_after = ids(t.update([det(1, 0.0), det(1, 5.0)]))
    assert first == [1] and second == [2]
    assert set(first) & set(third_after) == set() or third_after == [2, 3]
    assert max(third_after) == 3, third_after


def test_two_objects_keep_distinct_ids_while_approaching():
    t = ObjectTracker(min_hits=1, max_age=8)
    a, b = 0.0, 3.0
    assert ids(t.update([det(1, a), det(1, b)])) == [1, 2]
    history = []
    for _ in range(8):                                   # converge to 0.8 m apart, never closer
        a, b = a + 0.1, b - 0.15
        if b - a < 0.8:
            b = a + 0.8
        history.append(ids(t.update([det(1, a), det(1, b)])))
    assert all(h == [1, 2] for h in history), history
    by_id = {tr.track_id: tr.centroid[0] for tr in t.confirmed()}
    assert by_id[1] < by_id[2], f"ids swapped: {by_id}"


def test_a_second_object_appearing_does_not_disturb_the_first():
    t = ObjectTracker(min_hits=1, max_age=8)
    assert ids(t.update([det(1, 0.0)])) == [1]
    assert ids(t.update([det(1, 0.0), det(1, 4.0)])) == [1, 2]
    assert ids(t.update([det(1, 0.0)])) == [1, 2]         # 2 is missing but not yet retired
    first = [tr for tr in t.confirmed() if tr.track_id == 1][0]
    assert first.misses == 0 and first.hits == 3


def test_classes_are_tracked_independently():
    t = ObjectTracker(min_hits=1, max_age=8)
    out = t.update([det(1, 0.0), det(2, 0.0)])            # same place, different classes
    assert len(out) == 2 and {tr.class_id for tr in out} == {1, 2}
    assert len({tr.track_id for tr in out}) == 2


# ── prompt / class-definition changes ─────────────────────────────────────────────────────────────

def test_class_signature_change_resets_tracks():
    """Class ids are prompt positions, so a vocabulary change renumbers them; identity cannot survive."""
    t = ObjectTracker(min_hits=1, max_age=8)
    t.note_class_signature(["person", "bed"])
    assert ids(t.update([det(1, 0.0)])) == [1]
    changed = t.note_class_signature(["bed", "person"])   # same terms, different ids
    assert changed is True
    assert t.tracks == [], "tracks survived a class renumbering"
    assert ids(t.update([det(1, 0.0)])) == [2], "id was reused after a reset"


def test_unchanged_class_signature_does_not_reset():
    t = ObjectTracker(min_hits=1, max_age=8)
    t.note_class_signature(["person", "bed"])
    t.update([det(1, 0.0)])
    assert t.note_class_signature(["Person", " bed "]) is False   # normalised comparison
    assert ids(t.confirmed()) == [1]


# ── reported values ───────────────────────────────────────────────────────────────────────────────

def test_box_smoothing_damps_jitter_but_follows_motion():
    jittery = ObjectTracker(min_hits=1, max_age=8, box_smoothing=0.0)
    smooth = ObjectTracker(min_hits=1, max_age=8, box_smoothing=0.8)
    for cx in (0.0, 0.2, 0.0, 0.2, 0.0):                 # oscillating by 20 cm
        jittery.update([det(1, cx)])
        smooth.update([det(1, cx)])
    j = jittery.confirmed()[0].box[0][0]
    s = smooth.confirmed()[0].box[0][0]
    raw_min = box(0.0, 0.0, 0.9)[0][0]
    assert abs(j - raw_min) < 1e-9, "smoothing 0 should follow the raw box exactly"
    assert abs(s - raw_min) > 1e-3, "smoothing 0.8 should lag the raw box"


def test_as_dict_carries_what_a_consumer_needs():
    t = ObjectTracker(min_hits=1, max_age=8)
    t.update([det(1, 1.0, n=4242)])
    d = t.confirmed()[0].as_dict({1: "person"})
    for key in ("track_id", "class_id", "class_name", "centroid", "bbox_min", "bbox_max",
                "extent", "num_points", "cameras", "hits", "misses", "age"):
        assert key in d, f"missing {key}"
    assert d["class_name"] == "person" and d["num_points"] == 4242
    assert all(e > 0 for e in d["extent"])


def test_confirmed_output_is_ordered_by_id():
    t = ObjectTracker(min_hits=1, max_age=8)
    t.update([det(1, 0.0), det(1, 3.0), det(1, 6.0)])
    assert ids(t.confirmed()) == sorted(ids(t.confirmed()))


# ── end to end through fusion ─────────────────────────────────────────────────────────────────────

def test_two_cameras_one_person_yields_one_track():
    """The whole point: two cameras must not produce two identities for one person."""
    t = ObjectTracker(min_hits=1, max_age=8)
    for i in range(4):
        x = 0.05 * i
        o = [Observation(1, 1, 0, 50000, (x, 0, 0.9), box(x, 0, 0.9)),
             Observation(1, 1, 1, 40000, (x + 0.03, 0.01, 0.9), box(x + 0.03, 0.01, 0.9))]
        out = t.update(fuse_observations(o))
    assert ids(out) == [1], f"expected one identity, got {ids(out)}"
    assert list(out[0].cameras) == [0, 1]
    assert out[0].num_points == 90000


def test_instance_id_churn_does_not_churn_the_track_id():
    """SAM renumbers instances by confidence every frame; the track id must not follow."""
    t = ObjectTracker(min_hits=1, max_age=8)
    for frame, inst in enumerate([1, 2, 1, 3, 2]):        # arbitrary per-frame instance ids
        o = [Observation(1, inst, 0, 50000, (0.02 * frame, 0, 0.9), box(0.02 * frame, 0, 0.9))]
        out = t.update(fuse_observations(o))
    assert ids(out) == [1], f"track id followed the instance id: {ids(out)}"
    assert out[0].hits == 5


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
