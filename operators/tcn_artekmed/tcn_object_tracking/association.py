"""Pure geometry and matching for object association. No holoscan, no cupy, no numpy required.

Kept dependency-free so it is host-testable in milliseconds, which matters because this is where the
algorithmic risk lives: a wrong overlap test or a wrong merge order produces plausible-looking
objects that are quietly wrong.

Boxes are axis-aligned in the world frame, represented as `(min_xyz, max_xyz)` with each a 3-tuple.
World units are metres, so every threshold in this module is a real distance.
"""
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Box = Tuple[Vec3, Vec3]


def box_volume(box: Box) -> float:
    lo, hi = box
    return max(0.0, hi[0] - lo[0]) * max(0.0, hi[1] - lo[1]) * max(0.0, hi[2] - lo[2])


def box_intersection(a: Box, b: Box) -> Optional[Box]:
    """Overlap of two boxes, or None when they do not overlap on every axis."""
    lo = tuple(max(a[0][d], b[0][d]) for d in range(3))
    hi = tuple(min(a[1][d], b[1][d]) for d in range(3))
    if any(hi[d] <= lo[d] for d in range(3)):
        return None
    return (lo, hi)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union, 0.0 when disjoint.

    Note IoU alone is a poor gate for objects of very different size: a small box fully inside a large
    one scores only `small/large`. `box_containment` covers that case, and the matcher uses both.
    """
    inter = box_intersection(a, b)
    if inter is None:
        return 0.0
    vi = box_volume(inter)
    union = box_volume(a) + box_volume(b) - vi
    return vi / union if union > 0.0 else 0.0


def box_containment(a: Box, b: Box) -> float:
    """Fraction of the SMALLER box's volume that lies inside the other.

    This is what recognises "one camera sees a person's torso, another sees the whole person" as the
    same object: the torso box is small, mostly inside the full box, and its IoU is low.
    """
    inter = box_intersection(a, b)
    if inter is None:
        return 0.0
    smaller = min(box_volume(a), box_volume(b))
    return box_volume(inter) / smaller if smaller > 0.0 else 0.0


def box_union(a: Box, b: Box) -> Box:
    lo = tuple(min(a[0][d], b[0][d]) for d in range(3))
    hi = tuple(max(a[1][d], b[1][d]) for d in range(3))
    return (lo, hi)


#: Index of the vertical world axis. This is a property of the CALIBRATION, not a convention you can
#: assume: `tcn_depthimage_backprojection` applies each camera's `depth_extrinsics`, so "up" is
#: whatever the export's `camera_pose` made it. For the artekmed exports it is **y** -- verifiable by
#: printing object centres, where the vertical axis spans a narrow plausible-height band (~0.8 m)
#: while the two horizontal axes span the room (4-7 m). Getting this wrong does not fail loudly: the
#: footprint rule simply stops merging vertically split objects.
UP_AXIS_Y = 1
UP_AXIS_Z = 2


def _plane_axes(up_axis: int):
    return tuple(d for d in range(3) if d != up_axis)


def footprint_iou(a: Box, b: Box, up_axis: int = UP_AXIS_Y) -> float:
    """IoU of the two boxes' ground-plane footprints, ignoring the vertical axis."""
    p, q = _plane_axes(up_axis)
    lo = (max(a[0][p], b[0][p]), max(a[0][q], b[0][q]))
    hi = (min(a[1][p], b[1][p]), min(a[1][q], b[1][q]))
    if hi[0] <= lo[0] or hi[1] <= lo[1]:
        return 0.0
    inter = (hi[0] - lo[0]) * (hi[1] - lo[1])
    area_a = max(0.0, a[1][p] - a[0][p]) * max(0.0, a[1][q] - a[0][q])
    area_b = max(0.0, b[1][p] - b[0][p]) * max(0.0, b[1][q] - b[0][q])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def vertical_gap(a: Box, b: Box, up_axis: int = UP_AXIS_Y) -> float:
    """Gap between the two boxes along the vertical axis; 0.0 when they overlap on it."""
    u = up_axis
    if a[0][u] > b[1][u]:
        return a[0][u] - b[1][u]
    if b[0][u] > a[1][u]:
        return b[0][u] - a[1][u]
    return 0.0


def centroid_distance(a: Vec3, b: Vec3) -> float:
    return sum((a[d] - b[d]) ** 2 for d in range(3)) ** 0.5


class Observation:
    """One instance as seen by one camera in one frame."""

    __slots__ = ("class_id", "instance_id", "camera_index", "num_points", "centroid", "box", "sigma")

    def __init__(self, class_id: int, instance_id: int, camera_index: int, num_points: int,
                 centroid: Vec3, box: Box, sigma: Vec3 = (0.0, 0.0, 0.0)):
        self.class_id = int(class_id)
        self.instance_id = int(instance_id)
        self.camera_index = int(camera_index)
        self.num_points = int(num_points)
        self.centroid = tuple(float(v) for v in centroid)
        self.box = (tuple(float(v) for v in box[0]), tuple(float(v) for v in box[1]))
        self.sigma = tuple(float(v) for v in sigma)

    def __repr__(self):
        return (f"Observation(class={self.class_id} inst={self.instance_id} cam={self.camera_index} "
                f"n={self.num_points} c={tuple(round(v, 3) for v in self.centroid)})")


class Detection:
    """One physical object at one timestamp, after merging every camera's view of it."""

    __slots__ = ("class_id", "num_points", "centroid", "box", "cameras", "observations")

    def __init__(self, class_id: int, num_points: int, centroid: Vec3, box: Box,
                 cameras: Sequence[int], observations: int = 1):
        self.class_id = int(class_id)
        self.num_points = int(num_points)
        self.centroid = tuple(float(v) for v in centroid)
        self.box = box
        self.cameras = tuple(sorted(set(int(c) for c in cameras)))
        self.observations = int(observations)

    @property
    def extent(self) -> Vec3:
        return tuple(self.box[1][d] - self.box[0][d] for d in range(3))

    def __repr__(self):
        return (f"Detection(class={self.class_id} n={self.num_points} "
                f"c={tuple(round(v, 3) for v in self.centroid)} cams={self.cameras})")


def merge_observations(group: Sequence[Observation]) -> Detection:
    """Combine several views of one object: box union, point-weighted centroid.

    The centroid is weighted by point count rather than averaged over cameras, because a camera
    seeing a sliver of the object should not pull the centre as hard as one seeing all of it.
    """
    total = sum(o.num_points for o in group)
    if total <= 0:
        centroid = tuple(sum(o.centroid[d] for o in group) / len(group) for d in range(3))
    else:
        centroid = tuple(sum(o.centroid[d] * o.num_points for o in group) / total for d in range(3))
    box = group[0].box
    for o in group[1:]:
        box = box_union(box, o.box)
    return Detection(class_id=group[0].class_id, num_points=total, centroid=centroid, box=box,
                     cameras=[o.camera_index for o in group], observations=len(group))


def fuse_observations(observations: Sequence[Observation],
                      iou_threshold: float = 0.15,
                      containment_threshold: float = 0.6,
                      max_centroid_distance_m: float = 1.0,
                      footprint_iou_threshold: float = 0.4,
                      max_vertical_gap_m: float = 0.5,
                      up_axis: int = UP_AXIS_Y) -> List[Detection]:
    """Group observations of the same physical object into detections, per class.

    Deliberately merges observations from the SAME camera as well as across cameras: one physical
    object routinely fragments into several masks (an occluding arm, a partial detection), and each
    fragment arrives as its own instance id.

    Grouping is single-linkage: A merges with B if they overlap enough, and transitively. That is the
    right shape here because a chain of partial views of one person should collapse to one object, and
    the alternative (all-pairs agreement) would split an object seen from opposite sides.

    Two observations are linked when they share a class, their centroids are within
    `max_centroid_distance_m`, and any of:

    - **IoU >= iou_threshold** -- the ordinary case, two cameras seeing the same body
    - **containment >= containment_threshold** -- one view is a subset of the other, e.g. one camera
      sees only a torso while another sees the whole person
    - **footprint IoU >= footprint_iou_threshold and vertical gap <= max_vertical_gap_m** -- the
      occlusion-fragmentation case: a person split by an intervening table arrives as two masks whose
      3D boxes barely touch or do not touch at all, but which stand on the same footprint, one above
      the other. Neither of the volume rules can merge those, because they genuinely do not overlap.

    The centroid gate is what stops single-linkage from chaining across a room: two people standing
    close both overlap a third box between them, and without a distance limit the chain would fuse
    them into one object. The vertical rule is safe against two people side by side, because both span
    the full height and so neither is *above* the other.
    """
    by_class: Dict[int, List[Observation]] = {}
    for o in observations:
        if o.class_id == 0 or o.num_points <= 0:
            continue                      # background, or an empty placeholder row
        by_class.setdefault(o.class_id, []).append(o)

    detections: List[Detection] = []
    for class_id in sorted(by_class):
        items = by_class[class_id]
        # Union-find over the observations of this class.
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if centroid_distance(a.centroid, b.centroid) > max_centroid_distance_m:
                    continue
                linked = (box_iou(a.box, b.box) >= iou_threshold
                          or box_containment(a.box, b.box) >= containment_threshold
                          or (footprint_iou(a.box, b.box, up_axis) >= footprint_iou_threshold
                              and vertical_gap(a.box, b.box, up_axis) <= max_vertical_gap_m))
                if linked:
                    union(i, j)

        groups: Dict[int, List[Observation]] = {}
        for i, o in enumerate(items):
            groups.setdefault(find(i), []).append(o)
        # Deterministic order: largest object first, then by centroid, so the output does not depend
        # on dict iteration order or on which camera happened to report first.
        merged = [merge_observations(g) for g in groups.values()]
        merged.sort(key=lambda d: (-d.num_points, d.centroid))
        detections.extend(merged)
    return detections


def match_detections_to_tracks(detections: Sequence["object"],
                               tracks: Sequence["object"],
                               iou_threshold: float = 0.1,
                               max_centroid_distance_m: float = 1.0
                               ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy matching of detections to tracks within a class.

    Returns `(pairs, unmatched_detections, unmatched_tracks)` as index lists. Pairs are chosen in
    descending order of a score that prefers overlap and falls back to proximity, so the most
    confident association is committed first -- with a hard gate, so a bad pairing is left unmatched
    rather than forced.

    Greedy rather than optimal (Hungarian): with a handful of objects per class the difference is
    negligible, and greedy is deterministic and trivial to reason about when a gate rejects
    everything. `tracks` need only expose `class_id`, `centroid` and `box`.
    """
    candidates = []
    for di, d in enumerate(detections):
        for ti, t in enumerate(tracks):
            if d.class_id != t.class_id:
                continue
            dist = centroid_distance(d.centroid, t.centroid)
            if dist > max_centroid_distance_m:
                continue
            iou = box_iou(d.box, t.box)
            contain = box_containment(d.box, t.box)
            if iou < iou_threshold and contain < 0.5:
                continue
            # Overlap dominates; proximity breaks ties and rescues a just-moved object whose boxes
            # barely overlap between frames.
            score = max(iou, contain) + 1.0 / (1.0 + dist)
            candidates.append((score, di, ti))

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_d, used_t, pairs = set(), set(), []
    for _, di, ti in candidates:
        if di in used_d or ti in used_t:
            continue
        used_d.add(di)
        used_t.add(ti)
        pairs.append((di, ti))
    pairs.sort()
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    return pairs, unmatched_d, unmatched_t
