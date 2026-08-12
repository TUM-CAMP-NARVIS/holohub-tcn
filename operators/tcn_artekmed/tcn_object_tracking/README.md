# tcn_object_tracking

Turns per-camera per-instance statistics into **one entry per physical object** with a bounding box, a
centre of mass, and an id that stays the same while the object is observed — plus a console dump and a
box overlay for the point cloud.

## Why instance ids cannot be used as identities

`tcn_langsam.helpers.plan_panoptic_paint` numbers instances **per class, by descending detection
score, per camera, per frame**. So "instance 1" means "the most confident detection of this class, in
this camera, this frame". It changes when confidence ranking changes, and camera A's instance 1 is
unrelated to camera B's. That is the problem this package solves.

Two properties of the upstream pipeline make it tractable: points are already in **world space**
(backprojection applies `depth_extrinsics`), and a frame group is **timestamp-consistent**
(`tcn_stream_synchronizer`), so cross-camera association at one instant is well defined.

## The chain

```
per camera:  tcn_instance_stats ─┐
                                 ├─► InstanceFusionOp ─► ObjectTrackerOp ─┬─► ObjectConsoleSinkOp
             (ANY_SIZE receiver) ─┘      (stateless)        (stateful)     └─► ObjectBoxRendererOp ─► HolovizOp
```

| operator | role |
|---|---|
| `InstanceFusionOp` | merges every camera's view of one object into one detection, per timestamp |
| `ObjectTrackerOp` | assigns persistent ids across frames |
| `ObjectConsoleSinkOp` | prints objects, and a grep-able verdict at shutdown |
| `ObjectBoxRendererOp` | boxes as `LINES_3D` for the point-cloud view |

## Layout

```
tcn_object_tracking/
  association.py   pure: 3D IoU, containment, footprint IoU, box union, oriented boxes, weighted
                   centroid, matching
  tracker.py       pure: Track, lifecycle, id allocation, class-change reset
  ops.py           InstanceFusionOp, ObjectTrackerOp, ObjectConsoleSinkOp
  render.py        ObjectBoxRendererOp, box_input_specs, box_line_vertices
  tests/           75 host checks, no holoscan/cupy/numpy needed for the algorithm ones
```

`association.py` and `tracker.py` import **nothing** — that is deliberate. Identity bugs (swapped ids,
churn, resurrection) are invisible in a rendered scene and obvious in a unit test, so the algorithms
live where they can be tested in milliseconds.

## Fusion: what counts as "the same object"

Within one class, two observations are linked when their centroids are within
`max_centroid_distance_m` **and** any of:

- **3D IoU ≥ `iou_threshold`** — the ordinary case, two cameras seeing the same body
- **containment ≥ `containment_threshold`** — one view is a subset of the other, e.g. one camera sees
  only a torso while another sees the whole person (low IoU, high containment)
- **footprint IoU ≥ `footprint_iou_threshold` and vertical gap ≤ `max_vertical_gap_m`** — the
  occlusion case: a person split by an intervening table arrives as two masks whose 3D boxes barely
  touch or do not touch at all, but which stand on the same footprint, one above the other

Grouping is **single-linkage** and transitive, which is right because a chain of partial views should
collapse to one object. The centroid gate is what stops it chaining across a room. Fragments from the
**same camera** are merged too, since one object routinely arrives as several instance ids.

### `up_axis` is configuration, not convention

The vertical world axis is whatever the calibration's `camera_pose` made it — for the artekmed exports
it is **y**, verifiable by printing object centres: the vertical axis spans a narrow ~0.8 m band of
plausible object heights while the horizontals span 4–7 m of room. A wrong value does not fail loudly;
the footprint rule simply stops merging vertically split objects.

In a **YAML config write `up_axis: "axis_y"`, not `"y"`** — yaml-cpp resolves a bare *and a quoted*
`y` to boolean `true`, so the short form arrives as `True`. The operator rejects a boolean with an
error that says so.

## Rejecting non-objects and aggregates

This is what controls object *count*, and it needs calibrating against a scene whose contents you know.
Five mechanisms, applied in this order:

| stage | parameter | what it removes |
|---|---|---|
| per observation | `min_points` (in `tcn_instance_stats`) | specks |
| per observation | `min_extent_m` | flat slivers a mask fringe produces (e.g. 0.27 × 0.33 × **0.03** m), which no volume or overlap rule catches |
| per observation | `aggregate_*` | a box containing ≥ `aggregate_min_children` **mutually disjoint** smaller boxes of its class — a mask covering several objects. It is dropped and its constituents kept. |
| per detection | `min_cameras` | objects only one camera ever saw |
| per detection | `min_detection_points` | fused objects with too little evidence |

**The aggregate rule is why containment can be used as a merge rule at all.** One contained box is a
partial view (a torso inside a person) and still merges; several *disjoint* contained boxes mean the
container is a class-level blob, and merging through it would fuse distinct people into one identity.

Order matters: slivers are rejected first so one cannot count as a "child" that condemns a container.

### Measured, on the 4-camera replay (8 frames)

| `min_detection_points` | objects/frame | distinct ids | id-set changes |
|---|---|---|---|
| 0 | 22.0 | 24 | 4 |
| 2000 | 15.9 | 17 | 2 |
| **3000** (default) | **9.5** | **10** | **1** |

`min_cameras: 2` reaches the same place (8.5 / 9 / 1) by dropping single-camera objects outright.
Filtering on **evidence quantity** is preferred to filtering on **viewpoint count**, because a person
at the edge of the room may genuinely be visible to only one camera — so `min_cameras` stays at 1 by
default.

Counter-intuitively, **tightening `fusion_max_distance_m` makes the count worse** (18 → 22 → 24
detections at 1.0 → 0.4 → 0.25 m): the residual count is dominated by *under*-merging of one object's
views, not by over-merging of distinct objects. Measure before turning that knob.

## The oriented box

Each object carries two boxes: the world-axis `bbox_min`/`bbox_max`, and a box rotated about the
vertical (`yaw`, `oriented_extent`, `oriented_center`). The axis-aligned one is inflated for anything
standing at an angle to the world axes — see
[`tcn_instance_stats`](../tcn_instance_stats/README.md#the-yaw-oriented-box) for how a single camera's
orientation is measured.

Fusing them across cameras needs a decision the AABB does not, because the views disagree about the
angle. Averaging yaws is wrong (they wrap), and adopting the best-observed view's yaw outright produces
a box **larger** than the AABB whenever the views are spread out — on live data, 10.7 m² against the
axis-aligned 6.9 m². So:

- the box is the hull of **every** contributing view's corners, not the best view's box — otherwise a
  partial view gets reported as a tight fit
- that hull is evaluated at each view's yaw **and** at the axis-aligned box, and the tightest footprint
  wins

The axis-aligned box has to be an explicit candidate rather than falling out of `yaw = 0`: a
circumscribing rectangle at any angle but the minimum-area one pokes out past the AABB at its corners,
so the hull of those corners can exceed the axis-aligned union. Listing it is what makes **"the oriented
box is the AABB refined, never inflated"** true by construction, and that invariant is what makes the
box safe to consume.

Both boxes are smoothed together or not at all. The oriented extents are measured *in* the yaw frame, so
once the principal axis swings past `max_yaw_smoothing_delta_rad` the previous ones describe a box that
never existed and the new one is adopted whole — and since the pair is reported side by side, smoothing
one while adopting the other would make the invariant above appear violated by nothing but lag.

Association still matches on the **axis-aligned** box. That is conservative: it can over-merge two
objects whose AABBs overlap while their oriented boxes do not, but it never under-merges.
Rotated-rectangle overlap is the upgrade when that matters.

## Identity: what an id means

- a new detection starts a **tentative** track, reported only after `min_hits` observations, so a
  one-frame false positive never gets an id
- an unobserved track is kept for `max_age` frames, holding its last box, then **retired**
- ids are **never reused**, so a printed id refers to exactly one physical object for the process
  lifetime
- an object away longer than `max_age` gets a **new id** — a deliberate refusal to claim continuity
  that was not observed. Position-based revival would bind the wrong object whenever two of a class
  swap places out of view.
- a **prompt change resets every track**: class ids are 1-based prompt positions, so a vocabulary
  change renumbers them and an existing identity would come to mean something else

There is no motion model and no appearance model. The input runs at the mask rate (~5 fps) and carries
**no confidence values** (scores number the instances but never reach the panoptic map), so a Kalman
filter would be fitting noise. What the tracker does have is an explicit lifecycle.

## Message shape

```python
{"acq_timestamp": 4624253458484510,
 "objects": [{"track_id": 7, "class_id": 1, "class_name": "person",
              "centroid": (x, y, z), "bbox_min": (...), "bbox_max": (...), "extent": (...),
              "yaw": 0.67, "oriented_extent": (...), "oriented_center": (...),
              "num_points": 51427, "cameras": [0, 2],
              "hits": 12, "misses": 0, "age": 14}]}
```

## Box overlay

`ObjectBoxRendererOp` emits one `LINES_3D` tensor per class (12 edges → 24 vertices per box), coloured
from the same `build_panoptic_lut` the points use, so a box reads as "the box around *those* points".
It draws the **oriented** box by default (`oriented=False` for the axis-aligned one), falling back per
object to the AABB when an object has no usable orientation.
It emits every configured class every frame; a class with no objects gets a single **NaN** segment,
which the rasteriser culls — an empty tensor would make HolovizOp special-case the shape.

Append its specs to the point-cloud viewer's spec list:

```python
cloud_specs.extend(box_input_specs(classes, lut, line_width=3.0))
self.add_flow(tracker_op, box_renderer, {("objects", "objects")})
self.add_flow(box_renderer, cloud_visualizer, {("boxes", "receivers")})
```

## Configuration

`object_tracking` in the example application's YAML; see the comments there. Requires
`mask_depth_join.enabled`.

## Tests

```bash
python3 tests/test_association.py   # 43 -- geometry, fusion, oriented boxes, filters, matching
python3 tests/test_tracker.py       # 20 -- lifecycle, id stability, class reset, box smoothing
python3 tests/test_render.py        # 12 -- box edges, oriented boxes, degenerate cases (numpy only)
```

The tracker tests are the ones that matter: an object present for N frames keeps one id; an occlusion
shorter than `max_age` keeps it and a longer one does not; two same-class objects crossing do not
swap; per-frame instance-id churn does not churn the track id.

## Known limitations

- **No confidence.** Association is purely geometric.
- **Mask bleed at occlusion boundaries** puts points on background surfaces; the percentile trim in
  `tcn_instance_stats` mitigates it up to its breakdown point and does not eliminate it.
- **A fused object's yaw is one of the contributing views' yaws**, chosen by area, not a re-estimate
  from the fused points — the fusion never sees points, only boxes. For an object whose views disagree
  strongly the result falls back to the axis-aligned box, which is correct but says nothing about
  orientation.
- **Object count needs calibrating per scene.** The defaults bring the 4-camera replay to ~9.5
  objects/frame (from 22), but the right thresholds depend on depth resolution and scene scale. Some
  boxes remain larger than a person, which is either genuinely-merged pairs or mask bleed — the box
  overlay in the viewer is the fastest way to tell them apart.
- **Calibration error inflates fused boxes** directly, and the analytic-projection gate for the
  mask/depth join is still unwritten.
