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
  association.py   pure: 3D IoU, containment, footprint IoU, box union, weighted centroid, matching
  tracker.py       pure: Track, lifecycle, id allocation, class-change reset
  ops.py           InstanceFusionOp, ObjectTrackerOp, ObjectConsoleSinkOp
  render.py        ObjectBoxRendererOp, box_input_specs, box_line_vertices
  tests/           46 host checks, no holoscan/cupy/numpy needed for the algorithm ones
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
              "num_points": 51427, "cameras": [0, 2],
              "hits": 12, "misses": 0, "age": 14}]}
```

## Box overlay

`ObjectBoxRendererOp` emits one `LINES_3D` tensor per class (12 edges → 24 vertices per box), coloured
from the same `build_panoptic_lut` the points use, so a box reads as "the box around *those* points".
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
python3 tests/test_association.py   # 23 -- geometry, fusion, matching (host, no deps)
python3 tests/test_tracker.py       # 16 -- lifecycle, id stability, class reset (host, no deps)
python3 tests/test_render.py        #  7 -- box edges, degenerate cases (host, numpy only)
```

The tracker tests are the ones that matter: an object present for N frames keeps one id; an occlusion
shorter than `max_age` keeps it and a longer one does not; two same-class objects crossing do not
swap; per-frame instance-id churn does not churn the track id.

## Known limitations

- **No confidence.** Association is purely geometric.
- **Mask bleed at occlusion boundaries** puts points on background surfaces; the σ-trim in
  `tcn_instance_stats` mitigates it and does not eliminate it.
- **Over-segmentation is the current failure mode**, not crashes: on the 4-camera replay this reports
  ~24 objects/frame for a scene with a few people, with some boxes 2.5 m across. The thresholds and
  `min_points` need calibrating against a scene whose true contents are known — the console dump
  exists to make that visible.
- **Calibration error inflates fused boxes** directly, and the analytic-projection gate for the
  mask/depth join is still unwritten.
