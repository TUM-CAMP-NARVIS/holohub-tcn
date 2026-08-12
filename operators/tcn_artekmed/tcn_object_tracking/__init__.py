"""Tracked objects from segmented point clouds: cross-camera fusion, identity over time, console dump.

Turns the per-camera per-instance rows from `tcn_instance_stats` into one entry per physical object,
with a bounding box, a centre of mass and an id that stays the same while the object is observed.

Three stages, deliberately separate:

- **fusion** (`InstanceFusionOp`, stateless) merges every camera's view of one object at a single
  timestamp -- and merges fragments *within* one camera too, since a mask routinely splits.
- **tracking** (`ObjectTrackerOp`, stateful) assigns persistent ids across frames.
- **sink** (`ObjectConsoleSinkOp`) prints the result.

The algorithms live in `association.py` and `tracker.py`, which import neither holoscan nor cupy, so
they are host-testable in milliseconds -- see `tests/`. That matters because identity bugs (swapped
ids, churn, resurrection) are invisible in a rendered scene and obvious in a unit test.

Why instance ids cannot be used directly: they are assigned per class **by descending detection
score, per camera, per frame** (`tcn_langsam.helpers.plan_panoptic_paint`), so "instance 1" means "the
most confident detection in this camera this frame". It changes with confidence ranking and is
unrelated between cameras.
"""

__all__ = [
    # operators
    "InstanceFusionOp",
    "ObjectTrackerOp",
    "ObjectConsoleSinkOp",
    # pure algorithm, exported for reuse and for tests
    "Observation",
    "Detection",
    "fuse_observations",
    "merge_observations",
    "match_detections_to_tracks",
    "box_iou",
    "box_containment",
    "box_union",
    "footprint_iou",
    "ObjectTracker",
    "Track",
]

_LAZY = {
    "InstanceFusionOp": (".ops", "InstanceFusionOp"),
    "ObjectTrackerOp": (".ops", "ObjectTrackerOp"),
    "ObjectConsoleSinkOp": (".ops", "ObjectConsoleSinkOp"),
    "Observation": (".association", "Observation"),
    "Detection": (".association", "Detection"),
    "fuse_observations": (".association", "fuse_observations"),
    "merge_observations": (".association", "merge_observations"),
    "match_detections_to_tracks": (".association", "match_detections_to_tracks"),
    "box_iou": (".association", "box_iou"),
    "box_containment": (".association", "box_containment"),
    "box_union": (".association", "box_union"),
    "footprint_iou": (".association", "footprint_iou"),
    "ObjectTracker": (".tracker", "ObjectTracker"),
    "Track": (".tracker", "Track"),
}


def __getattr__(name):
    # Lazy: `ops` imports holoscan, but association/tracker must stay importable on a plain host so
    # their tests need nothing installed.
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    return getattr(import_module(module_name, __name__), attr)


def __dir__():
    return sorted(__all__)
