"""Holoscan operators wrapping the pure association and tracking logic.

Thin on purpose: everything with a decision in it lives in `association.py` and `tracker.py`, which are
host-testable. These operators only move data across ports.
"""
import logging

import numpy as np
from holoscan.core import ConditionType, IOSpec, Operator, OperatorSpec

from operators.tcn_artekmed.tcn_util.frame_identity import acq_timestamp_consensus

from .association import UP_AXIS_Y, UP_AXIS_Z, Observation, fuse_observations
from .tracker import ObjectTracker

log = logging.getLogger(__name__)

# Row columns, mirroring InstanceStatColumn in cuda/tcn_instance_stats_kernel.cuh. Positional by
# necessity -- a tensor has no field names -- so the two must be changed together.
COL_CAMERA, COL_COUNT = 0, 1
COL_CENTROID = slice(2, 5)
COL_MIN = slice(5, 8)
COL_MAX = slice(8, 11)
COL_SIGMA = slice(11, 14)
N_COLUMNS = 14

PANOPTIC_CLASS_SHIFT = 8
PANOPTIC_INSTANCE_MASK = 0xFF


def observations_from_message(msg, rows_name="rows", labels_name="labels"):
    """Decode one `tcn_instance_stats` message into Observations.

    The tensors are in HOST memory, so numpy reads them without a device copy. Rows whose count is 0
    are the no-instance placeholder and are skipped.
    """
    rows = msg.get(rows_name)
    labels = msg.get(labels_name)
    if rows is None or labels is None:
        return []
    rows = np.asarray(rows).reshape(-1, N_COLUMNS)
    labels = np.asarray(labels).reshape(-1)
    out = []
    for i in range(rows.shape[0]):
        count = int(rows[i, COL_COUNT])
        label = int(labels[i])
        if count <= 0 or label == 0:
            continue
        out.append(Observation(
            class_id=label >> PANOPTIC_CLASS_SHIFT,
            instance_id=label & PANOPTIC_INSTANCE_MASK,
            camera_index=int(rows[i, COL_CAMERA]),
            num_points=count,
            centroid=tuple(float(v) for v in rows[i, COL_CENTROID]),
            box=(tuple(float(v) for v in rows[i, COL_MIN]),
                 tuple(float(v) for v in rows[i, COL_MAX])),
            sigma=tuple(float(v) for v in rows[i, COL_SIGMA]),
        ))
    return out


class InstanceFusionOp(Operator):
    """Merges every camera's view of one object into one detection per object, per frame.

    Stateless: the output depends only on this frame's observations, which is what makes it testable
    with fixed inputs.
    """

    def __init__(self, fragment, *args, iou_threshold=0.15, containment_threshold=0.6,
                 max_centroid_distance_m=1.0, footprint_iou_threshold=0.4,
                 max_vertical_gap_m=0.5, up_axis="y", verbose=False, **kwargs):
        # The vertical world axis is a property of the calibration, not a convention: for the
        # artekmed exports it is y. A wrong value does not fail loudly -- the footprint rule simply
        # stops merging vertically split objects -- so it is configured, never assumed.
        #
        # Accepts "axis_x"/"axis_y"/"axis_z", the bare letters, or 0/1/2. In a YAML CONFIG use the
        # `axis_*` spelling: yaml-cpp resolves a bare *and even a quoted* `y` to boolean true, so
        # `up_axis: "y"` arrives here as True. That is checked for explicitly below, because the
        # resulting error would otherwise be baffling.
        if isinstance(up_axis, bool):
            raise ValueError(
                "up_axis arrived as a boolean, which means the YAML value was `y` or `n`: yaml-cpp "
                "resolves those to booleans even when quoted. Write `up_axis: \"axis_y\"` instead.")
        axes = {"x": 0, "y": UP_AXIS_Y, "z": UP_AXIS_Z,
                "axis_x": 0, "axis_y": UP_AXIS_Y, "axis_z": UP_AXIS_Z,
                "0": 0, "1": UP_AXIS_Y, "2": UP_AXIS_Z}
        key = str(up_axis).strip().lower()
        if key not in axes:
            raise ValueError(f"up_axis must be one of axis_x/axis_y/axis_z (or x/y/z, or 0/1/2), "
                             f"got {up_axis!r}")
        self.up_axis = axes[key]
        self.iou_threshold = float(iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.max_centroid_distance_m = float(max_centroid_distance_m)
        self.footprint_iou_threshold = float(footprint_iou_threshold)
        self.max_vertical_gap_m = float(max_vertical_gap_m)
        self.verbose = bool(verbose)
        self.frames = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # One port for every camera's stats operator: they all belong to the same frame group, and the
        # group is what fusion operates on.
        spec.input("receivers", size=IOSpec.ANY_SIZE)
        spec.output("detections")

    def compute(self, op_input, op_output, context):
        messages = op_input.receive("receivers")
        acq = acq_timestamp_consensus(op_input, "receivers", log)
        observations = []
        for msg in messages or ():
            if msg is not None:
                observations.extend(observations_from_message(msg))

        detections = fuse_observations(
            observations,
            iou_threshold=self.iou_threshold,
            containment_threshold=self.containment_threshold,
            max_centroid_distance_m=self.max_centroid_distance_m,
            footprint_iou_threshold=self.footprint_iou_threshold,
            max_vertical_gap_m=self.max_vertical_gap_m,
            up_axis=self.up_axis)
        self.frames += 1
        if self.verbose:
            log.info(f"InstanceFusionOp: frame {self.frames}: {len(observations)} observation(s) "
                     f"from {len({o.camera_index for o in observations})} camera(s) -> "
                     f"{len(detections)} detection(s)")
        op_output.emit({"acq_timestamp": acq, "detections": detections}, "detections",
                       acq_timestamp=acq)


class ObjectTrackerOp(Operator):
    """Assigns persistent ids to detections over time.

    Stateful, so its correctness is a property of a *sequence*; the lifecycle rules and their tests
    live in `tracker.py`.
    """

    def __init__(self, fragment, *args, min_hits=3, max_age=8, iou_threshold=0.1,
                 max_centroid_distance_m=1.0, box_smoothing=0.5, class_names=None,
                 verbose=False, **kwargs):
        self.tracker = ObjectTracker(min_hits=min_hits, max_age=max_age,
                                     iou_threshold=iou_threshold,
                                     max_centroid_distance_m=max_centroid_distance_m,
                                     box_smoothing=box_smoothing)
        # class id -> name, for readable output. Class ids are 1-based prompt positions.
        self.class_names = {i + 1: str(n) for i, n in enumerate(class_names or [])}
        self.verbose = bool(verbose)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("detections")
        # Conditionless: prompts arrive only when someone changes them, and gating on them would stall
        # the pipeline until the first update.
        spec.input("prompts").condition(ConditionType.NONE)
        spec.output("objects")

    def _maybe_reset_on_prompt_change(self, op_input):
        """A vocabulary change renumbers class ids, so identity cannot survive it."""
        try:
            msg = op_input.receive("prompts")
        except Exception:
            return
        prompts = msg.get("text_prompts") if msg and hasattr(msg, "get") else None
        if not prompts:
            return
        if self.tracker.note_class_signature(prompts):
            self.class_names = {i + 1: str(n) for i, n in enumerate(prompts)}
            log.info(f"ObjectTrackerOp: class definition changed to {list(prompts)}; tracks reset")

    def compute(self, op_input, op_output, context):
        self._maybe_reset_on_prompt_change(op_input)
        msg = op_input.receive("detections")
        detections = (msg or {}).get("detections", [])
        acq = (msg or {}).get("acq_timestamp", -1)
        tracks = self.tracker.update(detections)
        objects = [t.as_dict(self.class_names) for t in tracks]
        if self.verbose:
            log.info(f"ObjectTrackerOp: {len(detections)} detection(s) -> {len(objects)} "
                     f"confirmed track(s), {len(self.tracker.tracks)} live")
        op_output.emit({"acq_timestamp": acq, "objects": objects}, "objects", acq_timestamp=acq)

    def stop(self):
        log.info(f"ObjectTrackerOp: {self.tracker._next_id - 1} id(s) issued, "
                 f"{self.tracker.retired} track(s) retired")


class ObjectConsoleSinkOp(Operator):
    """Prints the tracked objects, and a grep-able verdict at shutdown.

    The verdict distinguishes "no objects were tracked" from "the operator never ran", which a
    per-frame print cannot.
    """

    def __init__(self, fragment, *args, print_every=1, **kwargs):
        self.print_every = max(1, int(print_every))
        self.frames = 0
        self.ids_seen = set()
        self.id_sets = []
        self.total_objects = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("objects")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("objects") or {}
        objects = msg.get("objects", [])
        self.frames += 1
        self.total_objects += len(objects)
        frame_ids = tuple(sorted(o["track_id"] for o in objects))
        self.ids_seen.update(frame_ids)
        self.id_sets.append(frame_ids)

        if self.frames % self.print_every:
            return
        log.info(f"--- tracked objects, frame {self.frames} "
                 f"(acq={msg.get('acq_timestamp')}): {len(objects)} object(s)")
        for o in objects:
            c, e = o["centroid"], o["extent"]
            log.info(f"    #{o['track_id']:<3d} {o['class_name']:<12s} "
                     f"centre=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) m  "
                     f"size=({e[0]:.2f}x{e[1]:.2f}x{e[2]:.2f}) m  "
                     f"pts={o['num_points']:<7d} cams={o['cameras']} "
                     f"hits={o['hits']}")

    def stop(self):
        if not self.frames:
            log.error("ObjectConsoleSinkOp: NO FRAMES -- the tracking path never ran")
            return
        # Churn: how often the reported id set changed. A stable scene should change it rarely.
        churn = sum(1 for a, b in zip(self.id_sets, self.id_sets[1:]) if a != b)
        verdict = "PASS" if self.ids_seen else "NO OBJECTS"
        log.info(f"ObjectConsoleSinkOp: {verdict} -- {self.frames} frames, "
                 f"{len(self.ids_seen)} distinct id(s) {sorted(self.ids_seen)}, "
                 f"{self.total_objects / self.frames:.1f} object(s)/frame, {churn} id-set change(s)")
