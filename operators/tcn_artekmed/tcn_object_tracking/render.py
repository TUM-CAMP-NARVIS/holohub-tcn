"""Render tracked bounding boxes as 3D line segments, into the same view as the point cloud.

Separate from `ops.py` because this is the only module that needs HolovizOp, and because rendering is
optional -- the object stream is useful headless.
"""
import logging

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import Operator, OperatorSpec
from holoscan.operators import HolovizOp

log = logging.getLogger(__name__)

#: The 12 edges of an axis-aligned box, as pairs of corner indices. A corner index is a bitmask over
#: (x, y, z): bit d set means take max on axis d. Independent of which axis is "up" -- an AABB's edge
#: set does not depend on orientation, unlike the fusion rules.
_BOX_EDGES = (
    (0b000, 0b001), (0b001, 0b011), (0b011, 0b010), (0b010, 0b000),   # y-z face at min x
    (0b100, 0b101), (0b101, 0b111), (0b111, 0b110), (0b110, 0b100),   # y-z face at max x
    (0b000, 0b100), (0b001, 0b101), (0b011, 0b111), (0b010, 0b110),   # the four x-parallel edges
)

VERTICES_PER_BOX = 2 * len(_BOX_EDGES)     # 24: LINES_3D consumes consecutive vertex PAIRS


def box_tensor_name(class_id: int) -> str:
    """Tensor (and Holoviz spec) name for one class's boxes. One place, so the app cannot drift."""
    return f"boxes_class_{class_id}"


def box_line_vertices(boxes) -> np.ndarray:
    """`[(min_xyz, max_xyz), ...]` -> `[1, N*24, 3]` float32 line-segment vertices.

    Returns a single degenerate NaN segment when there are no boxes: an empty tensor would make
    HolovizOp special-case the shape, and a NaN vertex is culled by the rasteriser -- the same
    convention `tcn_labeled_pointcloud` uses for an empty class.
    """
    if not boxes:
        return np.full((1, 2, 3), np.nan, dtype=np.float32)
    out = np.empty((1, len(boxes) * VERTICES_PER_BOX, 3), dtype=np.float32)
    w = 0
    for lo, hi in boxes:
        corner = [(lo[0] if not (i & 0b100) else hi[0],
                   lo[1] if not (i & 0b001) else hi[1],
                   lo[2] if not (i & 0b010) else hi[2]) for i in range(8)]
        for a, b in _BOX_EDGES:
            out[0, w] = corner[a]
            out[0, w + 1] = corner[b]
            w += 2
    return out


def box_input_specs(classes, lut, line_width=2.0, opacity=1.0):
    """HolovizOp specs for the box overlays, one per class so each gets its own colour.

    Colours come from the same panoptic LUT the point clouds use, so a class is the same colour as
    its points -- which is what makes a box readable as "the box around *those* points".

    `lut` is the cupy array from `build_panoptic_lut`; indexing it by `class << 8` gives the class's
    base colour, exactly as the point-cloud specs do.
    """
    specs = []
    for cls in classes:
        spec = HolovizOp.InputSpec(box_tensor_name(cls), HolovizOp.InputType.LINES_3D)
        rgba = cp.asnumpy(lut[cls << 8]).astype(float) / 255.0
        spec.color = [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(opacity)]
        spec.line_width = float(line_width)
        specs.append(spec)
    return specs


class ObjectBoxRendererOp(Operator):
    """Turns the tracked-object stream into per-class 3D line segments for HolovizOp.

    Emits one tensor per configured class, every frame, so the static Holoviz spec list built at
    compose time always has its inputs. A class with no objects this frame emits the degenerate NaN
    segment rather than nothing.

    Renders only *confirmed* tracks, because that is what the tracker emits -- a box appearing for one
    frame and vanishing is worse than a slightly late box.
    """

    def __init__(self, fragment, *args, classes, device=0, **kwargs):
        self.classes = [int(c) for c in classes]
        self.device = int(device)
        self.frames = 0
        self.drawn = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("objects")
        spec.output("boxes")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("objects") or {}
        objects = msg.get("objects", [])
        acq = msg.get("acq_timestamp", -1)

        by_class = {cls: [] for cls in self.classes}
        for o in objects:
            if o["class_id"] in by_class:
                by_class[o["class_id"]].append((o["bbox_min"], o["bbox_max"]))

        out = {}
        with cp.cuda.Device(self.device):
            for cls, boxes in by_class.items():
                out[box_tensor_name(cls)] = hs.as_tensor(
                    cp.asarray(box_line_vertices(boxes)))
        self.frames += 1
        self.drawn += sum(len(b) for b in by_class.values())
        op_output.emit(out, "boxes", acq_timestamp=acq)

    def stop(self):
        if self.frames:
            log.info(f"ObjectBoxRendererOp: {self.drawn / self.frames:.1f} box(es)/frame over "
                     f"{self.frames} frames")
