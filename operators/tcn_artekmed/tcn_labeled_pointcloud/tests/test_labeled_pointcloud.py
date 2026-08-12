"""Correctness gate for TcnLabeledPointcloudOp. Runs the real operator on synthetic inputs.

Must run inside the container (needs holoscan, cupy and the built module):

    PYTHONPATH=<build>/python/lib python3 tests/test_labeled_pointcloud.py

Expectations are derived from the input arrays with plain numpy (boolean selection), never by
restating the compaction under test.
"""
import logging
import sys

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator

from holohub.tcn_labeled_pointcloud import TcnLabeledPointcloudOp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_labeled_pointcloud")

H, W = 2, 5
N = H * W

# Positions are distinct and encode their own index, so a point gathered from the wrong source
# pixel is identifiable by its value rather than merely "different".
POSITIONS = np.arange(N * 3, dtype=np.float32).reshape(H, W, 3)

# Classes 1, 2 and background, with distinct instance ids so class-vs-instance confusion shows up.
#   idx: 0     1        2     3        4        5     6        7        8     9
#        bg   c1i1     c2i1   bg      c1i2     c2i2  c1i3     bg       bg    c2i3
LABELS = np.array([[0, (1 << 8) | 1, (2 << 8) | 1, 0, (1 << 8) | 2],
                   [(2 << 8) | 2, (1 << 8) | 3, 0, 0, (2 << 8) | 3]], dtype=np.uint16)


def expected_for(classes):
    """Points a class should contain, in source order -- computed with plain numpy selection."""
    flat_pos = POSITIONS.reshape(-1, 3)
    flat_lab = LABELS.reshape(-1)
    out = {}
    for cls in classes:
        if cls < 0:
            keep = flat_lab != 0
        else:
            keep = (flat_lab != 0) & ((flat_lab >> 8) == cls)
        out[cls] = (flat_pos[keep], flat_lab[keep])
    return out


class SourceOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("positions")
        spec.output("labels")

    def compute(self, op_input, op_output, context):
        op_output.emit({"pos": hs.as_tensor(cp.asarray(POSITIONS))}, "positions")
        op_output.emit({"lab": hs.as_tensor(cp.asarray(LABELS))}, "labels")


class CollectOp(Operator):
    def __init__(self, fragment, *args, classes, got, **kwargs):
        self.classes = list(classes)
        self.got = got
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for cls in self.classes:
            spec.input(f"in_{cls}")

    def compute(self, op_input, op_output, context):
        for cls in self.classes:
            msg = op_input.receive(f"in_{cls}")
            pos = cp.asnumpy(cp.asarray(msg.get("positions"))).reshape(-1, 3)
            lab = cp.asnumpy(cp.asarray(msg.get("labels"))).reshape(-1)
            self.got[cls] = (pos, lab)


class Harness(Application):
    def __init__(self, classes, got):
        self.classes = list(classes)
        self.got = got
        super().__init__()

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")
        src = SourceOp(self, CountCondition(self, count=1), name="src")
        op = TcnLabeledPointcloudOp(
            self,
            allocator=pool,
            classes=self.classes,
            cuda_device_ordinal=0,
            in_positions_tensor_name="pos",
            in_labels_tensor_name="lab",
            name="pointcloud",
        )
        ports = self.classes if self.classes else [-1]
        collect = CollectOp(self, classes=ports, got=self.got, name="collect")
        self.add_flow(src, op, {("positions", "positions"), ("labels", "labels")})
        for cls in ports:
            port = "class_all" if cls < 0 else f"class_{cls}"
            self.add_flow(op, collect, {(port, f"in_{cls}")})


def run(classes):
    got = {}
    Harness(classes, got).run()
    return got


def case_compaction_matches_numpy_selection():
    classes = [1, 2]
    got = run(classes)
    want = expected_for(classes)
    problems = []
    for cls in classes:
        wp, wl = want[cls]
        gp, gl = got.get(cls, (None, None))
        if gp is None:
            problems.append(f"class {cls}: no output")
            continue
        if gp.shape != wp.shape:
            problems.append(f"class {cls}: got {gp.shape[0]} points, want {wp.shape[0]}")
            continue
        if not np.array_equal(gp, wp):
            problems.append(f"class {cls}: positions differ -- got {gp.tolist()} want {wp.tolist()}")
        if not np.array_equal(gl, wl):
            problems.append(f"class {cls}: labels differ -- got {gl.tolist()} want {wl.tolist()}")
    return problems


def case_instance_ids_survive():
    """The packed label, not just the class, must reach the output."""
    got = run([1])
    _, lab = got[1]
    instances = sorted(int(v) & 0xFF for v in lab)
    if instances != [1, 2, 3]:
        return [f"instance ids lost: got {instances}, want [1, 2, 3]"]
    if any((int(v) >> 8) != 1 for v in lab):
        return [f"class byte wrong in {lab.tolist()}"]
    return []


def case_source_order_is_preserved():
    """Compaction must be stable: positions ascend because POSITIONS ascends with the index."""
    got = run([2])
    pos, _ = got[2]
    if not np.all(np.diff(pos[:, 0]) > 0):
        return [f"points are not in source order: {pos[:, 0].tolist()}"]
    return []


def case_absent_class_emits_one_nan_point():
    """A class with no points must still emit (a merger needs every input) and be unrenderable."""
    got = run([7])
    pos, lab = got[7]
    if pos.shape[0] != 1:
        return [f"absent class emitted {pos.shape[0]} points, want exactly 1"]
    if not np.all(np.isnan(pos)):
        return [f"absent class emitted a real point {pos.tolist()}, want NaN"]
    if int(lab[0]) != 0:
        return [f"absent class emitted label {int(lab[0])}, want 0"]
    return []


def case_empty_classes_emits_every_non_background_point():
    got = run([])
    wp, wl = expected_for([-1])[-1]
    gp, gl = got[-1]
    if not np.array_equal(gp, wp) or not np.array_equal(gl, wl):
        return [f"class_all: got {gp.shape[0]} points, want {wp.shape[0]} "
                f"(background must be excluded)"]
    return []


def case_background_is_never_emitted():
    got = run([0])
    pos, _ = got[0]
    # Class 0 can only be background, which is excluded, so this is the absent-class path.
    if pos.shape[0] != 1 or not np.all(np.isnan(pos)):
        return [f"class 0 emitted {pos.shape[0]} real points; background must never be a point"]
    return []


CASES = [
    case_compaction_matches_numpy_selection,
    case_instance_ids_survive,
    case_source_order_is_preserved,
    case_absent_class_emits_one_nan_point,
    case_empty_classes_emits_every_non_background_point,
    case_background_is_never_emitted,
]

if __name__ == "__main__":
    failures = []
    for fn in CASES:
        try:
            problems = fn()
        except Exception as e:                        # noqa: BLE001 - report, do not mask
            problems = [f"{fn.__name__}: raised {e!r}"]
        if problems:
            failures.append(fn.__name__)
            print("FAIL", fn.__name__)
            for p in problems:
                print("     ", p)
        else:
            print("PASS", fn.__name__)
    print(f"{len(CASES) - len(failures)}/{len(CASES)} cases passed")
    sys.exit(1 if failures else 0)
