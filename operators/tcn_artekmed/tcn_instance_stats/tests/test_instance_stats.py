"""Correctness gate for TcnInstanceStatsOp. Runs the real operator on synthetic point grids.

    PYTHONPATH=<build>/python/lib python3 tests/test_instance_stats.py

Every expectation is a hand-computed value or a numpy reduction over the same input, never a restating
of the kernel.
"""
import logging
import sys

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator

from holohub.tcn_instance_stats import TcnInstanceStatsOp

logging.basicConfig(level=logging.WARNING)

# Row columns, mirroring InstanceStatColumn in cuda/tcn_instance_stats_kernel.cuh.
CAM, CNT, CX, CY, CZ, MNX, MNY, MNZ, MXX, MXY, MXZ, SGX, SGY, SGZ = range(14)

H, W = 8, 8
PACK = lambda cls, inst: np.uint16((cls << 8) | inst)


def grid(points_by_label):
    """Build (positions [H,W,3], labels [H,W]) by laying each label's points out in row order."""
    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    i = 0
    for label, pts in points_by_label.items():
        for p in pts:
            positions[i] = p
            labels[i] = label
            i += 1
    assert i <= H * W, "too many points for the grid"
    return positions.reshape(H, W, 3), labels.reshape(H, W)


class SourceOp(Operator):
    def __init__(self, fragment, *args, positions, labels, **kwargs):
        self.positions, self.labels = positions, labels
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("positions")
        spec.output("labels")

    def compute(self, op_input, op_output, context):
        op_output.emit({"pos": hs.as_tensor(cp.asarray(self.positions))}, "positions")
        op_output.emit({"lab": hs.as_tensor(cp.asarray(self.labels))}, "labels")


class CollectOp(Operator):
    def __init__(self, fragment, *args, out, **kwargs):
        self.out = out
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("instances")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("instances")
        # Host tensors: numpy reads them directly, no device copy.
        rows = np.asarray(msg.get("rows")).reshape(-1, 14)
        labels = np.asarray(msg.get("labels")).reshape(-1)
        self.out["rows"], self.out["labels"] = rows.copy(), labels.copy()


class Harness(Application):
    def __init__(self, positions, labels, out, **op_kwargs):
        self.positions, self.labels, self.out, self.op_kwargs = positions, labels, out, op_kwargs
        super().__init__()

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")
        src = SourceOp(self, CountCondition(self, count=1),
                       positions=self.positions, labels=self.labels, name="src")
        op = TcnInstanceStatsOp(self, allocator=pool, cuda_device_ordinal=0,
                                in_positions_tensor_name="pos", in_labels_tensor_name="lab",
                                name="stats", **self.op_kwargs)
        sink = CollectOp(self, out=self.out, name="sink")
        self.add_flow(src, op, {("positions", "positions"), ("labels", "labels")})
        self.add_flow(op, sink, {("instances", "instances")})


def run(points_by_label, **op_kwargs):
    positions, labels = grid(points_by_label)
    out = {}
    Harness(positions, labels, out, **op_kwargs).run()
    return out["rows"], out["labels"]


def row_for(rows, labels, label):
    idx = np.where(labels == label)[0]
    return rows[idx[0]] if len(idx) else None


# ── cases ─────────────────────────────────────────────────────────────────────────────────────────

def case_exact_box_and_centroid():
    """A planted cuboid, no outliers: box and centroid must be exact."""
    pts = [(x, y, z) for x in (1.0, 2.0) for y in (10.0, 11.0) for z in (-5.0, -4.0)]  # 8 corners
    label = PACK(1, 1)
    rows, labels = run({label: pts}, min_points=1, sigma_k=100.0)
    r = row_for(rows, labels, label)
    if r is None:
        return ["no row emitted for the planted instance"]
    want_c = np.mean(pts, axis=0)
    problems = []
    if int(r[CNT]) != len(pts):
        problems.append(f"count {int(r[CNT])} != {len(pts)}")
    if not np.allclose(r[[CX, CY, CZ]], want_c, atol=1e-5):
        problems.append(f"centroid {r[[CX,CY,CZ]]} != {want_c}")
    if not np.allclose(r[[MNX, MNY, MNZ]], np.min(pts, axis=0), atol=1e-5):
        problems.append(f"bbox_min {r[[MNX,MNY,MNZ]]} != {np.min(pts, axis=0)}")
    if not np.allclose(r[[MXX, MXY, MXZ]], np.max(pts, axis=0), atol=1e-5):
        problems.append(f"bbox_max {r[[MXX,MXY,MXZ]]} != {np.max(pts, axis=0)}")
    return problems


def case_negative_coordinates_survive_the_minmax_encoding():
    """atomicMin/Max run on ordered-int-encoded floats; negatives are where that encoding breaks."""
    pts = [(-3.0, -2.0, -1.0), (-1.0, -4.0, -9.0), (2.0, 0.0, 0.5)]
    label = PACK(2, 1)
    rows, labels = run({label: pts}, min_points=1, sigma_k=100.0)
    r = row_for(rows, labels, label)
    problems = []
    if not np.allclose(r[[MNX, MNY, MNZ]], np.min(pts, axis=0), atol=1e-5):
        problems.append(f"bbox_min {r[[MNX,MNY,MNZ]]} != {np.min(pts, axis=0)} "
                        f"(ordered-int encoding is wrong for negatives)")
    if not np.allclose(r[[MXX, MXY, MXZ]], np.max(pts, axis=0), atol=1e-5):
        problems.append(f"bbox_max {r[[MXX,MXY,MXZ]]} != {np.max(pts, axis=0)}")
    return problems


def case_background_is_excluded():
    pts = [(0.0, 0.0, 0.0)] * 10
    rows, labels = run({np.uint16(0): pts}, min_points=1, sigma_k=100.0)
    if len(labels) and labels[0] != 0:
        return [f"unexpected labels {labels}"]
    if rows.shape[0] != 1 or int(rows[0][CNT]) != 0:
        return [f"background produced {rows.shape[0]} row(s) with count {int(rows[0][CNT])}; "
                f"expected one all-zero placeholder row"]
    return []


def case_outlier_blob_is_trimmed():
    """A far blob must inflate the untrimmed box and leave the trimmed one alone.

    This is the mask-bleed case: a mask edge landing on a distant surface.
    """
    core = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
            (0.05, 0.05, 0.05), (0.02, 0.08, 0.03), (0.07, 0.01, 0.09), (0.03, 0.06, 0.02)] * 4
    blob = [(9.0, 9.0, 9.0)]
    label = PACK(3, 1)
    loose = row_for(*run({label: core + blob}, min_points=1, sigma_k=100.0), label)
    tight = row_for(*run({label: core + blob}, min_points=1, sigma_k=1.5), label)
    problems = []
    if loose[MXX] < 8.0:
        problems.append(f"untrimmed box should include the blob, got max_x {loose[MXX]}")
    if tight[MXX] > 1.0:
        problems.append(f"trimmed box still includes the blob: max_x {tight[MXX]}")
    if int(tight[CNT]) != len(core):
        problems.append(f"trimmed count {int(tight[CNT])} != {len(core)} (core points)")
    # sigma is reported BEFORE trimming, so it stays large -- that is the bleed indicator.
    if tight[SGX] < 1.0:
        problems.append(f"pre-trim sigma_x {tight[SGX]} should stay large as a bleed signal")
    return problems


def case_min_points_drops_small_instances():
    small, big = PACK(4, 1), PACK(4, 2)
    rows, labels = run({small: [(0.0, 0.0, 0.0)] * 3,
                        big: [(5.0, 5.0, 5.0)] * 20}, min_points=10, sigma_k=100.0)
    problems = []
    if row_for(rows, labels, small) is not None:
        problems.append("a 3-point instance survived min_points=10")
    if row_for(rows, labels, big) is None:
        problems.append("the 20-point instance was dropped by min_points=10")
    return problems


def case_multiple_instances_and_classes_are_separate():
    a, b, c = PACK(1, 1), PACK(1, 2), PACK(2, 1)
    pts = {a: [(0.0, 0.0, 0.0)] * 8, b: [(4.0, 0.0, 0.0)] * 8, c: [(0.0, 7.0, 0.0)] * 8}
    rows, labels = run(pts, min_points=1, sigma_k=100.0)
    problems = []
    if sorted(int(x) for x in labels) != sorted(int(x) for x in pts):
        problems.append(f"labels {sorted(int(x) for x in labels)} != {sorted(int(x) for x in pts)}")
    for label, want in ((a, 0.0), (b, 4.0), (c, 0.0)):
        r = row_for(rows, labels, label)
        if r is None or abs(r[CX] - want) > 1e-5:
            problems.append(f"label {int(label)} centroid_x {None if r is None else r[CX]} != {want}")
    return problems


def case_camera_index_is_stamped():
    label = PACK(1, 1)
    rows, labels = run({label: [(1.0, 1.0, 1.0)] * 8}, min_points=1, sigma_k=100.0, camera_index=3)
    r = row_for(rows, labels, label)
    return [] if int(r[CAM]) == 3 else [f"camera_index {int(r[CAM])} != 3"]


def case_empty_frame_emits_a_zero_row():
    rows, labels = run({}, min_points=1)
    if rows.shape != (1, 14) or int(rows[0][CNT]) != 0:
        return [f"expected one all-zero row, got shape {rows.shape} count "
                f"{int(rows[0][CNT]) if rows.size else 'n/a'}"]
    return []


CASES = [
    case_exact_box_and_centroid,
    case_negative_coordinates_survive_the_minmax_encoding,
    case_background_is_excluded,
    case_outlier_blob_is_trimmed,
    case_min_points_drops_small_instances,
    case_multiple_instances_and_classes_are_separate,
    case_camera_index_is_stamped,
    case_empty_frame_emits_a_zero_row,
]

if __name__ == "__main__":
    failures = []
    for fn in CASES:
        try:
            problems = fn()
        except Exception as e:                        # noqa: BLE001 - report, do not mask
            problems = [f"raised {e!r}"]
        if problems:
            failures.append(fn.__name__)
            print("FAIL", fn.__name__)
            for p in problems:
                print("     ", p)
        else:
            print("PASS", fn.__name__)
    print(f"{len(CASES) - len(failures)}/{len(CASES)} cases passed")
    sys.exit(1 if failures else 0)
