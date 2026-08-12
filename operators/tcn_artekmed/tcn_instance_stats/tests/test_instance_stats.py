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
    rows, labels = run({label: pts}, min_points=1, trim_percentile=0.0)
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
    rows, labels = run({label: pts}, min_points=1, trim_percentile=0.0)
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
    rows, labels = run({np.uint16(0): pts}, min_points=1, trim_percentile=0.0)
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
    loose = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.0), label)
    # 1 outlier in 33 points is 3%, so the trim must discard at least that from each end.
    tight = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.05), label)
    problems = []
    if loose[MXX] < 8.0:
        problems.append(f"untrimmed box should include the blob, got max_x {loose[MXX]}")
    if tight[MXX] > 1.0:
        problems.append(f"trimmed box still includes the blob: max_x {tight[MXX]}")
    if int(tight[CNT]) > len(core):
        problems.append(f"trimmed count {int(tight[CNT])} exceeds the {len(core)} core points")
    # sigma is reported BEFORE trimming, so it stays large -- that is the bleed indicator.
    if tight[SGX] < 1.0:
        problems.append(f"pre-trim sigma_x {tight[SGX]} should stay large as a bleed signal")
    return problems


def case_percentile_trim_survives_the_masking_effect():
    """The case sigma-based trimming provably cannot handle.

    A blob holding 23% of an instance's points, 3 m away, inflates sigma to ~1.2 m -- so a +-2 sigma
    window spans [-1.6, 3.3] and CONTAINS the blob. Iterating cannot escape it: the first pass rejects
    nothing, so it is already at a fixed point.

    A percentile bound has a BREAKDOWN POINT equal to trim_percentile: it discards that fraction from
    each end by count, so it removes outliers up to that fraction however far away they are. Sigma's
    breakdown point is effectively zero, which is the whole difference.
    """
    core = [(0.0 + 0.01 * i, 0.0, 0.0) for i in range(40)]       # tight cluster near the origin
    blob = [(3.0 + 0.01 * i, 0.0, 0.0) for i in range(12)]        # 23% of the points, 3 m away
    label = PACK(5, 1)
    untrimmed = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.0), label)
    trimmed = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.25), label)
    if untrimmed is None or trimmed is None:
        return ["no row emitted"]
    problems = []
    if untrimmed[MXX] < 2.9:
        problems.append(f"untrimmed box should span the blob, got max_x {untrimmed[MXX]:.2f}")
    if trimmed[MXX] > 1.0:
        problems.append(f"a 25% trim did not expel a 23% blob: max_x {trimmed[MXX]:.2f}")
    if int(trimmed[CNT]) > len(core):
        problems.append(f"trimmed count {int(trimmed[CNT])} exceeds the {len(core)} core points")
    return problems


def case_trim_breaks_down_above_its_percentile():
    """The documented limit, asserted so it cannot be forgotten.

    An outlier population LARGER than trim_percentile survives, because the bound is defined by count.
    Rejecting an arbitrarily large second mode needs density or connected-component selection, which is
    the escalation named in the README -- not a bigger percentile, which would start eating the object.
    """
    core = [(0.0 + 0.01 * i, 0.0, 0.0) for i in range(40)]
    blob = [(3.0 + 0.01 * i, 0.0, 0.0) for i in range(12)]        # 23%
    label = PACK(7, 1)
    r = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.10), label)
    if r is None:
        return ["no row emitted"]
    if r[MXX] < 2.0:
        return [f"a 10% trim unexpectedly expelled a 23% blob (max_x {r[MXX]:.2f}); the breakdown "
                f"point is no longer trim_percentile and the documentation is wrong"]
    return []


def case_reported_sigma_is_the_untrimmed_one():
    """Sigma must stay the PRE-trim value: it is the evidence a mask covered two surfaces."""
    core = [(0.0, 0.0, 0.0)] * 40
    blob = [(3.0, 0.0, 0.0)] * 12
    label = PACK(6, 1)
    r = row_for(*run({label: core + blob}, min_points=1, trim_percentile=0.25), label)
    if r is None:
        return ["no row emitted"]
    # The trimmed points span ~0, but the untrimmed spread is >1 m -- that difference is the signal.
    if r[SGX] < 0.5:
        problems = [f"sigma_x {r[SGX]:.3f} looks trimmed; it must report the untrimmed spread"]
        return problems
    if (r[MXX] - r[MNX]) > 0.5:
        return [f"box was not trimmed: extent_x {r[MXX] - r[MNX]:.3f}"]
    return []


def case_min_points_drops_small_instances():
    small, big = PACK(4, 1), PACK(4, 2)
    rows, labels = run({small: [(0.0, 0.0, 0.0)] * 3,
                        big: [(5.0, 5.0, 5.0)] * 20}, min_points=10, trim_percentile=0.0)
    problems = []
    if row_for(rows, labels, small) is not None:
        problems.append("a 3-point instance survived min_points=10")
    if row_for(rows, labels, big) is None:
        problems.append("the 20-point instance was dropped by min_points=10")
    return problems


def case_multiple_instances_and_classes_are_separate():
    a, b, c = PACK(1, 1), PACK(1, 2), PACK(2, 1)
    pts = {a: [(0.0, 0.0, 0.0)] * 8, b: [(4.0, 0.0, 0.0)] * 8, c: [(0.0, 7.0, 0.0)] * 8}
    rows, labels = run(pts, min_points=1, trim_percentile=0.0)
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
    rows, labels = run({label: [(1.0, 1.0, 1.0)] * 8}, min_points=1, trim_percentile=0.0, camera_index=3)
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
    case_percentile_trim_survives_the_masking_effect,
    case_trim_breaks_down_above_its_percentile,
    case_reported_sigma_is_the_untrimmed_one,
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
