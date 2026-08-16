"""Correctness gate for TcnInstanceStatsOp. Runs the real operator on synthetic point grids.

    PYTHONPATH=<build>/python/lib python3 tests/test_instance_stats.py

Every expectation is a hand-computed value or a numpy reduction over the same input, never a restating
of the kernel.
"""
import logging
import math
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
YAW, OU, OV, OUP, OCX, OCY, OCZ = range(14, 21)
N_COLUMNS = 21

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
        rows = np.asarray(msg.get("rows")).reshape(-1, N_COLUMNS)
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
    if rows.shape != (1, N_COLUMNS) or int(rows[0][CNT]) != 0:
        return [f"expected one all-zero row, got shape {rows.shape} count "
                f"{int(rows[0][CNT]) if rows.size else 'n/a'}"]
    return []


# ── yaw-oriented box ──────────────────────────────────────────────────────────────────────────────

def plate(yaw_deg, half_u, half_v, nu=8, nv=3, heights=(0.0, 1.0)):
    """A rectangular plate lying in the ground plane (up = y), rotated `yaw_deg` about the vertical.

    Sampled as a grid so the extreme samples sit exactly on the rectangle's edges: the oriented extents
    are then hand-computable as `2*half_u` by `2*half_v`, independent of the kernel.
    """
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for i in range(nu):
        du = -half_u + 2 * half_u * i / (nu - 1)
        for j in range(nv):
            dv = -half_v + 2 * half_v * j / (nv - 1)
            for h in heights:
                pts.append((c * du - s * dv, h, s * du + c * dv))   # x = u, z = v, y = up
    return pts


def case_yaw_recovers_a_rotated_plate():
    """The point of oriented boxes: a plate at 30 degrees must report 30 degrees and its TRUE extents,
    while its axis-aligned box is inflated by the rotation."""
    half_u, half_v, yaw_deg = 0.6, 0.1, 30.0
    label = PACK(1, 1)
    r = row_for(*run({label: plate(yaw_deg, half_u, half_v)},
                     min_points=1, trim_percentile=0.0, up_axis=1), label)
    if r is None:
        return ["no row emitted"]
    problems = []
    got = math.degrees(r[YAW])
    if abs(got - yaw_deg) > 2.0:
        problems.append(f"yaw {got:.1f} deg != {yaw_deg} deg")
    if abs(r[OU] - 2 * half_u) > 0.02:
        problems.append(f"oriented extent along yaw {r[OU]:.3f} != {2 * half_u}")
    if abs(r[OV] - 2 * half_v) > 0.02:
        problems.append(f"oriented extent across yaw {r[OV]:.3f} != {2 * half_v}")
    # The AABB is inflated by the rotation: 1.2*cos30 + 0.2*sin30 = 1.14 m across x, and the oriented
    # footprint (0.24 m^2) is far tighter than the axis-aligned one (0.88 m^2).
    aabb_area = (r[MXX] - r[MNX]) * (r[MXZ] - r[MNZ])
    if r[OU] * r[OV] > 0.5 * aabb_area:
        problems.append(f"oriented footprint {r[OU] * r[OV]:.3f} m2 is not tighter than the "
                        f"axis-aligned {aabb_area:.3f} m2 -- the rotation is not being used")
    if abs(r[OUP] - 1.0) > 1e-4:
        problems.append(f"vertical extent {r[OUP]:.3f} != 1.0; yaw must not touch the up axis")
    return problems


def case_oriented_centre_sits_at_the_plate_centre():
    """The oriented centre is a rotated-frame midpoint, so a sign error in the inverse rotation moves it
    off the object -- invisible in the extents, obvious here."""
    label = PACK(2, 1)
    pts = [(x + 2.0, y, z - 1.0) for (x, y, z) in plate(40.0, 0.5, 0.15)]
    r = row_for(*run({label: pts}, min_points=1, trim_percentile=0.0, up_axis=1), label)
    if r is None:
        return ["no row emitted"]
    want = np.mean(pts, axis=0)          # a symmetric grid: the centroid IS the box centre
    got = r[[OCX, OCY, OCZ]]
    if not np.allclose(got, want, atol=2e-3):
        return [f"oriented centre {got} != plate centre {want}"]
    return []


def case_isotropic_footprint_reports_no_yaw():
    """A square footprint has no orientation to find. Fitting one to noise makes the box spin frame to
    frame, so the anisotropy guard must report yaw 0 and fall back to the axis-aligned extents."""
    label = PACK(3, 1)
    r = row_for(*run({label: plate(25.0, 0.3, 0.3, nu=5, nv=5, heights=(0.0,))},
                     min_points=1, trim_percentile=0.0, up_axis=1, min_anisotropy=1.5), label)
    if r is None:
        return ["no row emitted"]
    problems = []
    if abs(r[YAW]) > 1e-6:
        problems.append(f"yaw {math.degrees(r[YAW]):.2f} deg on a square footprint; the "
                        f"min_anisotropy guard is not holding")
    if abs(r[OU] - (r[MXX] - r[MNX])) > 1e-4 or abs(r[OV] - (r[MXZ] - r[MNZ])) > 1e-4:
        problems.append(f"with yaw 0 the oriented extents {r[[OU, OV]]} must equal the axis-aligned "
                        f"{[r[MXX] - r[MNX], r[MXZ] - r[MNZ]]}")
    return problems


def case_min_anisotropy_zero_always_orients():
    """The guard is a threshold, not a hard-coded rule: at 0 even a square gets an orientation. Proves
    the previous case tests the guard rather than a kernel that never rotates anything."""
    label = PACK(4, 1)
    r = row_for(*run({label: plate(25.0, 0.3, 0.3, nu=5, nv=5, heights=(0.0,))},
                     min_points=1, trim_percentile=0.0, up_axis=1, min_anisotropy=0.0), label)
    if r is None:
        return ["no row emitted"]
    # A square's principal axis is degenerate, so the ANGLE is arbitrary -- only that one was chosen
    # is meaningful, and that the box still contains the points.
    if r[OU] * r[OV] < (0.6 * 0.6) - 1e-3:
        return [f"oriented footprint {r[OU] * r[OV]:.4f} m2 is smaller than the 0.36 m2 square it "
                f"must contain"]
    return []


def case_oriented_box_is_never_larger_than_the_aabb():
    """An oriented box refines the axis-aligned one, so its volume can only be smaller or equal. A
    violation means the two boxes are describing different point sets."""
    problems = []
    for i, yaw_deg in enumerate((0.0, 15.0, 45.0, 70.0, -35.0)):
        label = PACK(5, i + 1)
        r = row_for(*run({label: plate(yaw_deg, 0.55, 0.12)},
                         min_points=1, trim_percentile=0.0, up_axis=1), label)
        if r is None:
            problems.append(f"no row emitted at yaw {yaw_deg}")
            continue
        aabb = ((r[MXX] - r[MNX]) * (r[MXY] - r[MNY]) * (r[MXZ] - r[MNZ]))
        oriented = r[OU] * r[OV] * r[OUP]
        if oriented > aabb + 1e-4:
            problems.append(f"at yaw {yaw_deg} deg the oriented volume {oriented:.4f} exceeds the "
                            f"axis-aligned {aabb:.4f}")
    return problems


def case_up_axis_selects_the_vertical():
    """up_axis is configuration, not a convention: with up = z the same plate must be oriented in the
    x-y plane instead, and reporting the z extent as vertical."""
    label = PACK(6, 1)
    # a plate in the x-y plane at 30 deg, 1.0 m tall along z
    pts = [(x, z, y) for (x, y, z) in plate(30.0, 0.6, 0.1)]
    r = row_for(*run({label: pts}, min_points=1, trim_percentile=0.0, up_axis=2), label)
    if r is None:
        return ["no row emitted"]
    problems = []
    if abs(math.degrees(r[YAW]) - 30.0) > 2.0:
        problems.append(f"yaw {math.degrees(r[YAW]):.1f} deg != 30 deg with up_axis=2")
    if abs(r[OUP] - 1.0) > 1e-4:
        problems.append(f"vertical extent {r[OUP]:.3f} != 1.0; up_axis=2 was not honoured")
    return problems


def case_component_filter_removes_a_detached_bleed_blob():
    """The case percentiles and sigma both lose: an outlier population LARGER than any breakdown
    point, but not attached to the object.

    24 object points and 16 blob points -- the blob is 40% of the instance, far above both
    trim_percentile and anything sigma clipping could survive. It is spatially detached, so
    connectivity removes it regardless of how many there are.
    """
    label = PACK(1, 1)
    # Object: a contiguous 3x8 patch of pixels on a plane at z = 0.
    obj = [(x * 0.02, y * 0.02, 0.0) for y in range(3) for x in range(8)]
    # Blob: a contiguous 2x8 patch 2 m behind, laid out in later rows so it is a separate island
    # in the grid AND separated in depth. A whole grid row of background lies between them.
    pad = [(0.0, 0.0, 0.0)] * 8                       # one row of label-0 padding
    blob = [(x * 0.02, y * 0.02, 2.0) for y in range(2) for x in range(8)]

    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    for i, pt in enumerate(obj):
        positions[i], labels[i] = pt, label
    base = len(obj) + len(pad)
    for j, pt in enumerate(blob):
        positions[base + j], labels[base + j] = pt, label

    out_off, out_on = {}, {}
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), out_off,
            min_points=1, trim_percentile=0.0, component_filter=False).run()
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), out_on,
            min_points=1, trim_percentile=0.0,
            component_filter=True, component_max_gap_m=0.05,
            component_min_fraction=1.0).run()

    problems = []
    r_off = row_for(out_off["rows"], out_off["labels"], label)
    r_on = row_for(out_on["rows"], out_on["labels"], label)
    if r_off is None or r_on is None:
        return [f"missing row: off={r_off is not None} on={r_on is not None}"]
    if int(r_off[CNT]) != len(obj) + len(blob):
        problems.append(f"precondition: filter OFF kept {int(r_off[CNT])}, expected "
                        f"{len(obj) + len(blob)} -- the blob was already being removed")
    if r_off[MXZ] < 1.5:
        problems.append(f"precondition: filter OFF box max z {r_off[MXZ]:.3f} does not reach the "
                        f"blob at z=2.0, so this case is not testing what it claims")
    if int(r_on[CNT]) != len(obj):
        problems.append(f"filter ON kept {int(r_on[CNT])} points, expected exactly the {len(obj)} "
                        f"object points")
    if abs(r_on[MXZ]) > 1e-4:
        problems.append(f"filter ON box still reaches z={r_on[MXZ]:.3f}; the blob survived")
    return problems


def case_component_filter_keeps_a_contiguous_object_whole():
    """It must not erode a legitimate object: one connected surface stays entirely intact."""
    label = PACK(2, 3)
    pts = [(x * 0.02, y * 0.02, 0.0) for y in range(5) for x in range(8)]
    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    for i, pt in enumerate(pts):
        positions[i], labels[i] = pt, label
    out = {}
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), out,
            min_points=1, trim_percentile=0.0,
            component_filter=True, component_max_gap_m=0.05,
            component_min_fraction=1.0).run()
    r = row_for(out["rows"], out["labels"], label)
    if r is None:
        return ["the whole object was filtered away"]
    if int(r[CNT]) != len(pts):
        return [f"kept {int(r[CNT])} of {len(pts)} points; a contiguous surface was split"]
    return []


def case_component_filter_gap_is_a_surface_test_not_a_mask_test():
    """With a gap large enough to bridge the depth step, the blob must come BACK -- otherwise the
    filter is separating on grid adjacency alone and the 3D predicate is dead code."""
    label = PACK(3, 2)
    obj = [(x * 0.02, y * 0.02, 0.0) for y in range(3) for x in range(8)]
    blob = [(x * 0.02, y * 0.02, 0.10) for y in range(2) for x in range(8)]
    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    for i, pt in enumerate(obj):
        positions[i], labels[i] = pt, label
    for j, pt in enumerate(blob):
        positions[len(obj) + j], labels[len(obj) + j] = pt, label   # grid-adjacent, 0.10 m behind

    tight, loose = {}, {}
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), tight,
            min_points=1, trim_percentile=0.0,
            component_filter=True, component_max_gap_m=0.05,
            component_min_fraction=1.0).run()
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), loose,
            min_points=1, trim_percentile=0.0,
            component_filter=True, component_max_gap_m=0.50,
            component_min_fraction=1.0).run()
    r_tight = row_for(tight["rows"], tight["labels"], label)
    r_loose = row_for(loose["rows"], loose["labels"], label)
    problems = []
    if r_tight is None or int(r_tight[CNT]) != len(obj):
        got = "none" if r_tight is None else int(r_tight[CNT])
        problems.append(f"gap 0.05 kept {got}, expected {len(obj)} (the 0.10 m step should split)")
    if r_loose is None or int(r_loose[CNT]) != len(obj) + len(blob):
        got = "none" if r_loose is None else int(r_loose[CNT])
        problems.append(f"gap 0.50 kept {got}, expected {len(obj) + len(blob)} -- the 3D gap "
                        f"predicate is not being used, only grid adjacency")
    return problems


def case_component_filter_off_by_default():
    """Default must be the exact pre-existing behaviour."""
    label = PACK(4, 1)
    obj = [(x * 0.02, 0.0, 0.0) for x in range(8)]
    blob = [(x * 0.02, 0.0, 2.0) for x in range(8)]
    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    for i, pt in enumerate(obj + blob):
        positions[i], labels[i] = pt, label
    out = {}
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), out,
            min_points=1, trim_percentile=0.0).run()
    r = row_for(out["rows"], out["labels"], label)
    if r is None or int(r[CNT]) != len(obj) + len(blob):
        got = "none" if r is None else int(r[CNT])
        return [f"default kept {got}, expected all {len(obj) + len(blob)}: the filter is on by "
                f"default, which silently changes every existing pipeline"]
    return []


def case_component_min_fraction_keeps_a_secondary_island():
    """The guard against the trap that `min_fraction: 1.0` is.

    A mask has HOLES -- occlusion and invalid-depth pixels are label 0 -- and a hole breaks grid
    adjacency, so a real object routinely arrives as several islands. Keeping strictly the largest
    then deletes most of it. Measured on a 4-camera capture, 1.0 destroyed the computer, the monitor
    and 2 of 5 chairs; 0.1 lost nothing.

    Here: one object split into 24 + 16 points by a row of label-0 holes, both at the SAME depth, so
    they are one surface separated only by the hole. 1.0 keeps 24; a fraction below 16/24 keeps all
    40.
    """
    label = PACK(5, 1)
    big = [(x * 0.02, y * 0.02, 0.0) for y in range(3) for x in range(8)]     # 24
    small = [(x * 0.02, (y + 4) * 0.02, 0.0) for y in range(2) for x in range(8)]  # 16
    positions = np.zeros((H * W, 3), np.float32)
    labels = np.zeros(H * W, np.uint16)
    for i, pt in enumerate(big):
        positions[i], labels[i] = pt, label
    base = len(big) + 8                                   # one row of label-0 holes between them
    for j, pt in enumerate(small):
        positions[base + j], labels[base + j] = pt, label

    strict, lenient = {}, {}
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), strict,
            min_points=1, trim_percentile=0.0, component_filter=True,
            component_max_gap_m=0.05, component_min_fraction=1.0).run()
    Harness(positions.reshape(H, W, 3), labels.reshape(H, W), lenient,
            min_points=1, trim_percentile=0.0, component_filter=True,
            component_max_gap_m=0.05, component_min_fraction=0.5).run()
    r_s = row_for(strict["rows"], strict["labels"], label)
    r_l = row_for(lenient["rows"], lenient["labels"], label)
    problems = []
    if r_s is None or int(r_s[CNT]) != len(big):
        got = "none" if r_s is None else int(r_s[CNT])
        problems.append(f"fraction 1.0 kept {got}, expected only the largest island ({len(big)})")
    if r_l is None or int(r_l[CNT]) != len(big) + len(small):
        got = "none" if r_l is None else int(r_l[CNT])
        problems.append(f"fraction 0.5 kept {got}, expected both islands "
                        f"({len(big) + len(small)}); 16/24 = 0.67 is above the threshold")
    return problems


def case_yaw_is_computed_on_the_trimmed_points():
    """Orientation must come from the surviving points, not the raw ones: a far outlier blob would
    otherwise drag the principal axis onto itself and rotate the box away from the object."""
    label = PACK(7, 1)
    pts = plate(0.0, 0.6, 0.1, nu=8, nv=3, heights=(0.0,))       # long axis exactly along x
    blob = [(2.0, 0.0, 2.0)] * 3                                  # 11% of the points, off-diagonal
    r = row_for(*run({label: pts + blob}, min_points=1, trim_percentile=0.15, up_axis=1), label)
    if r is None:
        return ["no row emitted"]
    if abs(math.degrees(r[YAW])) > 5.0:
        return [f"yaw {math.degrees(r[YAW]):.1f} deg: the outlier blob is still steering the "
                f"orientation, so the yaw pass is not reading the trimmed point set"]
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
    case_yaw_recovers_a_rotated_plate,
    case_oriented_centre_sits_at_the_plate_centre,
    case_isotropic_footprint_reports_no_yaw,
    case_min_anisotropy_zero_always_orients,
    case_oriented_box_is_never_larger_than_the_aabb,
    case_up_axis_selects_the_vertical,
    case_yaw_is_computed_on_the_trimmed_points,
    case_component_filter_removes_a_detached_bleed_blob,
    case_component_filter_keeps_a_contiguous_object_whole,
    case_component_filter_gap_is_a_surface_test_not_a_mask_test,
    case_component_filter_off_by_default,
    case_component_min_fraction_keeps_a_secondary_island,
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
