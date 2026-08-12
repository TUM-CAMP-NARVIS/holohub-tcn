"""Correctness gate for TcnLabelSamplerOp. Runs the real operator on synthetic inputs.

Must run inside the container (needs holoscan, cupy and the built module):

    PYTHONPATH=<build>/python/lib python3 tests/test_label_sampler.py

Every expectation is computed independently of the kernel -- from the texcoord that was fed in and
the label image that was fed in -- never by restating the sampling arithmetic under test.
"""
import logging
import sys

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator

from holohub.tcn_label_sampler import TcnLabelSamplerOp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_label_sampler")

LABEL_H, LABEL_W = 4, 4
UNLABELED = 0

# A label image whose every pixel is distinct, so a sample landing one pixel off is visible rather
# than absorbed by a neighbour that happens to share a value. Class = value >> 8, so these span
# classes 1..4 with instances 1..4.
LABELS = np.array([[(c << 8) | (r + 1) for c in range(1, LABEL_W + 1)] for r in range(LABEL_H)],
                  dtype=np.uint16)


def uv_for_pixel(col, row):
    """The texcoord that addresses label pixel (col,row) under the operator's documented mapping.

    Written from the CONTRACT ([0,1] maps onto [0, N-1], nearest sample wins), not from the kernel.
    """
    return (col / (LABEL_W - 1), row / (LABEL_H - 1))


class SourceOp(Operator):
    def __init__(self, fragment, *args, uv, **kwargs):
        self.uv = np.asarray(uv, dtype=np.float32)      # (H, W, 2)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("labels")
        spec.output("texcoords")

    def compute(self, op_input, op_output, context):
        op_output.emit({"labels": hs.as_tensor(cp.asarray(LABELS))}, "labels")
        op_output.emit({"texcoords": hs.as_tensor(cp.asarray(self.uv))}, "texcoords")


class CheckOp(Operator):
    def __init__(self, fragment, *args, expect_labels, expect_mask, case, results, **kwargs):
        self.expect_labels = np.asarray(expect_labels, dtype=np.uint16)
        self.expect_mask = np.asarray(expect_mask, dtype=np.uint8)
        self.case = case
        self.results = results
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("labels_out")
        spec.input("mask_out")

    def compute(self, op_input, op_output, context):
        got_labels = cp.asnumpy(cp.asarray(op_input.receive("labels_out").get("out_labels")))
        got_mask = cp.asnumpy(cp.asarray(op_input.receive("mask_out").get("out_mask")))
        got_labels = got_labels.reshape(self.expect_labels.shape)
        got_mask = got_mask.reshape(self.expect_mask.shape)

        problems = []
        if not np.array_equal(got_labels, self.expect_labels):
            bad = np.argwhere(got_labels != self.expect_labels)
            problems.append(f"labels differ at {len(bad)} px, first {tuple(bad[0])}: "
                            f"got {got_labels[tuple(bad[0])]} want {self.expect_labels[tuple(bad[0])]}")
        if not np.array_equal(got_mask, self.expect_mask):
            bad = np.argwhere(got_mask != self.expect_mask)
            problems.append(f"mask differs at {len(bad)} px, first {tuple(bad[0])}: "
                            f"got {got_mask[tuple(bad[0])]} want {self.expect_mask[tuple(bad[0])]}")
        self.results.append((self.case, problems))


class Harness(Application):
    def __init__(self, uv, expect_labels, expect_mask, case, results,
                 select_classes=(), unlabeled_value=UNLABELED):
        self.uv = uv
        self.expect_labels = expect_labels
        self.expect_mask = expect_mask
        self.case = case
        self.results = results
        self.select_classes = list(select_classes)
        self.unlabeled_value = unlabeled_value
        super().__init__()

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")
        src = SourceOp(self, CountCondition(self, count=1), uv=self.uv, name="src")
        sampler = TcnLabelSamplerOp(
            self,
            allocator=pool,
            cuda_device_ordinal=0,
            in_labels_tensor_name="labels",
            in_texcoord_tensor_name="texcoords",
            out_labels_tensor_name="out_labels",
            out_mask_tensor_name="out_mask",
            select_classes=self.select_classes,
            unlabeled_value=self.unlabeled_value,
            name="sampler",
        )
        check = CheckOp(self, expect_labels=self.expect_labels, expect_mask=self.expect_mask,
                        case=self.case, results=self.results, name="check")
        self.add_flow(src, sampler, {("labels", "labels"), ("texcoords", "texcoords")})
        self.add_flow(sampler, check, {("labels_out", "labels_out"), ("mask_out", "mask_out")})


def run_case(case, uv, expect_labels, expect_mask, **kwargs):
    results = []
    Harness(np.asarray(uv, dtype=np.float32), expect_labels, expect_mask, case, results,
            **kwargs).run()
    if not results:
        return [f"{case}: the check operator never ran"]
    _, problems = results[0]
    return [f"{case}: {p}" for p in problems]


def case_exact_pixel_addressing():
    """Every depth pixel addresses a distinct label pixel; all must come back exactly."""
    uv = np.zeros((LABEL_H, LABEL_W, 2), dtype=np.float32)
    for r in range(LABEL_H):
        for c in range(LABEL_W):
            uv[r, c] = uv_for_pixel(c, r)
    return run_case("exact addressing", uv, LABELS, np.full(LABELS.shape, 255, np.uint8))


def case_nearest_not_bilinear():
    """A texcoord between two label pixels must return one of them, never a blend.

    The midpoint between (0,0) and (1,0) is the decisive test: bilinear would return the mean of
    two packed ids, which is a third id belonging to neither class.
    """
    a, b = int(LABELS[0, 0]), int(LABELS[0, 1])
    u_mid = (0 + 1 / (LABEL_W - 1)) / 2
    uv = np.zeros((1, 1, 2), dtype=np.float32)
    uv[0, 0] = (u_mid, 0.0)

    results = []
    Harness(uv, np.array([[a]], np.uint16), np.array([[255]], np.uint8),
            "nearest (midpoint)", results).run()
    got = results[0]
    # Either neighbour is acceptable at an exact midpoint; the blend is not.
    if not got[1]:
        return []
    # Re-check against the other neighbour before calling it a failure.
    results2 = []
    Harness(uv, np.array([[b]], np.uint16), np.array([[255]], np.uint8),
            "nearest (midpoint)", results2).run()
    if not results2[0][1]:
        return []
    return [f"nearest: midpoint returned neither {a} nor {b} -- the sampler is interpolating"]


def case_invalid_texcoords_are_unlabeled():
    """NaN (no valid depth) and out-of-frustum uv must both yield unlabeled + mask 0."""
    uv = np.array([[[np.nan, np.nan], [0.0, 0.0]],
                   [[1.5, 0.5], [0.5, -0.2]]], dtype=np.float32)
    expect_labels = np.array([[UNLABELED, int(LABELS[0, 0])],
                              [UNLABELED, UNLABELED]], dtype=np.uint16)
    expect_mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    return run_case("invalid texcoords", uv, expect_labels, expect_mask)


def case_select_classes_filters_only_the_mask():
    """select_classes changes mask_out and must leave labels_out untouched."""
    uv = np.zeros((1, LABEL_W, 2), dtype=np.float32)
    for c in range(LABEL_W):
        uv[0, c] = uv_for_pixel(c, 0)
    row = LABELS[0:1, :].copy()
    classes = (row >> 8).astype(np.int64)
    chosen = [int(classes[0, 0]), int(classes[0, 2])]
    expect_mask = np.where(np.isin(classes, chosen), 255, 0).astype(np.uint8)
    return run_case("select_classes", uv, row, expect_mask, select_classes=chosen)


def case_nonzero_unlabeled_value_never_masks():
    """A non-zero unlabeled_value that collides with a real class must still mask as 0.

    Guards the reason rejection is tracked separately from the sampled label: if the kernel decided
    'selected' from the label value alone, an unlabeled pixel carrying a class-1 value would be
    marked as a detection of class 1.
    """
    collide = int(LABELS[0, 0])                       # a genuine class-1 packed label
    uv = np.array([[[np.nan, np.nan], [0.0, 0.0]]], dtype=np.float32)
    expect_labels = np.array([[collide, int(LABELS[0, 0])]], dtype=np.uint16)
    expect_mask = np.array([[0, 255]], dtype=np.uint8)
    return run_case("nonzero unlabeled_value", uv, expect_labels, expect_mask,
                    unlabeled_value=collide)


CASES = [
    case_exact_pixel_addressing,
    case_nearest_not_bilinear,
    case_invalid_texcoords_are_unlabeled,
    case_select_classes_filters_only_the_mask,
    case_nonzero_unlabeled_value_never_masks,
]

if __name__ == "__main__":
    failures = []
    for fn in CASES:
        try:
            problems = fn()
        except Exception as e:                        # noqa: BLE001 - report, do not mask
            problems = [f"{fn.__name__}: raised {e!r}"]
        if problems:
            failures.extend(problems)
            print("FAIL", fn.__name__)
            for p in problems:
                print("     ", p)
        else:
            print("PASS", fn.__name__)
    print(f"{len(CASES) - len({f.split(':')[0] for f in failures})}/{len(CASES)} cases passed")
    sys.exit(1 if failures else 0)
