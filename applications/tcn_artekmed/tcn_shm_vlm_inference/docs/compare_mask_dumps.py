#!/usr/bin/env python3
"""Compare two directories of panoptic-mask dumps and return a pass/fail gate.

This is the correctness gate for every optimisation that must not change output. It exists
because the live SHM source is not reproducible, so "did this change the masks?" could
previously only be answered by looking at the screen. See
`specs/2026-08-10-replay-harness-design.md` §2.3.

Dumps are written by the harness as `frame<NNNNNN>_<camera>.npy`, one array per camera per
replayed frame. Values are PACKED panoptic ids: `(class_id << 8) | instance_id`, so equality
is exact and needs no tolerance.

Two modes, matching the two kinds of gate we actually have:

  exact (default)   Any differing pixel fails. Use for changes that must be bit-identical:
                    CUDA graph capture, pure refactors, and the harness's own
                    self-consistency check (two identical runs MUST agree, or the harness
                    cannot gate anything).

  --iou-gate T      Per-class IoU must be >= T everywhere. Use for changes that legitimately
                    perturb floating-point association and can therefore flip pixels at a
                    mask boundary: batched SAM decode (4a), and FP16 engine rebuilds (A1).

Exit code is 0 on pass, 1 on gate failure, 2 on a structural problem (mismatched frame sets,
shapes, or an unreadable dump) -- a structural problem is never a pass.

Usage:
    python3 compare_mask_dumps.py DIR_A DIR_B
    python3 compare_mask_dumps.py DIR_A DIR_B --iou-gate 0.999
    python3 compare_mask_dumps.py DIR_A DIR_B --iou-gate 0.999 --top 10
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

DUMP_RE = re.compile(r"^frame(?P<frame>\d+)_(?P<camera>.+)\.npy$")

CLASS_SHIFT = 8
INSTANCE_MASK = (1 << CLASS_SHIFT) - 1


def die_structural(msg):
    """Exit 2 = structural problem, distinct from exit 1 = gate failure.

    Note `sys.exit("string")` exits with code 1, which would make a missing dump directory
    indistinguishable from a legitimate mask difference. Always go through here.
    """
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def index_dumps(d: Path):
    """Map (frame_number, camera) -> path for every dump in `d`.

    Exits 2 if the directory is missing or holds no recognisable dumps -- an empty comparison
    trivially "passes" every gate, which is the most dangerous possible outcome for a gate
    script.
    """
    if not d.is_dir():
        die_structural(f"not a directory: {d}")
    out = {}
    skipped = []
    for p in sorted(d.iterdir()):
        if p.suffix != ".npy":
            continue
        m = DUMP_RE.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        out[(int(m.group("frame")), m.group("camera"))] = p
    if skipped:
        print(f"  note: ignored {len(skipped)} non-matching .npy file(s) in {d}, "
              f"e.g. {skipped[:3]}")
    if not out:
        die_structural(f"no 'frame<NNNNNN>_<camera>.npy' dumps found in {d}. "
                       f"Refusing to report a pass on an empty comparison.")
    return out


def split_packed(a):
    """Packed panoptic ids -> (class_ids, instance_ids)."""
    return (a >> CLASS_SHIFT).astype(np.int64), (a & INSTANCE_MASK).astype(np.int64)


def per_class_iou(a, b):
    """IoU per class id present in either map, as {class_id: iou}.

    A class present in one map and absent from the other gets IoU 0.0 -- that is a real
    disagreement (a whole object appeared or vanished) and must not be silently skipped.
    """
    ca, _ = split_packed(a)
    cb, _ = split_packed(b)
    ious = {}
    for cls in sorted(set(np.unique(ca).tolist()) | set(np.unique(cb).tolist())):
        if cls == 0:                      # background is not an object
            continue
        ma, mb = (ca == cls), (cb == cls)
        union = np.count_nonzero(ma | mb)
        ious[int(cls)] = 1.0 if union == 0 else np.count_nonzero(ma & mb) / union
    return ious


def instance_counts(a):
    """{class_id: number of distinct non-zero instance ids} for one map."""
    cls, inst = split_packed(a)
    out = {}
    for c in np.unique(cls):
        if c == 0:
            continue
        out[int(c)] = int(len(np.unique(inst[cls == c])))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compare two directories of panoptic-mask dumps (pass/fail gate).")
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--iou-gate", type=float, default=None, metavar="T",
                    help="pass if per-class IoU >= T everywhere (e.g. 0.999). "
                         "Omit for an exact-equality gate.")
    ap.add_argument("--top", type=int, default=5,
                    help="how many worst offenders to list (default 5)")
    args = ap.parse_args()

    A, B = index_dumps(args.dir_a), index_dumps(args.dir_b)

    # Structural checks first. Comparing only the intersection would let a run that emitted
    # half the frames pass.
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        detail = []
        if only_a:
            detail.append(f"only in {args.dir_a}: {len(only_a)}, e.g. {only_a[:5]}")
        if only_b:
            detail.append(f"only in {args.dir_b}: {len(only_b)}, e.g. {only_b[:5]}")
        die_structural("dump sets differ. " + "; ".join(detail))

    keys = sorted(A)
    frames = sorted({f for f, _ in keys})
    cameras = sorted({c for _, c in keys})
    mode = "exact equality" if args.iou_gate is None else f"per-class IoU >= {args.iou_gate}"
    print(f"Comparing {len(keys)} dumps: {len(frames)} frame(s) x {len(cameras)} camera(s)")
    print(f"  A: {args.dir_a}\n  B: {args.dir_b}\n  gate: {mode}\n")

    diffs = []            # (frac_differing, key, n_diff, size)
    worst_iou = []        # (iou, key, class_id)
    count_mismatches = []
    total_diff = 0
    total_px = 0

    for key in keys:
        a, b = np.load(A[key]), np.load(B[key])
        if a.shape != b.shape:
            die_structural(f"shape mismatch at frame{key[0]:06d}_{key[1]}: "
                           f"{a.shape} vs {b.shape}")
        n_diff = int(np.count_nonzero(a != b))
        total_diff += n_diff
        total_px += a.size
        if n_diff:
            diffs.append((n_diff / a.size, key, n_diff, a.size))
        ka, kb = instance_counts(a), instance_counts(b)
        if ka != kb:
            count_mismatches.append((key, ka, kb))
        if args.iou_gate is not None:
            for cls, iou in per_class_iou(a, b).items():
                worst_iou.append((iou, key, cls))

    print(f"Pixels differing: {total_diff} / {total_px} "
          f"({100.0 * total_diff / max(total_px, 1):.6f}%)")
    print(f"Dumps with any difference: {len(diffs)} / {len(keys)}")

    if count_mismatches:
        print(f"\nPer-class instance-count mismatches: {len(count_mismatches)}")
        for key, ka, kb in count_mismatches[:args.top]:
            print(f"  frame{key[0]:06d}_{key[1]}: A={ka} B={kb}")
    else:
        print("Per-class instance counts: identical everywhere")

    if diffs:
        print(f"\nWorst {min(args.top, len(diffs))} by differing-pixel fraction:")
        for frac, key, n, size in sorted(diffs, reverse=True)[:args.top]:
            print(f"  frame{key[0]:06d}_{key[1]}: {n}/{size} ({100 * frac:.6f}%)")

    if args.iou_gate is not None and worst_iou:
        worst_iou.sort()
        print(f"\nLowest {min(args.top, len(worst_iou))} per-class IoU "
              f"(of {len(worst_iou)} class-instances compared):")
        for iou, key, cls in worst_iou[:args.top]:
            print(f"  frame{key[0]:06d}_{key[1]} class {cls}: IoU {iou:.6f}")

    # Verdict. The instance-count check binds in BOTH modes: an IoU gate alone would pass a
    # run that merged two instances of the same class into one.
    if args.iou_gate is None:
        ok = (total_diff == 0)
        why = "identical" if ok else f"{total_diff} pixel(s) differ"
    else:
        lowest = worst_iou[0][0] if worst_iou else 1.0
        ok = (lowest >= args.iou_gate) and not count_mismatches
        why = (f"min IoU {lowest:.6f} >= {args.iou_gate}" if lowest >= args.iou_gate
               else f"min IoU {lowest:.6f} < {args.iou_gate}")
        if count_mismatches:
            why += f"; {len(count_mismatches)} instance-count mismatch(es)"

    print(f"\n{'PASS' if ok else 'FAIL'}: {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
