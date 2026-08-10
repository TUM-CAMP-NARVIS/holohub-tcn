# SPDX-License-Identifier: Apache-2.0
"""MaskDumpOp: writes one `.npy` per camera per tick from the langsam_multicam mask
collector's output, for the deterministic replay harness. See
`docs/specs/2026-08-10-replay-harness-design.md` §2.2. Harness-specific, so it lives in the
app (this file), not in `operators/`.

Input shape matches exactly what `LabelMapColorizeOp` receives from `MaskCollectorOp`
(`langsam_multicam_fragment.py`): a dict of ``{f"{camera_id}_mask": <device uint16 (H, W)
panoptic map>}``, values packed ``(class_id << 8) | instance_id``.

Filenames are ``frame<NNNNNN>_<camera_id>.npy`` (six digits, zero-padded), matching
`docs/compare_mask_dumps.py`'s ``DUMP_RE`` exactly. The frame number is the replayer's TRUE
source dataset frame number (``frame_source.frame_index``) whenever one is available, NOT a
tick counter -- with `loop` on, a tick counter and the source frame number diverge, and the
compare script keys on the filename. `frame_source=None` (the `source: "shm"` case) falls
back to an internal tick counter deliberately, not as a degraded workaround: a live stream has
no dataset frame number to report in the first place, so a tick counter IS the correct label
there.
"""
import logging
import os

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("MaskDumpOp")

_MASK_SUFFIX = "_mask"


class MaskDumpOp(Operator):
    """Dumps per-camera panoptic mask maps to ``<out_dir>/frame<NNNNNN>_<camera_id>.npy``.

    Args:
        out_dir: directory to write into. Created if absent. Construction FAILS if it already
            exists and is non-empty -- mixing two runs' dumps into one directory would
            silently corrupt a comparison (design doc §2.2).
        frame_source: an object exposing a read-only ``.frame_index`` attribute (e.g.
            ``TcnDatasetReplayerOp``) giving the true source dataset frame number most
            recently emitted. Pass ``None`` to use an internal tick counter instead (the
            correct choice for ``source: "shm"``, where no dataset frame number exists).
    """

    def __init__(self, fragment, *args, out_dir, frame_source=None, **kwargs):
        self.out_dir = str(out_dir)
        self.frame_source = frame_source
        self._tick = 0

        if os.path.isdir(self.out_dir):
            if os.listdir(self.out_dir):
                raise ValueError(
                    f"MaskDumpOp: mask_dump_dir {self.out_dir!r} already exists and is "
                    f"non-empty. Refusing to mix two runs' dumps into one directory -- point "
                    f"mask_dump_dir at an empty or new directory."
                )
        elif os.path.exists(self.out_dir):
            raise ValueError(f"MaskDumpOp: mask_dump_dir {self.out_dir!r} exists and is not a directory")
        else:
            os.makedirs(self.out_dir, exist_ok=False)

        if self.frame_source is None:
            log.warning(
                'MaskDumpOp: no frame_source given -- dumping with an internal tick counter. '
                'This is correct for source: "shm" (no dataset frame number exists there). If '
                'this is source: "dataset", dump filenames will be WRONG whenever the '
                'replayer loops -- pass frame_source=<the replayer op> instead.'
            )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("masks")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("masks")

        if self.frame_source is not None:
            frame_number = self.frame_source.frame_index
            if frame_number is None:
                # Should not happen once the replayer has emitted at least once, but skip
                # rather than mislabel a dump with a bogus frame number.
                log.warning("MaskDumpOp: frame_source.frame_index is None; skipping this tick")
                return
        else:
            frame_number = self._tick
        self._tick += 1

        for name in sorted(msg.keys()):
            camera_id = name[: -len(_MASK_SUFFIX)] if name.endswith(_MASK_SUFFIX) else name
            tensor = msg.get(name)
            arr = cp.asnumpy(cp.asarray(tensor))  # preserve dtype exactly: packed uint16 ids
            out_path = os.path.join(self.out_dir, f"frame{frame_number:06d}_{camera_id}.npy")
            np.save(out_path, arr)
