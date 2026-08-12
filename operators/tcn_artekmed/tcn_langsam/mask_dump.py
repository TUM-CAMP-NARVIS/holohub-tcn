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
from operators.tcn_artekmed.tcn_util.frame_identity import tensor_names
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
            log.info(
                "MaskDumpOp: no frame_source given -- index_manifest.tsv will not be written. "
                "Dump filenames are unaffected: they are always the arrival index (see compute)."
            )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("masks")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("masks")

        # Label by ARRIVAL INDEX, not by the replayer's current frame_index.
        #
        # The replayer is several frames ahead of the masks arriving here: the pipeline holds
        # ~2 frames in flight, so reading `frame_source.frame_index` at this moment labels
        # frame 0's masks as "frame 2". Measured 2026-08-10: a 6-frame run produced dumps
        # named frame000002..frame000005 -- an offset, not a lost tail.
        #
        # That is not merely cosmetic. The offset equals the pipeline latency, which CHANGES
        # with configuration (sam_batched_decode and FP16 engines both alter timing). Two runs
        # could each emit four files named 2..5 holding DIFFERENT source frames, and
        # compare_mask_dumps.py would then compare mismatched content and report a confident
        # pass or fail. Labelling by arrival index is safe instead: the collector emits one
        # message per tick in source order, so index i is the i-th completed frame in BOTH
        # runs, and if the two runs complete different numbers of frames the compare script's
        # structural check fails loudly (exit 2) instead of comparing the wrong pairs.
        #
        # `frame_source.frame_index` is still recorded, in index_manifest.tsv, so a dump can be
        # traced back to its source frame for diagnosis without being load-bearing for the gate.
        frame_number = self._tick
        self._tick += 1
        if self.frame_source is not None:
            src = self.frame_source.frame_index
            with open(os.path.join(self.out_dir, "index_manifest.tsv"), "a") as mf:
                if frame_number == 0:
                    mf.write("arrival_index\treplayer_frame_index\tloop_count\n")
                mf.write(f"{frame_number}\t{src}\t"
                         f"{getattr(self.frame_source, 'loop_count', '')}\n")

        for name in tensor_names(msg):
            camera_id = name[: -len(_MASK_SUFFIX)] if name.endswith(_MASK_SUFFIX) else name
            tensor = msg.get(name)
            arr = cp.asnumpy(cp.asarray(tensor))  # preserve dtype exactly: packed uint16 ids
            out_path = os.path.join(self.out_dir, f"frame{frame_number:06d}_{camera_id}.npy")
            np.save(out_path, arr)
