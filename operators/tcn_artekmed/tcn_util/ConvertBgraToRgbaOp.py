import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("DepthImageApplyMaskOp")


class ConvertBgraToRgbaOp(Operator):
    """Convert BGRA -> RGBA efficiently on the GPU (CuPy)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        tensor = msg.get("")
        if tensor is None:
            return

        bgra = cp.asarray(tensor)

        # Expect HxWx4; if your stream can differ, you may want stricter checks/logging here.
        if bgra.ndim != 3 or bgra.shape[-1] < 4:
            return

        # Allocate once and permute channels via assignments (avoids extra temporaries).
        rgba = cp.empty((bgra.shape[0], bgra.shape[1], 4), dtype=bgra.dtype)
        rgba[..., 0] = bgra[..., 2]  # R <- B
        rgba[..., 1] = bgra[..., 1]  # G <- G
        rgba[..., 2] = bgra[..., 0]  # B <- R
        rgba[..., 3] = bgra[..., 3]  # A <- A

        op_output.emit({"": hs.as_tensor(rgba)}, "output")