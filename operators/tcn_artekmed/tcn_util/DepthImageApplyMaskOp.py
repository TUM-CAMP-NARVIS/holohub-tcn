import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("DepthImageApplyMaskOp")

# Elementwise kernel for efficient masking
# Logic: depth * (mask == expected_val)
_apply_mask_kernel = cp.ElementwiseKernel(
    in_params='T depth, U mask, bool invert',
    out_params='T out',
    operation='''
        if (invert) {
            // If invert is true, we keep pixels where mask is 0 (background)
            out = (mask == 0) ? depth : 0;
        } else {
            // If invert is false, we keep pixels where mask is 1 (foreground)
            out = (mask != 0) ? depth : 0;
        }
    ''',
    name='apply_mask_kernel'
)

class DepthImageApplyMaskOp(Operator):
    def __init__(
            self,
            fragment: Any,
            *args,
            **kwargs,
    ):
        self._out_buffer = None
        self._buffer_shape = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("depth_image")
        spec.input("mask_image")
        spec.output("output")

        spec.param("invert_mask", False)

    def compute(self, op_input, op_output, context):
        depth_message = op_input.receive("depth_image")
        mask_message = op_input.receive("mask_image")

        if not depth_message or not mask_message:
            return

        depth_buffer = cp.asarray(depth_message.get(""))
        mask_buffer = cp.asarray(mask_message.get(""))

        # Synchronize streams
        depth_stream = op_input.receive_cuda_stream("depth_image", False)
        mask_stream = op_input.receive_cuda_stream("mask_image", False)
        out_stream = context.allocate_cuda_stream(f"{self.name}_apply_mask_stream")
        context.synchronize_streams([depth_stream, mask_stream], out_stream)

        # Lazy pre-allocation
        if self._buffer_shape != depth_buffer.shape:
            self._buffer_shape = depth_buffer.shape
            self._out_buffer = cp.zeros(self._buffer_shape, dtype=depth_buffer.dtype)

        with cp.cuda.ExternalStream(out_stream):
            # Run the kernel
            _apply_mask_kernel(
                depth_buffer, 
                mask_buffer, 
                self.invert_mask, 
                self._out_buffer
            )

            # Emit the result
            out_tensor = hs.as_tensor(self._out_buffer)
            op_output.set_cuda_stream(out_stream, "output")
            op_output.emit({"": out_tensor}, "output")
