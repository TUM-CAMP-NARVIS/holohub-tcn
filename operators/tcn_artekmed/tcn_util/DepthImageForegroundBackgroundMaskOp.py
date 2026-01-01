import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec, ConditionType
log = logging.getLogger("DepthImageForegroundBackgroundMaskOp")

# Define the ElementwiseKernel for the BGFG mask logic
# This performs all operations in a single CUDA kernel pass
_bgfg_kernel = cp.ElementwiseKernel(
    in_params='T depth, T bg, float32 sensitivity',
    out_params='uint8 fg_out, uint8 bg_out',
    operation='''
        // depth is in mm, convert to meters-ish scale for error calc
        T err = (T)(((depth / 1000.0f) + 11.0f) * sensitivity);
        T fg_val = depth + err;
        T bg_val = bg - err;
        
        fg_out = (bg_val > fg_val) ? 1 : 0;
        bg_out = (bg_val <= fg_val) ? 1 : 0;
    ''',
    name='bgfg_mask_kernel'
)

class DepthImageForegroundBackgroundMaskOp(Operator):
    def __init__(
            self,
            fragment: Any,
            *args,
            enable_foreground: bool = True,
            enable_background: bool = False,
            **kwargs,
    ):
        self.enable_foreground = enable_foreground
        self.enable_background = enable_background
        
        # Pre-allocated memory for output tensors
        self._fg_out = None
        self._bg_out = None
        self._buffer_shape = None
        
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("depth_image")
        spec.input("background_image")

        fgm = spec.output("foreground_mask")
        if not self.enable_foreground:
            fgm.condition(ConditionType.NONE)
            
        bgm = spec.output("background_mask")
        if not self.enable_background:
            bgm.condition(ConditionType.NONE)

        spec.param("sensitivity", 1.0)

    def compute(self, op_input, op_output, context):
        depth_message = op_input.receive("depth_image")
        bg_message = op_input.receive("background_image")

        depth_buffer = cp.asarray(depth_message.get(""))
        bg_buffer = cp.asarray(bg_message.get(""))

        if depth_buffer is None or bg_buffer is None:
            return

        # Synchronize streams
        depth_stream = op_input.receive_cuda_stream("depth_image", False)
        bg_stream = op_input.receive_cuda_stream("background_image", False)
        out_stream = context.allocate_cuda_stream(f"{self.name}_fgbg_mask_stream")
        context.synchronize_streams([depth_stream, bg_stream], out_stream)

        # Lazy allocation of output buffers based on input shape
        if self._buffer_shape != depth_buffer.shape:
            self._buffer_shape = depth_buffer.shape
            self._fg_out = cp.zeros(self._buffer_shape, dtype=cp.uint8)
            self._bg_out = cp.zeros(self._buffer_shape, dtype=cp.uint8)

        with cp.cuda.ExternalStream(out_stream):
            # Execute the pre-compiled CUDA kernel
            _bgfg_kernel(
                depth_buffer, 
                bg_buffer, 
                self.sensitivity, 
                self._fg_out, 
                self._bg_out
            )

            if self.enable_foreground:
                # Use a view to emit so we don't copy, but be aware of downstream lifetime
                fg_tensor = hs.as_tensor(self._fg_out)
                op_output.set_cuda_stream(out_stream, "foreground_mask")
                op_output.emit({"": fg_tensor}, "foreground_mask")

            if self.enable_background:
                bg_tensor = hs.as_tensor(self._bg_out)
                op_output.set_cuda_stream(out_stream, "background_mask")
                op_output.emit({"": bg_tensor}, "background_mask")

