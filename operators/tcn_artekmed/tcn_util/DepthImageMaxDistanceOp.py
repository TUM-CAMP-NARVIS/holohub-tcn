import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("DepthImageMaxDistanceOp")


class DepthImageMaxDistanceOp(Operator):

    def __init__(
            self,
            fragment: Any,
            *args,
            **kwargs,
    ):
        self.max_buffer_ = None
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")


    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        port_stream = op_input.receive_cuda_stream("input", True)

        input_buffer = message.get("")
        if input_buffer is not None:
            input_buffer = cp.asarray(input_buffer)
            with cp.cuda.ExternalStream(port_stream):
                if self.max_buffer_ is None:
                    self.max_buffer_ = cp.zeros(input_buffer.shape, dtype=input_buffer.dtype)
                self.max_buffer_ = cp.maximum(cp.asarray(input_buffer), self.max_buffer_)
            # what happens if a reference to this buffer is is used while the operator is executed again?
            di_tensor = hs.as_tensor(self.max_buffer_)
            op_output.emit({"": di_tensor}, "output")

