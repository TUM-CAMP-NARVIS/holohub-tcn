import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("DepthImageMaxDistanceOp")


class FlattenTensorOp(Operator):

    def __init__(
            self,
            fragment: Any,
            *args,
            **kwargs,
    ):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")
        spec.param("message_name")


    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        port_stream = op_input.receive_cuda_stream("input", True)

        input_buffer = message.get(self.message_name)
        if input_buffer is not None:
            input_buffer = cp.asarray(input_buffer)
            with cp.cuda.ExternalStream(port_stream):
                output_shape = list(input_buffer.shape[:])
                output_shape[1] = output_shape[0]*output_shape[1]
                output_shape[0] = 1
                output_buffer = input_buffer.reshape(tuple(output_shape))
            di_tensor = hs.as_tensor(output_buffer)
            op_output.emit({self.message_name: di_tensor}, "output")
        else:
            log.warning(f"FlattenTensorOp: received empty tensor for: {self.message_name}")

