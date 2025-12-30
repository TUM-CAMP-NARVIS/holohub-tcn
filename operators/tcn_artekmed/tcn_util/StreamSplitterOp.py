import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

log = logging.getLogger("StreamSplitterOp")

class StreamSplitterOp(Operator):

    def __init__(
            self,
            fragment: Any,
            channel_names: Any,
            *args,
            **kwargs,
    ):
        self.channel_names = channel_names
        # Need to call the base class constructor last

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers")
        for name in self.channel_names:
            spec.output(name)


    def compute(self, op_input, op_output, context):
        # @clarify: what if there is a cuda stream on the upstream?
        message = op_input.receive("receivers")
        for channel_name in self.channel_names:
            stream_ptr = context.allocate_cuda_stream(f"{self.name}_{channel_name}")
            if stream_ptr is None:
                raise RuntimeError("Failed to allocate cuda stream from stream-pool.")
            di_tensor = hs.as_tensor(cp.asarray(message.get(channel_name)))
            op_output.emit({"": di_tensor}, channel_name)
            op_output.set_cuda_stream(stream_ptr, channel_name)
