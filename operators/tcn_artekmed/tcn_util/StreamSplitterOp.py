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
        message = op_input.receive("receivers")
        for channel_name in self.channel_names:
            di_tensor = hs.as_tensor(cp.asarray(message.get(channel_name)))
            op_output.emit({"": di_tensor}, channel_name)
