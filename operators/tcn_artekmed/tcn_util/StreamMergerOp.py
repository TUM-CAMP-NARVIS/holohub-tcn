import logging
from typing import Any

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from operators.tcn_artekmed.tcn_shm_io import DeviceContextService
log = logging.getLogger("StreamMergerOp")


class StreamMergerOp(Operator):

    def __init__(
            self,
            fragment: Any,
            cuda_stream_pool: Any,
            input_port_names: Any,
            input_message_name: str,
            output_message_name: str,
            fuse_buffers: bool,
            *args,
            **kwargs,
    ):
        self.input_port_names = input_port_names
        self.fuse_buffers = fuse_buffers
        self.input_message_name = input_message_name
        self.output_message_name = output_message_name
        self.ctx_service = None

        # Need to call the base class constructor last
        super().__init__(fragment, cuda_stream_pool, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for name in self.input_port_names:
            spec.input(name)
        spec.output("output")
        self.ctx_service = self.service(DeviceContextService)


    def compute(self, op_input, op_output, context):
        all_messages = []
        input_streams = []
        for name in self.input_port_names:
            message = op_input.receive(name)
            port_stream = op_input.receive_cuda_stream(name, False)
            if port_stream:
                input_streams.append(port_stream)
            log.debug(f"Merge {name} message: {message.keys()} -> {self.input_message_name}")
            value = message.get(self.input_message_name)
            if value is None:
                raise ValueError(f"Invalid payload for message with keys: {list(message.keys())} for {self.input_message_name}")
            all_messages.append((name, cp.asarray(value)))

        output_stream = None
        if input_streams:
            output_stream = context.allocate_cuda_stream(self.name)
            context.synchronize_streams(input_streams, output_stream)
            op_output.set_cuda_stream(output_stream, "output")

        if self.fuse_buffers:
            with cp.cuda.ExternalStream(output_stream):
                fused_buffer = cp.concatenate((m[1] for m in all_messages), axis=1)
            log.debug("Fused buffer to {}".format(fused_buffer.shape))
            di_tensor = hs.as_tensor(fused_buffer)
            op_output.set_cuda_stream(output_stream, "output")
            op_output.emit({self.output_message_name: di_tensor}, "output")
        else:
            out_message = dict()
            with cp.cuda.ExternalStream(output_stream):
                for name, buffer in all_messages:
                    camera_name = self.ctx_service.get_camera_name_from_port_name(name)
                    message_name = f"{camera_name}_{self.output_message_name}"
                    di_tensor = hs.as_tensor(buffer)
                    out_message[message_name] = di_tensor
            op_output.set_cuda_stream(output_stream, "output")
            op_output.emit(out_message, "output")

