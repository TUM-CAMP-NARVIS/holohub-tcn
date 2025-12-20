#
# Place the license header here
#
import os
import logging
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import queue
from concurrent.futures import Future, ThreadPoolExecutor

import holoscan as hs
import numpy as np

from holoscan.gxf import Entity
from holoscan.logger import LogLevel, set_log_level
from holoscan.resources import CudaStreamPool, UnboundedAllocator, BlockMemoryPool
from holoscan.conditions import AsynchronousCondition, AsynchronousEventState, BooleanCondition, CountCondition, PeriodicCondition, MessageAvailableCondition, DownstreamMessageAffordableCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec, Tracker
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler
from holoscan.operators import HolovizOp

import iceoryx2 as iox2
from tcnart.core.semantic_type import SemanticType
from shm_receiver import ShmSynchronizedBufferReceiver
from shm_serde import shm_transport_enum

log = logging.getLogger(__name__)




class ShmSubscriberOp(Operator):
    """Simple zenoh subscriber.

    On each tick, it transmits a received message to the "out" port.

    **==Named Outputs==**

        out : bytes
        the received payload.
    """

    def __init__(
        self,
        fragment: Any,
        subscriber: Any,
        stream_name: str,
        channel_config: Any,
        cycle_time_ms: int,
        *args,
        **kwargs,
    ):
        self.subscriber = subscriber
        self.channel_config = channel_config
        self.stream_name = stream_name
        self.cycle_time_ms = cycle_time_ms
        self.pool = None

        self.executor_ = ThreadPoolExecutor(max_workers=1)
        self.future_ = None  # will be set during start()
        self.async_cond_ = AsynchronousCondition(fragment, name="async_cond")
        self.buffer = queue.Queue()

        # Need to call the base class constructor last
        super().__init__(fragment, self.async_cond_, *args, **kwargs)

    def on_receive(self, user_header: Any, message: Any):
        """Function to be supplied as callback

        When the condition's event_state is EVENT_WAITING, set to EVENT_DONE. This function will
        only exit once the condition is set to EVENT_NEVER.
        """
        if self.async_cond_.event_state == AsynchronousEventState.EVENT_NEVER:
            return False

        # log.debug(f"on_receive frame with timestamp {user_header.timestamp}")
        data = {}
        frame_timestamp = user_header.timestamp
        for port in message.ports:
            if port.data.portType == shm_transport_enum.CameraPortType.colorimage:
                # log.debug(f"added port data: {port.name}")
                md = port.data.metadata
                mv = memoryview(port.data.data)
                # for now we copy the payload from shm to a newly allocated numpy array
                # this could be improved by preallocating these arrays (double buffering)
                # and use np.copy_to(src, dst) to avoid repeated allocations
                data[port.name] = np.frombuffer(mv, dtype=np.uint8).reshape(
                    md.header.dimY, md.header.dimX, int(md.header.bitsPerElement/8)
                ).copy()
            elif port.data.portType == shm_transport_enum.CameraPortType.depthimage:
                # log.debug(f"added port data: {port.name}")
                md = port.data.metadata
                mv = memoryview(port.data.data)
                # for now we copy the payload from shm to a newly allocated numpy array
                # this could be improved by preallocating these arrays (double buffering)
                # and use np.copy_to(src, dst) to avoid repeated allocations
                data[port.name] = np.frombuffer(mv, dtype=np.uint16).reshape(
                    md.header.dimY, md.header.dimX, 1
                ).copy()

        if data:
            # how does ts relate to fragment.scheduler().clock.timestamp()?
            # log.debug(f"put data for {frame_timestamp} into queue")
            self.buffer.put((frame_timestamp, data))

            if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE
            return True

        return False

    def receiver_mainloop(self):
        while self.subscriber is not None:
            if not self.subscriber.receive_frame(self.on_receive, self.cycle_time_ms):
                log.warning("could not receive frame.")

    def setup(self, spec: OperatorSpec):
        spec.output("outputs")
        spec.output("output_specs")

    def start(self):
        self.subscriber.subscribe(self.stream_name)

        self.future_ = self.executor_.submit(self.receiver_mainloop)
        assert isinstance(self.future_, Future)

    def compute(self, op_input, op_output, context):
        scheduler = self.fragment.scheduler()
        clock = scheduler.clock
        ts = clock.timestamp()

        frame_ts, data = self.buffer.get()
        log.debug(f"got data for {frame_ts} from queue")
        message = {k:hs.as_tensor(v) for k,v in data.items()}

        self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING
        op_output.emit(message, "outputs", acq_timestamp=ts)

        num_videos = len(message.keys())

        # Determine grid size (e.g., 2 for 2x2, 3 for 3x3)
        grid_size = int(np.ceil(np.sqrt(num_videos))) if num_videos > 0 else 1
        tile_size = 1.0 / grid_size

        output_specs = []
        for i, port_name in enumerate(message.keys()):
            # Compute row and column index
            row = i // grid_size
            col = i % grid_size

            # Compute normalized offsets (0.0 to 1.0)
            # Note: Holoviz usually uses (x, y) for offsets
            offset_x = col * tile_size
            offset_y = row * tile_size

            spec = HolovizOp.InputSpec(port_name, HolovizOp.InputType.COLOR)
            views = []
            view = HolovizOp.InputSpec.View()
            view.offset_x = offset_x
            view.offset_y = offset_y
            view.width = tile_size
            view.height = tile_size
            views.append(view)
            spec.views = views
            output_specs.append(spec)

        op_output.emit(output_specs, "output_specs", acq_timestamp=ts)


    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        if self.subscriber is not None:
            self.subscriber = None
        self.future_.result()


class App(hs.core.Application):
    def compose(self):
        # Add your operators here
        print("Starting TCN Shm Receiver")

        shm_config = self.kwargs("shared_memory")

        stream_name = shm_config.get("stream_name")
        cycle_time_ms = shm_config.get("cycle_time_ms")

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        shm_receiver = ShmSynchronizedBufferReceiver(node)

        log.info("Find cameras in shared memory")
        camera_names = shm_receiver.discover_devices()
        device_contexts = {}
        for camera_name in camera_names:
            log.info(f"Retrieving camera_info: {camera_name}")
            ctx = shm_receiver.retrieve_device_context(camera_name)
            if ctx is not None:
                device_contexts[camera_name] = ctx

        log.info(f"Retrieve channel config for stream: {stream_name}")
        channels_config = shm_receiver.retrieve_channel_config(stream_name)
        subscriber_op = ShmSubscriberOp(self, shm_receiver, stream_name, channels_config, cycle_time_ms)

        visualizer = HolovizOp(
            self,
            name="visualizer",
            allocator=CudaStreamPool(
                self,
                name="cuda_stream",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=channels_config.get("numPorts", 1),
            ),
            **self.kwargs("holoviz"),
        )
        self.add_flow(subscriber_op, visualizer, {("outputs", "receivers")})
        self.add_flow(subscriber_op, visualizer, {("output_specs", "input_specs")})


def main(config_file=None):
    # make configurable or use holoscan debug level here too
    logging.basicConfig(level=logging.DEBUG)
    set_log_level(LogLevel.INFO)

    app = App()
    app.config(config_file)

    scheduler = EventBasedScheduler(app, worker_thread_number=8, name="ebs")
    app.scheduler(scheduler)

    with Tracker(app) as tracker:
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        tracker.print()


if __name__ == "__main__":


    parser = ArgumentParser(description="ARTEKMED Holoscan SHM Client.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_zhm_receiver.yaml")
    else:
        config_file = args.config

    main(config_file=config_file)
