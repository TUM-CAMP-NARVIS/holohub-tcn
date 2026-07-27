#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import math

import iceoryx2 as iox2

import holoscan as hs
import cupy as cp
import numpy as np
import json

from holoscan.core import Operator, OperatorSpec, Tracker
from holoscan.conditions import CountCondition, PeriodicCondition, BooleanCondition
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp, holoviz
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler
from holoscan.resources import CudaStreamPool, BlockMemoryPool, MemoryStorageType, RMMAllocator
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3

from operators.tcn_artekmed.tcn_processing import ShmSimpleBackprojectionSubgraph, ShmConnection

from tcnart.core.semantic_type import SemanticType
from tcnart.core.semantic_type.model import ImageFormatTypes

log = logging.getLogger(__name__)


class DummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("input")
        log.debug(f"DummySink received input {self.name}")


class PointCloudDummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("positions")
        spec.input("texcoords")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("positions")
        sig2 = op_input.receive("texcoords")



class App(hs.core.Application):
    def compose(self):

        # Add your operators here
        print("Starting TCN Shm Receiver")
        stop_cond = BooleanCondition(self, name="stop_cond")

        # read configuration
        camera_streams_config = self.kwargs("camera_stream_processing")
        cuda_device_id = camera_streams_config.get("device_id", 0)
        block_memory_buffer_size = camera_streams_config.get("buffer_size", 8)

        shm_config = self.kwargs("shared_memory")
        shm_stream_name = shm_config.get("stream_name")
        cycle_time_ms = shm_config.get("cycle_time_ms")

        connection = ShmConnection(shm_stream_name, cycle_time_ms)
        if not connection.connect_shm():
            raise RuntimeError("could not connect to shm")

        max_frame_size = connection.get_max_tensor_size()
        num_channels = connection.get_num_streams()

        log.info(f"create cuda-stream pool with {num_channels} reserved streams on device {cuda_device_id}")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=cuda_device_id,
            stream_flags=0,
            stream_priority=0,
            reserved_size=num_channels,
            max_size=256,
        )

        log.info(f"create device-memory pool with {max_frame_size} bytes, {num_channels * block_memory_buffer_size} blocks on device {cuda_device_id}")
        device_memory_pool = BlockMemoryPool(
            self,
            name="shm_subscriber_device_pool",
            storage_type=MemoryStorageType.DEVICE,
            block_size=max_frame_size,
            num_blocks=num_channels * block_memory_buffer_size,
            dev_id=cuda_device_id
        )

        camera_streams = ShmSimpleBackprojectionSubgraph(self, "sbs",
                                                         stream_pool=cuda_stream_pool,
                                                         allocator=device_memory_pool,
                                                         shm_connection=connection,
                                                         fuse_buffers=False)
        output_specs = connection.get_output_specs()

        log.info("create color visualizer")
        color_visualizer = HolovizOp(
            self,
            name="color_visualizer",
            tensors=output_specs["color_output_specs"],
            allocator=device_memory_pool,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("color_holoviz"),
        )


        log.debug("Flow: camera_streams -> color_visualizer (color_outputs -> receivers)")
        self.add_flow(camera_streams, color_visualizer, {("color_outputs", "receivers")})

        # drain unused ports
        depth_dummy_sink = DummySinkOp(self, name="depth_dummy")
        self.add_flow(camera_streams, depth_dummy_sink, {("depth_outputs", "input")})

        pc_dummy_sink = PointCloudDummySinkOp(self, name="pointcloud_dummy")
        self.add_flow(camera_streams, pc_dummy_sink, {("texcoord_outputs", "texcoords")})
        self.add_flow(camera_streams, pc_dummy_sink, {("position_outputs", "positions")})


def main(config_file=None, scheduler_type="greedy", log_level="info", with_tracker=False):

    # warn,info,debug,debug_holoscan,debug_iceoryx,debug_all,trace
    if log_level == "trace":
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.TRACE)
        iox2.set_log_level(iox2.LogLevel.Trace)
    elif log_level == "debug_all":
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.DEBUG)
        iox2.set_log_level(iox2.LogLevel.Debug)
    elif log_level == "debug_iceoryx":
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Debug)
    elif log_level == "debug_holoscan":
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.DEBUG)
        iox2.set_log_level(iox2.LogLevel.Info)
    elif log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Info)
    elif log_level == "info":
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Info)
    elif log_level == "warn":
        logging.basicConfig(level=logging.WARN)
        set_log_level(LogLevel.WARN)
        iox2.set_log_level(iox2.LogLevel.Warn)
    else:
        raise ValueError(f"Invalid log level: {log_level}")

    app = App()
    app.config(config_file)

    scheduler = None
    if scheduler_type == "greedy":
        scheduler = GreedyScheduler(app, name="gs", stop_on_deadlock=True)
    elif scheduler_type == "event_based":
        scheduler = EventBasedScheduler(app, worker_thread_number=24, name="ebs")
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    app.scheduler(scheduler)

    if not with_tracker:
        try:
            app.run()
        except KeyboardInterrupt:
            pass
    else:
        with Tracker(app,
                     num_start_messages_to_skip=15,
                     num_last_messages_to_discard=15) as tracker:
            try:
                app.run()
            except KeyboardInterrupt:
                pass
            tracker.print()


if __name__ == "__main__":

    parser = ArgumentParser(description="ARTEKMED Holoscan SHM Example Receiver.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        default="greedy",
        help=("Set scheduler type [greedy,event_based]"),
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="info",
        help=("Set the log level [warn,info,debug,debug_holoscan,debug_iceoryx,debug_all]"),
    )
    parser.add_argument('-t', '--tracking', action='store_true', help='Enable dataflow tracking')

    args = parser.parse_args()

    if args.config == "none":
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_shm_backprojection_test.yaml")
    else:
        config_file = args.config

    main(config_file=config_file, scheduler_type=args.scheduler, log_level=args.log_level, with_tracker=args.tracking)
