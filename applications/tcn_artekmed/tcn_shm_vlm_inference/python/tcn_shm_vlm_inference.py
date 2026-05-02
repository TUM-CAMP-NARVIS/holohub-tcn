#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import threading
import iceoryx2 as iox2

import numpy as np
import cupy as cp
import cv2

import holoscan as hs
# from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
# from holohub.tcn_depthimage_temporal_filter import TcnDepthImageTemporalFilterOp
# from holohub.tcn_depthimage_weights import TcnDepthImageWeightsOp
# from holohub.tcn_texture_sampler import TcnTextureSamplerOp
# from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
#     RigidTransform, CameraParameters, make_rigid_transform
from operators.tcn_artekmed.tcn_shm_io import (ShmSubscriberOp, DeviceContextService, XYLookupTableSourceOp,
                                               create_shm_subscriber)
# from operators.tcn_artekmed.tcn_shm_io import ParameterRpcServer
from operators.tcn_artekmed.tcn_util import (StreamSplitterOp, StreamMergerOp, FlattenTensorOp,
                                             DepthImageMaxDistanceOp, DepthImageForegroundBackgroundMaskOp,
                                             DepthImageApplyMaskOp, ConvertBgraToRgbaOp, RotateImage180Op )

from holoscan.conditions import CountCondition
from holoscan.core import Operator, OperatorSpec, Tracker
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp
from holoscan.operators import holoviz
from holoscan.pose_tree import Pose3
from holoscan.resources import CudaStreamPool
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from holoscan.resources import RMMAllocator, UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler

from holoscan.pose_tree import PoseTreeManager, SO3
from holoscan.operators import (
    FormatConverterOp,
    InferenceOp,
)

from tcnart.core.semantic_type import SemanticType
from tcnart.core.semantic_type.model import ImageFormatTypes

from da2_fragment import DA2MetricProcessingSubgraph, DA2PostprocessorOp
from langsam_fragment import LangSamProcessingSubgraph


log = logging.getLogger(__name__)


class DummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("input")
        # log.debug(f"DummySink received input {self.name}")



class App(hs.core.Application):
    def compose(self):

        # Add your operators here
        print("Starting TCN Shm Receiver")

        camera_streams_config = self.kwargs("camera_stream_processing")
        cuda_device_id = camera_streams_config.get("device_id", 0)
        block_memory_buffer_size = camera_streams_config.get("buffer_size", 8)

        shm_config = self.kwargs("shared_memory")
        debug_output_config = self.kwargs("debug_output")

        shm_stream_name = shm_config.get("stream_name")
        cycle_time_ms = shm_config.get("cycle_time_ms")

        node, shm_receiver = create_shm_subscriber()

        log.info("Find cameras in shared memory")
        camera_names = shm_receiver.discover_devices()
        device_contexts = {}
        for camera_name in camera_names:
            log.info(f"Retrieving camera_info: {camera_name}")
            ctx = shm_receiver.retrieve_device_context(camera_name)
            if ctx is not None:
                device_contexts[camera_name] = ctx

        # Register the ctx_service with the fragment
        ctx_service = DeviceContextService(device_contexts)
        self.register_service(ctx_service)

        # # create pose tree service for fragment
        pts = PoseTreeManager(
            self,
            name="pose_tree_manager",
            **self.kwargs("pose_tree_config"),
        )
        self.register_service(pts)

        # # configure pose-tree
        pose_tree_config = None
        # pose_tree_config = {"frames": [], "edges": []}
        # pose_tree_config["frames"].append("world_origin")
        # for name in camera_names:
        #     log.info(f"Create Reference frames for camera {name}")
        #     # the frames are intentionally called like the streams to simplify lookup
        #     depth_channel_name = f"{name}_depthimage"
        #     color_channel_name = f"{name}_colorimage"
        #     pose_tree_config["frames"].append(depth_channel_name)
        #     pose_tree_config["frames"].append(color_channel_name)
        #     pose_tree_config["edges"].append(("world_origin", depth_channel_name, convert_rigid_transform_to_pose3(ctx_service.get_depth_extrinsics(name))))
        #     pose_tree_config["edges"].append((color_channel_name, depth_channel_name, convert_rigid_transform_to_pose3(ctx_service.get_color_to_depth(name))))

        log.info(f"Retrieve channel config for stream: {shm_stream_name}")
        channels_config = shm_receiver.retrieve_channel_config(shm_stream_name)
        channel_semantic_types = {
            v['name']: SemanticType(v['status']['bufferInfo']['semanticType']) for v in channels_config['ports']
        }

        depth_streams_config = []
        color_streams_config = []
        max_frame_size = 0
        num_channels = len(channels_config["ports"])

        log.info(channels_config)

        for channel in channels_config["ports"]:
            max_frame_size = max(max_frame_size, channel["status"]["bufferInfo"]["frameSize"])
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)

            if channel["status"]["portType"] == "colorimage":
                color_streams_config.append(channel)

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

        log.info(f"create subscriber op {shm_stream_name}")
        subscriber_op = ShmSubscriberOp(self, cuda_stream_pool, device_memory_pool, shm_receiver, shm_stream_name,
                                        channels_config, pose_tree_config, cycle_time_ms, name="shm_subscriber")

        log.info("create stream_splitter op")
        # XXX only one for now
        split_op = StreamSplitterOp(self, cuda_stream_pool, [v["name"] for v in color_streams_config[:1]], name="stream_splitter")
        self.add_flow(subscriber_op, split_op, {("color_outputs", "receivers")})

        current_config = color_streams_config[0]
        channel_st = SemanticType(current_config['status']['bufferInfo']['semanticType'])

        di_sink = DummySinkOp(self, name="depth_image_sink")
        self.add_flow(subscriber_op, di_sink, {("depth_outputs", "input")})


        config_rotate_image = True
        have_camera_consumer = False

        inference_input = (split_op, "camera01_colorimage")

        if config_rotate_image:
            log.info("rotate image enabled")
            rotate_op = RotateImage180Op(self)
            self.add_flow(inference_input[0], rotate_op, {(inference_input[1], "input")})
            inference_input = (rotate_op, "output")

        need_convert_bgra = False
        if channel_st.content_type.get_format_type() == ImageFormatTypes.Rgba:
            need_convert_bgra = False
        elif channel_st.content_type.get_format_type() == ImageFormatTypes.Bgra:
            need_convert_bgra = True # camera_streams_config.get("enable_da2", False) or camera_streams_config.get("enable_langsam", False)
        else:
            log.warning(f"Unsupported format type: {channel_st.content_type.get_format_type()}")

        col_conv = None
        if need_convert_bgra:
            log.info("convert bgra to rgba enabled")
            col_conv = ConvertBgraToRgbaOp(self, name="color_converter_rgba")
            self.add_flow(inference_input[0], col_conv, {(inference_input[1], "input")})
            inference_input = (col_conv, "output")

        if camera_streams_config.get("enable_da2", False):
            da_pipeline = DA2MetricProcessingSubgraph(self, "camera01_da2_pipeline", self.kwargs)

            holoviz_args = self.kwargs("holoviz")

            # Register mouse event callbacks
            holoviz = HolovizOp(
                self,
                allocator=device_memory_pool,
                name="holoviz",
                window_title="DepthAnything v2",
                **holoviz_args,
            )

            self.add_flow(inference_input[0], da_pipeline, {(inference_input[1], "input")})
            self.add_flow(da_pipeline, holoviz, {("output_image", "receivers")})
            self.add_flow(da_pipeline, holoviz, {("output_specs", "input_specs")})
            have_camera_consumer = True

        if camera_streams_config.get("enable_langsam", False):
            langsam_pipeline = LangSamProcessingSubgraph(self, "camera01_langsam_pipeline", self.kwargs)

            holoviz_args = self.kwargs("langsam_holoviz")

            langsam_holoviz = HolovizOp(
                self,
                allocator=device_memory_pool,
                name="langsam_holoviz",
                window_title="LangSAM (Grounding DINO + SAM2)",
                **holoviz_args,
            )

            self.add_flow(inference_input[0], langsam_pipeline, {(inference_input[1], "input")})
            self.add_flow(langsam_pipeline, langsam_holoviz, {("output_masks", "receivers")})
            have_camera_consumer = True

        if not have_camera_consumer:
            cs_sink = DummySinkOp(self, name="camera_stream_sink")
            self.add_flow(split_op, cs_sink, {("camera01_colorimage", "input")})


        log.info("create color visualizer")

        color_visualizer = HolovizOp(
            self,
            name="color_visualizer",
            allocator=device_memory_pool,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("color_holoviz"),
        )

        self.add_flow(inference_input[0], color_visualizer, {(inference_input[1], "receivers")})
        #self.add_flow(subscriber_op, color_visualizer, {("color_output_specs", "input_specs")})


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

    parser = ArgumentParser(description="ARTEKMED Holoscan VLM Inference.")

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
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_shm_vlm_inference.yaml")
    else:
        config_file = args.config

    main(config_file=config_file, scheduler_type=args.scheduler, log_level=args.log_level, with_tracker=args.tracking)
