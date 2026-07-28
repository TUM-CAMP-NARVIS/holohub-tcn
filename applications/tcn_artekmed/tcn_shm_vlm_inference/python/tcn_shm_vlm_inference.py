#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import threading
import math
import iceoryx2 as iox2

import numpy as np
import cupy as cp
import matplotlib
import cv2

import holoscan as hs
from holohub.tcn_shm_subscriber import TcnShmSubscriberOp as ShmSubscriberOp
from holohub.tcn_device_context._tcn_device_context import DeviceContextService
from holohub.tcn_shm_subscriber._tcn_shm_subscriber import discover_shm
from holohub.tcn_stream_splitter import TcnStreamSplitterOp as StreamSplitterOp
from holohub.tcn_convert_bgra_to_rgba import TcnConvertBgraToRgbaOp as ConvertBgraToRgbaOp

from operators.tcn_artekmed.tcn_util import RotateImage180Op

from holoscan.conditions import AsynchronousCondition, CountCondition
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

from da3_fragment import DA3MetricProcessingSubgraph, DA3PostprocessorOp
from da2_fragment import DA2MetricProcessingSubgraph, DA2PostprocessorOp
from langsam_fragment import LangSamProcessingSubgraph


log = logging.getLogger(__name__)


def create_tiled_input_specs(
    tensor_names,
    input_type=HolovizOp.InputType.COLOR,
    semantic_types=None,
    semantic_key=None,
):
    grid_size = int(math.ceil(math.sqrt(len(tensor_names)))) if tensor_names else 1
    tile_size = 1.0 / grid_size

    specs = []
    for i, tensor_name in enumerate(tensor_names):
        row = i // grid_size
        col = i % grid_size

        spec = HolovizOp.InputSpec(tensor_name, input_type)
        view = HolovizOp.InputSpec.View()
        view.offset_x = col * tile_size
        view.offset_y = row * tile_size
        view.width = tile_size
        view.height = tile_size
        spec.views = [view]

        if semantic_types is not None:
            key = semantic_key(tensor_name) if semantic_key is not None else tensor_name
            st = semantic_types.get(key)
            if st is not None:
                if st.content_type.get_format_type() == ImageFormatTypes.Rgba:
                    spec.image_format = holoviz._holoviz_str_to_image_format["r8g8b8a8_unorm"]
                elif st.content_type.get_format_type() == ImageFormatTypes.Bgra:
                    spec.image_format = holoviz._holoviz_str_to_image_format["b8g8r8a8_unorm"]
                else:
                    log.warning(f"Unsupported format type: {st.content_type.get_format_type()}")

        specs.append(spec)

    return specs




class DepthColormapOp(Operator):
    def __init__(self, fragment, *args, near=0.3, far=6.0, cmap="turbo", **kwargs):
        self.near, self.far = near, far
        lut = (matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.lut = cp.asarray(lut)                      # (256,3) on GPU
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        tensor = msg.get("image")
        if tensor is None:
            log.warning(f"da3 inference output is None {msg.keys()}")
            return
        log.info(f"{dir(tensor)} {type(tensor)}")
        depth = cp.from_dlpack(tensor)
        depth = cp.squeeze(depth).astype(cp.float32)

        valid = depth > 0                                # 0 / sky / invalid
        d = cp.clip((depth - self.near) / (self.far - self.near), 0.0, 1.0)
        idx = (d * 255).astype(cp.uint8)
        rgb = self.lut[idx]                              # [H,W,3] uint8
        a = cp.where(valid, 255, 0).astype(cp.uint8)[..., None]
        rgba = cp.concatenate([rgb, a], axis=-1)         # [H,W,4] uint8, RGBA
        op_output.emit({"image": rgba, "display_text": msg.get("display_text")}, "out")


class DummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("input")
        # log.info(f"DummySink received input {self.name}: {sig1}")



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

        log.info("Discover contents in shared memory")
        receiver_config = discover_shm(shm_stream_name)
        shm_receiver = receiver_config["receiver"]
        camera_names = receiver_config["camera_names"]
        device_contexts = receiver_config["device_contexts"]

        # Register the ctx_service with the fragment
        ctx_service = DeviceContextService.create(device_contexts)
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

        channels_config = receiver_config["channels_config"]
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

        color_output_specs = create_tiled_input_specs(
            [channel["name"] for channel in color_streams_config],
            semantic_types=channel_semantic_types,
        )

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
        shm_async_condition = AsynchronousCondition(self, name="shm_async_condition")
        subscriber_op = ShmSubscriberOp(self, cuda_stream_pool,
                                        allocator=device_memory_pool,
                                        async_condition=shm_async_condition,
                                        receiver=shm_receiver,
                                        stream_name=shm_stream_name,
                                        cycle_time_ms=cycle_time_ms,
                                        name="shm_subscriber")

        log.info(f"create stream_splitter op: {color_streams_config[:1]}")
        # XXX only one for now
        split_op = StreamSplitterOp(self, cuda_stream_pool,
                                    channel_names=[v["name"] for v in color_streams_config[:1]],
                                    name="stream_splitter")
        self.add_flow(subscriber_op, split_op, {("color_outputs", "receivers")})

        current_config = color_streams_config[0]
        channel_st = SemanticType(current_config['status']['bufferInfo']['semanticType'])

        di_sink = DummySinkOp(self, name="depth_image_sink")
        self.add_flow(subscriber_op, di_sink, {("depth_outputs", "input")})


        config_rotate_image = False
        have_camera_consumer = False

        inference_input = (split_op, "camera01_colorimage")
        camera_device_context = device_contexts["camera01"]

        if config_rotate_image:
            log.info("rotate image enabled")
            rotate_op = RotateImage180Op(self)
            self.add_flow(inference_input[0], rotate_op, {(inference_input[1], "input")})
            inference_input = (rotate_op, "output")

        need_convert_bgra = False
        if channel_st.content_type.get_format_type() == ImageFormatTypes.Rgba:
            need_convert_bgra = False
        elif channel_st.content_type.get_format_type() == ImageFormatTypes.Bgra:
            need_convert_bgra = True # camera_streams_config.get("enable_da3", False) or camera_streams_config.get("enable_langsam", False)
        else:
            log.warning(f"Unsupported format type: {channel_st.content_type.get_format_type()}")

        col_conv = None
        if need_convert_bgra:
            log.info("convert bgra to rgba enabled")
            col_conv = ConvertBgraToRgbaOp(self,
                                           allocator=device_memory_pool,
                                           in_tensor_name="",
                                           out_tensor_name="",
                                           name="color_converter_rgba")
            self.add_flow(inference_input[0], col_conv, {(inference_input[1], "input")})
            inference_input = (col_conv, "output")

        if camera_streams_config.get("enable_da2", False):
            da2_pipeline = DA2MetricProcessingSubgraph(self, "camera01_da2_pipeline", self.kwargs)

            holoviz_args = self.kwargs("da2_holoviz")

            # Register mouse event callbacks
            holoviz = HolovizOp(
                self,
                allocator=device_memory_pool,
                name="da2_holoviz",
                window_title="DepthAnything v2",
                **holoviz_args,
            )

            self.add_flow(inference_input[0], da2_pipeline, {(inference_input[1], "input")})
            self.add_flow(da2_pipeline, holoviz, {("output_image", "receivers")})
            self.add_flow(da2_pipeline, holoviz, {("output_specs", "input_specs")})
            have_camera_consumer = True

        if camera_streams_config.get("enable_da3", False):
            da3_pipeline = DA3MetricProcessingSubgraph(self, "camera01_da3_pipeline", self.kwargs,
                                                       device_context=camera_device_context)
            # dco = DepthColormapOp(self, name="da3_dco", near=0.3, far=10.0, cmap="turbo")

            holoviz_args = self.kwargs("da3_holoviz")

            # Register mouse event callbacks
            holoviz = HolovizOp(
                self,
                allocator=device_memory_pool,
                name="da3_holoviz",
                window_title="DepthAnything v3",
                tensors=[dict(name="image", type="color", opacity=1.0, priority=0)],
                **holoviz_args,
            )

            self.add_flow(inference_input[0], da3_pipeline, {(inference_input[1], "input")})
            # self.add_flow(da3_pipeline, dco, {("output_image", "in")})
            self.add_flow(da3_pipeline, holoviz, {("output_image", "receivers")})
            # self.add_flow(dco, holoviz, {("out", "receivers")})
            self.add_flow(da3_pipeline, holoviz, {("output_specs", "input_specs")})
            have_camera_consumer = True

        if camera_streams_config.get("enable_langsam", False):
            langsam_pipeline = LangSamProcessingSubgraph(self, "camera01_langsam_pipeline",
                                                         device_memory_pool,
                                                         self.kwargs)

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
            self.add_flow(inference_input[0], cs_sink, {(inference_input[1], "input")})


        log.info(f"create color visualizer: {color_output_specs}")

        color_visualizer = HolovizOp(
            self,
            name="color_visualizer",
            # tensors=color_output_specs,
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
