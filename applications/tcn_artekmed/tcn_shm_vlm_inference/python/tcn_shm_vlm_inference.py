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
from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_temporal_filter import TcnDepthImageTemporalFilterOp
from holohub.tcn_depthimage_weights import TcnDepthImageWeightsOp
from holohub.tcn_texture_sampler import TcnTextureSamplerOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
    RigidTransform, CameraParameters, make_rigid_transform
from operators.tcn_artekmed.tcn_shm_io import (ShmSubscriberOp, DeviceContextService, XYLookupTableSourceOp,
                                               create_shm_subscriber)
# from operators.tcn_artekmed.tcn_shm_io import ParameterRpcServer
from operators.tcn_artekmed.tcn_util import (StreamSplitterOp, StreamMergerOp, FlattenTensorOp,
                                             DepthImageMaxDistanceOp, DepthImageForegroundBackgroundMaskOp,
                                             DepthImageApplyMaskOp )

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


log = logging.getLogger(__name__)

def to_holoviz_pose(pose : Pose3) -> holoviz.Pose3D:
    result : holoviz.Pose3D = make_pose() # helper from module as cannot be created in python otherwise
    result.translation = pose.translation
    result.rotation = pose.rotation.matrix().flatten()
    return result

def convert_rigid_transform_to_pose3(input: RigidTransform) -> Pose3:
    log.debug(f"RigidTransform translation: {input.translation} rotation: {input.rotation}")
    r = SO3.from_quaternion(input.rotation)
    return Pose3(r, input.translation)



class DummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("input")
        # log.debug(f"DummySink received input {self.name}")


class PointCloudDummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("positions")
        spec.input("texcoords")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("positions")
        sig2 = op_input.receive("texcoords")
        # print("received positions and texcoords")



class DA2PostprocessorOp(Operator):
    """Operator that does postprocessing before sending resulting image to Holoviz"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.image_dim = 518
        self.mouse_pressed = False
        self.display_modes = ["original", "depth", "side-by-side", "interactive"]
        self.idx = 1
        self.current_display_mode = self.display_modes[self.idx]
        # In interactive mode, how much of the original video to show
        self.ratio = 0.5

    def setup(self, spec: OperatorSpec):
        """
        input:  "input_depthmap"  - Input tensors representing depthmap from inference
        input:  "input_image"     - Input tensor representing the RGB image
        output: "output_image"    - The image for Holoviz to display
        output: "output_specs"    - Text to show the current display mode

        This operator's output image depends on the current display mode, if set to

            * "original": output the original image from input source
            * "depth": output the color depthmap based on the depthmap returned from
                       Depth Anything V2 model
            * "side-by-side": output a side-by-side view of the original image next to
                              the color depthmap
            * "interactive": allow user to control how much of the image to show as
                             original while the rest shows the color depthmap

        Returns:
            None
        """
        spec.input("input_depthmap")
        spec.input("input_image")
        spec.output("output_image")
        spec.output("output_specs")

    def clamp(self, value, min_value=0, max_value=1):
        """Clamp value between [min_value, max_value]"""
        return max(min_value, min(max_value, value))

    def toggle_display_mode(self, *args):
        mouse_button = args[0]
        action = args[1]

        LEFT_BUTTON = 0
        PRESSED = 0

        # If event is for the middle or right mouse button, update some values for interactive mode
        #   - update the status of whether the button is being pressed or released
        #   - update the ratio of the original image to display
        if mouse_button.value != LEFT_BUTTON:
            self.mouse_pressed = action.value == PRESSED
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size
            return

        # When left mouse button is pressed, update the display mode
        if action.value == PRESSED:
            self.idx = (self.idx + 1) % len(self.display_modes)
            self.current_display_mode = self.display_modes[self.idx]

    # Update cursor position which will be used in interactive mode
    def cursor_pos_callback(self, *args):
        self.x = args[0]
        if self.mouse_pressed:
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size

    # Update size of holoviz framer buffer which will be used to calculate self.ratio
    def framebuffer_size_callback(self, *args):
        self.framebuffer_size = args[0]

    def normalize(self, depth_map):
        min_value = cp.min(depth_map)
        max_value = cp.max(depth_map)
        normalized = (depth_map - min_value) / (max_value - min_value)
        return 255 - (normalized * 255)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input_depthmap")
        in_image = op_input.receive("input_image")

        # Convert input to cupy array
        inference_output = cp.asarray(in_message.get("inference_output")).squeeze()

        image = cp.asarray(in_image.get("preprocessed"))

        if self.current_display_mode == "original":
            # Display the original image
            image = (image * 255).astype(cp.uint8)
            output_image = image
        elif self.current_display_mode == "depth":
            # Display the color depthmap
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            output_image = depth_colormap
        elif self.current_display_mode == "side-by-side":
            # Display both original and color depthmap images side-by-side
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            output_image = cp.hstack((image, depth_colormap))
        else:
            # Interactive mode
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            pos = int(self.image_dim * self.ratio)
            output_image = cp.hstack(
                (
                    image[:, :pos, :],
                    depth_colormap[
                        :,
                        pos:,
                    ],
                )
            )

        # Position display mode text near bottom left corner of Holoviz window
        display_mode_text = np.asarray([(0.025, 0.9)])

        # Create output message
        out_message = {"display_mode": display_mode_text, "image": hs.as_tensor(output_image)}
        op_output.emit(out_message, "output_image")

        # holoviz specs for displaying the current display mode
        specs = []
        spec = HolovizOp.InputSpec("display_mode", "text")
        spec.text = [self.current_display_mode]
        spec.color = [1.0, 1.0, 1.0, 1.0]
        spec.priority = 1
        specs.append(spec)
        op_output.emit(specs, "output_specs")



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
        # channel_semantic_types = {
        #     v['name']: SemanticType(v['status']['bufferInfo']['semanticType']) for v in channels_config['ports']
        # }

        depth_streams_config = []
        color_streams_config = []
        max_frame_size = 0
        num_channels = len(channels_config["ports"])

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

        di_sink = DummySinkOp(self, name="depth_image_sink")
        self.add_flow(subscriber_op, di_sink, {("depth_outputs", "input")})

        in_dtype = "rgba8888"

        pool = UnboundedAllocator(self, name="pool")
        da2_preprocessor_args = self.kwargs("da2_preprocessor")
        da2_preprocessor = FormatConverterOp(
            self,
            name="da2_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **da2_preprocessor_args,
        )

        da2_inference_args = self.kwargs("da2_inference")
        da2_inference_args["model_path_map"] = {
            "depth": os.path.join("/srv/models/active/depth_anything_v2", "depth_anything_v2_vits.onnx")
        }

        da2_inference = InferenceOp(
            self,
            name="da2_inference",
            allocator=pool,
            **da2_inference_args,
        )

        da2_postprocessor = DA2PostprocessorOp(self, name="da2_postprocessor", allocator=pool)

        holoviz_args = self.kwargs("holoviz")

        # Register mouse event callbacks
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="DepthAnything v2",
            mouse_button_callback=da2_postprocessor.toggle_display_mode,
            cursor_pos_callback=da2_postprocessor.cursor_pos_callback,
            framebuffer_size_callback=da2_postprocessor.framebuffer_size_callback,
            **holoviz_args,
        )

        self.add_flow(split_op, da2_preprocessor, {("camera01_colorimage", "source_video")})
        self.add_flow(da2_preprocessor, da2_postprocessor, {("tensor", "input_image")})
        self.add_flow(da2_preprocessor, da2_inference, {("", "receivers")})
        self.add_flow(da2_inference, da2_postprocessor, {("transmitter", "input_depthmap")})
        self.add_flow(da2_postprocessor, holoviz, {("output_image", "receivers")})
        self.add_flow(da2_postprocessor, holoviz, {("output_specs", "input_specs")})






        log.info("create color visualizer")
        color_visualizer = HolovizOp(
            self,
            name="color_visualizer",
            allocator=device_memory_pool,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("color_holoviz"),
        )

        self.add_flow(split_op, color_visualizer, {("camera01_colorimage", "receivers")})
        # self.add_flow(subscriber_op, color_visualizer, {("color_outputs", "receivers")})
        # self.add_flow(subscriber_op, color_visualizer, {("color_output_specs", "input_specs")})


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
