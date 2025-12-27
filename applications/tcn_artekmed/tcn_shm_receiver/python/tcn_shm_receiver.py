#
# Place the license header here
#
import os
import logging
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import queue
from concurrent.futures import Future, ThreadPoolExecutor
import re
import time

import holoscan as hs
import numpy as np
import cupy as cp

from holoscan.gxf import Entity
from holoscan.logger import LogLevel, set_log_level
from holoscan.resources import CudaStreamPool, UnboundedAllocator, BlockMemoryPool
from holoscan.resources import RMMAllocator
from holoscan.conditions import AsynchronousCondition, AsynchronousEventState, BooleanCondition, CountCondition, PeriodicCondition, MessageAvailableCondition, DownstreamMessageAffordableCondition
from holoscan.core import Application, DefaultFragmentService, ConditionType, IOSpec, Operator, OperatorSpec, Tracker
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler
from holoscan.operators import HolovizOp
from holoscan.operators import holoviz
from holoscan.pose_tree import SO3, Pose3, PoseTreeManager, PoseTree, PoseTreeAccessMethod

from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, make_pose

import iceoryx2 as iox2
from tcnart.core.semantic_type import SemanticType
from shm_receiver import ShmSynchronizedBufferReceiver
from shm_serde import shm_transport_enum

from pyxylt import create_xy_lookup_table_from_intrinsics, IntrinsicParameters


log = logging.getLogger(__name__)

def to_holoviz_pose(pose : Pose3) -> holoviz.Pose3D:
    result : holoviz.Pose3D = make_pose() # helper from module as cannot be created in python otherwise
    result.translation = pose.translation
    result.rotation = pose.rotation.matrix().flatten()
    return result


class DeviceContextService(DefaultFragmentService):
    """A simple fragment service that holds an integer value."""

    def __init__(self, device_contexts: Dict[str, Any]):
        super().__init__()
        self._device_contexts = device_contexts
        self.portname_cameraname_match = re.compile("^(camera[0-9]+)_.*$")

    def get_camera_name_from_port_name(self, port_name: str) -> str | None:
        m = self.portname_cameraname_match.match(port_name)
        if m is not None:
            return m.group(1)

    def raw_context(self) -> Dict[str, Any]:
        """Get the value stored in the service."""
        return self._device_contexts

    def get_device_context(self, camera_name: str) -> Dict[str, Any]:
        if camera_name not in self._device_contexts:
            log.error(f"no camera found with name: {camera_name}")
            return None
        return self._device_contexts[camera_name]

    def get_device_calibration(self, camera_name: str) -> Dict[str, Any]:
        ctx = self.get_device_context(camera_name)
        if ctx is None:
            return None
        if "calibration" not in ctx:
            log.error(f"no calibration found for camera_name: {camera_name}")
            return None
        return ctx["calibration"]

    def get_depth_camera_model(self, camera_name: str) -> CameraModel | None:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "depthCameraParameters" not in calib:
            log.error(f"no depthCameraParameters found for camera_name: {camera_name}")
            return None
        params = calib["depthCameraParameters"]
        return self._camera_model_from_dict(params)

    def get_color_camera_model(self, camera_name: str) -> CameraModel | None:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "colorCameraParameters" not in calib:
            log.error(f"no colorCameraParameters found for camera_name: {camera_name}")
            return None
        params = calib["colorCameraParameters"]
        return self._camera_model_from_dict(params)

    def _camera_model_from_dict(self, params) -> CameraModel:
        model = CameraModel()
        model.distortion_type = DistortionType.Brown
        model.dimensions.x = params["width"]
        model.dimensions.y = params["height"]
        model.focal_length.x = params["fovX"]
        model.focal_length.y = params["fovY"]
        model.principal_point.x = params["cX"]
        model.principal_point.y = params["cY"]
        model.skew_value = 1.0
        model.distortion_coefficients = [
            params["distortionParams"]["k1"],
            params["distortionParams"]["k2"],
            params["distortionParams"]["tx"],
            params["distortionParams"]["ty"],
            params["distortionParams"]["k3"],
            params["distortionParams"]["k4"],
            params["distortionParams"]["k5"],
            params["distortionParams"]["k6"],
        ]

        return model

    def get_xy_table_intrinsics(self, camera_name: str) -> IntrinsicParameters | None:
        model = self.get_depth_camera_model(camera_name)
        if model is None:
            return None
        intrinsics = IntrinsicParameters()
        intrinsics.fov_x = float(model.focal_length.x)
        intrinsics.fov_y = float(model.focal_length.y)
        intrinsics.c_x = float(model.principal_point.x)
        intrinsics.c_y = float(model.principal_point.y)
        intrinsics.width = int(model.dimensions.x)
        intrinsics.height = int(model.dimensions.y)
        intrinsics.tangential_distortion = [
            model.distortion_coefficients[2],
            model.distortion_coefficients[3],
        ]
        intrinsics.radial_distortion = [
            model.distortion_coefficients[0],
            model.distortion_coefficients[1],
            model.distortion_coefficients[4],
            model.distortion_coefficients[5],
            model.distortion_coefficients[6],
            model.distortion_coefficients[7],
        ]
        return intrinsics

    def get_xy_table(self, camera_name: str) -> np.ndarray | None:
        intrinsics : IntrinsicParameters | None = self.get_xy_table_intrinsics(camera_name)
        if intrinsics is None:
            return None
        log.info(f"Creating xy-table for {camera_name}")
        xy_lookup_table = create_xy_lookup_table_from_intrinsics(intrinsics)

        if not xy_lookup_table:
            raise RuntimeError(f"Failed to create XY lookup table for camera {camera_name}")
        if xy_lookup_table.width == 0 or xy_lookup_table.height == 0 or len(xy_lookup_table.data) == 0:
            raise RuntimeError(f"XY lookup table for camera {camera_name} is empty despite success=True")

        return np.array(xy_lookup_table.data, dtype=np.float32).reshape((xy_lookup_table.height, xy_lookup_table.width, 2))

    def get_depth_extrinsics(self, camera_name: str) -> Pose3 | None:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "cameraPose" not in calib:
            log.error(f"no cameraPose found for camera_name: {camera_name}")
            return None
        params = calib["cameraPose"]
        return self._pose3d_from_dict(params)

    def get_color_to_depth(self, camera_name: str) -> Pose3 | None:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "color2depthTransform" not in calib:
            log.error(f"no color2depthTransform found for camera_name: {camera_name}")
            return None
        params = calib["color2depthTransform"]
        return self._pose3d_from_dict(params)


    def _pose3d_from_dict(self, params) -> Pose3 | None:
        return Pose3(
            SO3.from_quaternion(np.asarray([
                params["rotation"]["x"],
                params["rotation"]["y"],
                params["rotation"]["z"],
                params["rotation"]["w"],
            ])),
            np.asarray([
                params["translation"]["x"],
                params["translation"]["y"],
                params["translation"]["z"],
            ])
        )


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
                data[port.name] = cp.asarray(np.frombuffer(mv, dtype=np.uint8).reshape(
                    md.header.dimY, md.header.dimX, int(md.header.bitsPerElement/8)
                ))
            elif port.data.portType == shm_transport_enum.CameraPortType.depthimage:
                # log.debug(f"added port data: {port.name}")
                md = port.data.metadata
                mv = memoryview(port.data.data)
                data[port.name] = cp.asarray(np.frombuffer(mv, dtype=np.uint16).reshape(
                    md.header.dimY, md.header.dimX, 1
                ))

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
            # hardcoded ..
            if "color" in port_name:
                spec.image_format = holoviz._holoviz_str_to_image_format["b8g8r8a8_unorm"]
            output_specs.append(spec)

        op_output.emit(output_specs, "output_specs", acq_timestamp=ts)


    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        if self.subscriber is not None:
            self.subscriber = None
        self.future_.result()


class CameraStreamExtractor(Operator):

    def __init__(
            self,
            fragment: Any,
            channel_config: Any,
            *args,
            **kwargs,
        ):
        self.channel_config = channel_config
        self.channel_static_buffers = {}
        # Need to call the base class constructor last

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers")
        for config in self.channel_config:
            channel_name = config["name"]
            spec.output(channel_name)
            spec.output(f"{channel_name}_xy_table")

        self.ctx_service = self.service(DeviceContextService)
        # self.pose_tree_manager = self.service(PoseTreeManager)
        # pose_tree = self.pose_tree_manager.tree

        for config in self.channel_config:
            channel_name = config["name"]
            camera_name = self.ctx_service.get_camera_name_from_port_name(channel_name)
            xytable = self.ctx_service.get_xy_table(camera_name)
            xytable_dev = cp.asarray(xytable)
            self.channel_static_buffers[channel_name] = {
                "xy_table": {"": hs.as_tensor(xytable_dev)},
            }

    def compute(self, op_input, op_output, context):
        message = op_input.receive("receivers")

        for config in self.channel_config:
            channel_name = config["name"]
            di_tensor = hs.as_tensor(cp.asarray(message.get(channel_name)))
            op_output.emit({"": di_tensor}, channel_name)
            for key, value in self.channel_static_buffers[channel_name].items():
                if value is None:
                    log.warning(f"static buffer {key} is None")
                op_output.emit(value, f"{channel_name}_{key}")


class BPDummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("depth_image")
        spec.input("xy_table")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("depth_image")
        sig2 = op_input.receive("xy_table")


class PointCloudDummySinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("positions")
        spec.input("texcoords")

    def compute(self, op_input, op_output, context):
        sig1 = op_input.receive("positions")
        sig2 = op_input.receive("texcoords")
        print("received positions and texcoords")

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

        # Register the ctx_service with the fragment
        ctx_service = DeviceContextService(device_contexts)
        self.register_service(ctx_service)

        # # create pose tree service for fragment
        # pose_tree_config = self.kwargs("pose_tree_config")  # see pose_tree_basic.yaml

        # pts = PoseTreeManager(
        #     self,
        #     name="pose_tree_manager",
        #     **pose_tree_config,
        # )
        # self.register_service(pts)

        # # configure pose-tree
        # pts.tree.create_frame("world_origin")
        # for name in camera_names:
        #     log.info(f"Create Reference frames for camera {name}")
        #     # the frames are intentionally called like the streams to simplify lookup
        #     depth_channel_name = f"{name}_depthimage"
        #     color_channel_name = f"{name}_colorimage"
        #     pts.tree.create_frame(depth_channel_name)
        #     pts.tree.create_frame(color_channel_name)
        #     # connect the frames
        #     pts.tree.create_edges("world_origin", depth_channel_name)
        #     pts.tree.create_edges(depth_channel_name, color_channel_name)
        #     # set transforms
        #     # pts.tree.set("world_origin", depth_channel_name, 0, ctx_service.get_depth_extrinsics(name))
        #     # pts.tree.set(color_channel_name, depth_channel_name, 0, ctx_service.get_color_to_depth(name))



        log.info(f"Retrieve channel config for stream: {stream_name}")
        channels_config = shm_receiver.retrieve_channel_config(stream_name)
        subscriber_op = ShmSubscriberOp(self, shm_receiver, stream_name, channels_config, cycle_time_ms, name="shm_subscriber")

        depth_streams_config = []
        for channel in channels_config["ports"]:
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)

        split_op = CameraStreamExtractor(self, depth_streams_config, name="stream_splitter")
        self.add_flow(subscriber_op, split_op, {("outputs", "receivers")})

        sink_ops = []
        for channel in depth_streams_config:
            channel_name = channel["name"]
            camera_name = ctx_service.get_camera_name_from_port_name(channel_name)
            color_params = ctx_service.get_color_camera_model(camera_name)
            bp_op = TcnDepthImageBackprojectionOp(
                self, 
                allocator=RMMAllocator(self, name=f"rmm-allocator_{channel_name}", **self.kwargs("rmm_allocator")),
                depth_units_per_meter=1000.0,
                near_limit_m=0.01,
                far_limit_m=10.0,
                color_image_width=color_params.dimensions.x,
                color_image_height=color_params.dimensions.y,
                color_params=ctx_service.get_depth_camera_model(camera_name),
                depth_extrinsics=ctx_service.get_depth_extrinsics(camera_name),
                color_to_depth=ctx_service.get_color_to_depth(camera_name),
                name=f"{camera_name}_backprojection",
                )
            self.add_flow(split_op, bp_op, {
                (channel_name, "depth_image"),
                (f"{channel_name}_xy_table", "xy_table"),
                (f"{channel_name}_depth_params", "depth_params"),
                (f"{channel_name}_color_params", "color_params"),
                (f"{channel_name}_color_to_depth", "color_to_depth"),
                (f"{channel_name}_depth_extrinsics", "depth_extrinsics"),
            })
            sink_ops.append(bp_op)

            sink_op = PointCloudDummySinkOp(self, name=f"{camera_name}_sink")
            self.add_flow(bp_op, sink_op, {
                ("positions", "positions"),
                ("texcoords", "texcoords"),
                })
            sink_ops.append(sink_op)

            # sink_op = SinkOp(self, name=f"{camera_name}_sink")
            # self.add_flow(split_op, sink_op, {
            #     (channel_name, "depth_image"),
            #     (f"{channel_name}_xy_table", "xy_table"),
            #     (f"{channel_name}_depth_params", "depth_params"),
            #     (f"{channel_name}_color_params", "color_params"),
            #     (f"{channel_name}_color_to_depth", "color_to_depth"),
            #     (f"{channel_name}_depth_extrinsics", "depth_extrinsics"),
            #     })
            # sink_ops.append(sink_op)

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
    logging.basicConfig(level=logging.INFO)
    set_log_level(LogLevel.TRACE)
    iox2.set_log_level(iox2.LogLevel.Warn)

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
