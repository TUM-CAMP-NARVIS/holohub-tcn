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
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, RigidTransform, CameraParameters, make_rigid_transform

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

    def get_camera_name_from_port_name(self, port_name: str) -> Optional[str]:
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

    def get_depth_camera_model(self, camera_name: str) -> Optional[CameraModel]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "depthCameraParameters" not in calib:
            log.error(f"no depthCameraParameters found for camera_name: {camera_name}")
            return None
        params = calib["depthCameraParameters"]
        return self._camera_model_from_dict(params)

    def get_color_camera_model(self, camera_name: str) -> Optional[CameraModel]:
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

    def get_xy_table_intrinsics(self, camera_name: str) -> Optional[IntrinsicParameters]:
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

    def get_xy_table(self, camera_name: str) -> Optional[np.ndarray]:
        intrinsics : Optional[IntrinsicParameters] = self.get_xy_table_intrinsics(camera_name)
        if intrinsics is None:
            return None
        log.info(f"Creating xy-table for {camera_name}")
        xy_lookup_table = create_xy_lookup_table_from_intrinsics(intrinsics)

        if not xy_lookup_table:
            raise RuntimeError(f"Failed to create XY lookup table for camera {camera_name}")
        if xy_lookup_table.width == 0 or xy_lookup_table.height == 0 or len(xy_lookup_table.data) == 0:
            raise RuntimeError(f"XY lookup table for camera {camera_name} is empty despite success=True")

        return np.array(xy_lookup_table.data, dtype=np.float32).reshape((xy_lookup_table.height, xy_lookup_table.width, 2))

    def get_depth_extrinsics(self, camera_name: str) -> Optional[RigidTransform]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "cameraPose" not in calib:
            log.error(f"no cameraPose found for camera_name: {camera_name}")
            return None
        params = calib["cameraPose"]
        return self._pose3d_from_dict(params)

    def get_color_to_depth(self, camera_name: str) -> Optional[RigidTransform]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "color2depthTransform" not in calib:
            log.error(f"no color2depthTransform found for camera_name: {camera_name}")
            return None
        params = calib["color2depthTransform"]
        return self._pose3d_from_dict(params)


    def _pose3d_from_dict(self, params) -> Optional[RigidTransform]:
        return make_rigid_transform(
            hs.as_tensor(np.asarray([
                params["translation"]["x"],
                params["translation"]["y"],
                params["translation"]["z"],
            ])),
            hs.as_tensor(np.asarray([
                params["rotation"]["x"],
                params["rotation"]["y"],
                params["rotation"]["z"],
                params["rotation"]["w"],
            ]))
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
        color_data = {}
        depth_data = {}
        frame_timestamp = user_header.timestamp
        for port in message.ports:
            if port.data.portType == shm_transport_enum.CameraPortType.colorimage:
                # log.debug(f"added port data: {port.name}")
                md = port.data.metadata
                mv = memoryview(port.data.data)
                color_data[port.name] = cp.asarray(np.frombuffer(mv, dtype=np.uint8).reshape(
                    md.header.dimY, md.header.dimX, int(md.header.bitsPerElement/8)
                ))
            elif port.data.portType == shm_transport_enum.CameraPortType.depthimage:
                # log.debug(f"added port data: {port.name}")
                md = port.data.metadata
                mv = memoryview(port.data.data)
                depth_data[port.name] = cp.asarray(np.frombuffer(mv, dtype=np.uint16).reshape(
                    md.header.dimY, md.header.dimX, 1
                ))

        if color_data or depth_data:
            # how does ts relate to fragment.scheduler().clock.timestamp()?
            # log.debug(f"put data for {frame_timestamp} into queue")
            self.buffer.put((frame_timestamp, (color_data, depth_data)))

            if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE
            return True

        return False

    def receiver_mainloop(self):
        while self.subscriber is not None:
            if not self.subscriber.receive_frame(self.on_receive, self.cycle_time_ms):
                log.warning("could not receive frame.")

    def setup(self, spec: OperatorSpec):
        spec.output("color_outputs")
        spec.output("color_output_specs").condition(hs.core.ConditionType.NONE)
        spec.output("depth_outputs")
        spec.output("depth_output_specs").condition(hs.core.ConditionType.NONE)

    def start(self):
        self.subscriber.subscribe(self.stream_name)

        self.future_ = self.executor_.submit(self.receiver_mainloop)
        assert isinstance(self.future_, Future)

    def compute(self, op_input, op_output, context):
        scheduler = self.fragment.scheduler()
        clock = scheduler.clock
        ts = clock.timestamp()

        frame_ts, (color_data, depth_data) = self.buffer.get()
        log.debug(f"got data for {frame_ts} from queue")

        color_message = {k:hs.as_tensor(v) for k,v in color_data.items()}
        depth_message = {k:hs.as_tensor(v) for k,v in depth_data.items()}

        self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING
        op_output.emit(color_message, "color_outputs", acq_timestamp=ts)
        op_output.emit(depth_message, "depth_outputs", acq_timestamp=ts)

        num_videos = len(color_message.keys())
        # assume they are the same
        #num_depth_videos = len(depth_message.keys())


        # Determine grid size (e.g., 2 for 2x2, 3 for 3x3)
        grid_size = int(np.ceil(np.sqrt(num_videos))) if num_videos > 0 else 1
        tile_size = 1.0 / grid_size

        color_output_specs = []
        for i, port_name in enumerate(color_message.keys()):
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
            spec.image_format = holoviz._holoviz_str_to_image_format["b8g8r8a8_unorm"]
            color_output_specs.append(spec)

        op_output.emit(color_output_specs, "color_output_specs", acq_timestamp=ts)

        depth_output_specs = []
        for i, port_name in enumerate(depth_message.keys()):
            # Compute row and column index
            row = i // grid_size
            col = i % grid_size

            # Compute normalized offsets (0.0 to 1.0)
            # Note: Holoviz usually uses (x, y) for offsets
            offset_x = col * tile_size
            offset_y = row * tile_size

            # still using color as uint16 is not supported as depth-format
            spec = HolovizOp.InputSpec(port_name, HolovizOp.InputType.COLOR)
            views = []
            view = HolovizOp.InputSpec.View()
            view.offset_x = offset_x
            view.offset_y = offset_y
            view.width = tile_size
            view.height = tile_size
            views.append(view)
            spec.views = views
            depth_output_specs.append(spec)

        op_output.emit(depth_output_specs, "depth_output_specs", acq_timestamp=ts)


    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        if self.subscriber is not None:
            self.subscriber = None
        self.future_.result()


class XYLookupTableSourceOp(Operator):
    def __init__(self, fragment: Any, *args, **kwargs):
        self.ctx_service = None
        self.xy_table_data = None
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        self.xy_table_data = cp.asarray(self.ctx_service.get_xy_table(self.camera_name))

    def setup(self, spec: OperatorSpec):
        spec.output("xy_table")
        spec.param("camera_name")
        self.ctx_service = self.service(DeviceContextService)

    def compute(self, op_input, op_output, context):
        if self.xy_table_data is not None:
            try:
                xytable_tensor = hs.as_tensor(self.xy_table_data)
                op_output.emit({"": xytable_tensor}, "xy_table")
            except Exception as e:
                log.exception(e)
        else:
            log.error(f"XYLookupTableSourceOp: Could not create XY Table for camera: {self.camera_name}")


class StreamSplitterOp(Operator):

    def __init__(
            self,
            fragment: Any,
            channel_config: Any,
            *args,
            **kwargs,
        ):
        self.channel_config = channel_config
        # Need to call the base class constructor last

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers")
        for config in self.channel_config:
            spec.output(config["name"])


    def compute(self, op_input, op_output, context):
        message = op_input.receive("receivers")
        for config in self.channel_config:
            channel_name = config["name"]
            di_tensor = hs.as_tensor(cp.asarray(message.get(channel_name)))
            op_output.emit({"": di_tensor}, channel_name)


class StreamMergerOp(Operator):

    def __init__(
            self,
            fragment: Any,
            input_names: Any,
            message_name: str,
            fuse_buffers: bool,
            *args,
            **kwargs,
    ):
        self.input_names = input_names
        self.fuse_buffers = fuse_buffers
        self.message_name = message_name
        self.ctx_service = None
        # Need to call the base class constructor last

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for name in self.input_names:
            spec.input(name)
        spec.output("output")
        self.ctx_service = self.service(DeviceContextService)


    def compute(self, op_input, op_output, context):
        all_messages = []
        for name in self.input_names:
            message = op_input.receive(name)
            all_messages.append((name, cp.asarray(message.get(name))))

        if self.fuse_buffers:
            fused_buffer = cp.concatenate((m[1] for m in all_messages))
            log.info("Fused buffer to {}".format(fused_buffer.shape))
            di_tensor = hs.as_tensor(fused_buffer)
            op_output.emit({self.message_name: di_tensor}, "output")
        else:
            out_message = dict()
            for name, buffer in all_messages:
                camera_name = self.ctx_service.get_camera_name_from_port_name(name)
                message_name = f"{camera_name}_{self.message_name}"
                di_tensor = hs.as_tensor(buffer)
                out_message[message_name] = di_tensor
            op_output.emit(out_message, "output")


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
        # print("received positions and texcoords")
        import pdb;pdb.set_trace()

class App(hs.core.Application):
    def compose(self):

        # Add your operators here
        print("Starting TCN Shm Receiver")

        shm_config = self.kwargs("shared_memory")
        debug_output_config = self.kwargs("debug_output")

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

        log.info(f"create subscriber op {stream_name}")
        subscriber_op = ShmSubscriberOp(self, shm_receiver, stream_name, channels_config, cycle_time_ms, name="shm_subscriber")

        depth_streams_config = []
        color_streams_config = []
        for channel in channels_config["ports"]:
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)
            elif channel["status"]["portType"] == "colorimage":
                color_streams_config.append(channel)

        log.info("create stream_splitter op")
        split_op = StreamSplitterOp(self, depth_streams_config, name="stream_splitter")
        self.add_flow(subscriber_op, split_op, {("depth_outputs", "receivers")})


        points_visualizer = None
        if debug_output_config.get("enable_pointcloud", False):
            log.info("create pointclouds debug-view")
            # configure pointcloud debug viewer
            # num_pointclouds = len(camera_names)
            # # Determine grid size (e.g., 2 for 2x2, 3 for 3x3)
            # pointclouds_grid_size = int(np.ceil(np.sqrt(num_pointclouds))) if num_pointclouds > 0 else 1
            # pointclouds_tile_size = 1.0 / pointclouds_grid_size

            pointclouds_output_specs = []
            # for i, camera_name in enumerate(camera_names):
            #     # Compute row and column index
            #     row = i // pointclouds_grid_size
            #     col = i % pointclouds_grid_size
            #
            #     # Compute normalized offsets (0.0 to 1.0)
            #     # Note: Holoviz usually uses (x, y) for offsets
            #     offset_x = col * pointclouds_tile_size
            #     offset_y = row * pointclouds_tile_size
            #
            #     spec = HolovizOp.InputSpec(camera_name, HolovizOp.InputType.POINTS_3D)
            #     views = []
            #     view = HolovizOp.InputSpec.View()
            #     view.offset_x = offset_x
            #     view.offset_y = offset_y
            #     view.width = pointclouds_tile_size
            #     view.height = pointclouds_tile_size
            #     views.append(view)
            #     spec.views = views
            #     spec.color = [1.0, 0.0, 0.0, 1.0]
            #     pointclouds_output_specs.append(spec)

            # identity_pose = make_rigid_transform(
            #     hs.as_tensor(np.asarray([0.,0.,0.])),
            #     hs.as_tensor(np.asarray([0.,0.,0.,1.]))
            # )


            bg_spec = HolovizOp.InputSpec("dummy", HolovizOp.InputType.RECTANGLES)
            bg_spec.priority = -1
            bg_spec.color = [0.9, 0.9, 0.9, 1.0]
            pointclouds_output_specs.append(bg_spec)

            spec = HolovizOp.InputSpec("positions", HolovizOp.InputType.POINTS_3D)
            views = []
            view = HolovizOp.InputSpec.View()
            view.offset_x = 0
            view.offset_y = 0
            view.width = 1
            view.height = 1
            views.append(view)
            spec.views = views
            spec.color = [1.0, 0.0, 0.0, 1.0]
            pointclouds_output_specs.append(spec)

            points_visualizer = HolovizOp(
                self,
                name="points_visualizer",
                tensors=pointclouds_output_specs,
                allocator=CudaStreamPool(
                    self,
                    name="cuda_stream",
                    dev_id=0,
                    stream_flags=0,
                    stream_priority=0,
                    reserved_size=1,
                    max_size=len(camera_names),
                ),
                **self.kwargs("points_holoviz"),
            )

        log.info("define per depthimage processing pipeline")
        sink_ops = []

        merge_connections = []
        for channel in depth_streams_config:
            channel_name = channel["name"]
            camera_name = ctx_service.get_camera_name_from_port_name(channel_name)
            color_params = ctx_service.get_color_camera_model(camera_name)

            log.info(f"create xylookuptable source: {camera_name}")
            xylt_op = XYLookupTableSourceOp(self,
                                            CountCondition(self, count=1),
                                            name=f"xylt_loader_{camera_name}",
                                            camera_name=camera_name,
                                            )
            sink_ops.append(xylt_op)

            log.info(f"create backprojection: {camera_name}")
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
                out_tensor_name=camera_name,
                enable_positions=True,
                # points visualizer does not consume texcoords
                enable_texcoords=points_visualizer is None,
                enable_depth_float=False,
                name=f"{camera_name}_backprojection",
                )
            sink_ops.append(bp_op)

            self.add_flow(split_op, bp_op, {
                (channel_name, "depth_image"),
            })
            self.add_flow(xylt_op, bp_op, {
                ("xy_table", "xy_table")
            })

            merge_connections.append((bp_op, {("positions", f"{camera_name}_positions")}))

            # debug view..
            if points_visualizer is not None:
                self.add_flow(bp_op, points_visualizer, {("positions", "receivers")})
            else:
                sink_op = PointCloudDummySinkOp(self, name=f"{camera_name}_sink")
                self.add_flow(bp_op, sink_op, {
                    ("positions", "positions"),
                    ("texcoords", "texcoords"),
                    })
                sink_ops.append(sink_op)

        # merge Pointclouds
        merge_inputs = list({list(v[1])[0][1] for v in merge_connections})
        log.info(f"Merge Position Streams: {merge_inputs}")
        merge_op = StreamMergerOp(self, merge_inputs, "positions", True)
        for op, conn in merge_connections:
            self.add_flow(op, merge_op, conn)

        self.add_flow(merge_op, points_visualizer, {("output", "receivers")})


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

        if debug_output_config.get("enable_colorimage", False):
            log.info("create color visualizer")
            color_visualizer = HolovizOp(
                self,
                name="color_visualizer",
                allocator=CudaStreamPool(
                    self,
                    name="cuda_stream",
                    dev_id=0,
                    stream_flags=0,
                    stream_priority=0,
                    reserved_size=1,
                    max_size=len(color_streams_config),
                ),
                **self.kwargs("color_holoviz"),
            )

            self.add_flow(subscriber_op, color_visualizer, {("color_outputs", "receivers")})
            self.add_flow(subscriber_op, color_visualizer, {("color_output_specs", "input_specs")})
        else:
            # need a consumer for color_images
            ci_sink = DummySinkOp(self, name="color_image_sink")
            self.add_flow(subscriber_op, ci_sink, {("color_outputs", "input")})

        if debug_output_config.get("enable_depthimage", False):
            log.info("create depth visualizer")
            depth_visualizer = HolovizOp(
                self,
                name="depth_visualizer",
                allocator=CudaStreamPool(
                    self,
                    name="cuda_stream",
                    dev_id=0,
                    stream_flags=0,
                    stream_priority=0,
                    reserved_size=1,
                    max_size=len(depth_streams_config),
                ),
                **self.kwargs("depth_holoviz"),
            )

            self.add_flow(subscriber_op, depth_visualizer, {("depth_outputs", "receivers")})
            self.add_flow(subscriber_op, depth_visualizer, {("depth_output_specs", "input_specs")})


def main(config_file=None):
    # make configurable or use holoscan debug level here too
    logging.basicConfig(level=logging.INFO)
    set_log_level(LogLevel.INFO)
    iox2.set_log_level(iox2.LogLevel.Warn)

    app = App()
    app.config(config_file)

    scheduler = EventBasedScheduler(app, worker_thread_number=24, name="ebs")
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
