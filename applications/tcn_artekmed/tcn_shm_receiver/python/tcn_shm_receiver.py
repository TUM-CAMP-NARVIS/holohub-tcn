#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import iceoryx2 as iox2

import numpy as np
import holoscan as hs
from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_temporal_filter import TcnDepthImageTemporalFilterOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
    RigidTransform, CameraParameters, make_rigid_transform
from operators.tcn_artekmed.tcn_shm_io import ShmSubscriberOp, DeviceContextService, XYLookupTableSourceOp, create_shm_subscriber
from operators.tcn_artekmed.tcn_util import StreamSplitterOp, StreamMergerOp

from holoscan.conditions import CountCondition
from holoscan.core import Operator, OperatorSpec, Tracker
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp
from holoscan.operators import holoviz
from holoscan.pose_tree import Pose3
from holoscan.resources import CudaStreamPool
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from holoscan.resources import RMMAllocator
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler

from holoscan.pose_tree import PoseTreeManager, SO3

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
        pose_tree_config = self.kwargs("pose_tree_config")  # see pose_tree_basic.yaml

        pts = PoseTreeManager(
            self,
            name="pose_tree_manager",
            **pose_tree_config,
        )
        self.register_service(pts)

        # # configure pose-tree
        pts.tree.create_frame("world_origin")
        for name in camera_names:
            log.info(f"Create Reference frames for camera {name}")
            # the frames are intentionally called like the streams to simplify lookup
            depth_channel_name = f"{name}_depthimage"
            color_channel_name = f"{name}_colorimage"
            pts.tree.create_frame(depth_channel_name)
            pts.tree.create_frame(color_channel_name)
            # connect the frames
            pts.tree.create_edges("world_origin", depth_channel_name)
            pts.tree.create_edges(depth_channel_name, color_channel_name)
            # set transforms
            pts.tree.set("world_origin", depth_channel_name, 0,
                         convert_rigid_transform_to_pose3(ctx_service.get_depth_extrinsics(name)))
            pts.tree.set(color_channel_name, depth_channel_name, 0,
                         convert_rigid_transform_to_pose3(ctx_service.get_color_to_depth(name)))

        log.info(f"Retrieve channel config for stream: {shm_stream_name}")
        channels_config = shm_receiver.retrieve_channel_config(shm_stream_name)

        depth_streams_config = []
        color_streams_config = []
        max_frame_size = 0
        num_channels = len(channels_config["ports"])

        for channel in channels_config["ports"]:
            max_frame_size = max(max_frame_size, channel["status"]["bufferInfo"]["frameSize"])
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)
            elif channel["status"]["portType"] == "colorimage":
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
        subscriber_op = ShmSubscriberOp(self, cuda_stream_pool, device_memory_pool, shm_receiver, shm_stream_name, channels_config, cycle_time_ms, name="shm_subscriber")

        log.info("create stream_splitter op")
        split_op = StreamSplitterOp(self, cuda_stream_pool, [v["name"] for v in depth_streams_config], name="stream_splitter")
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
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("points_holoviz"),
            )

        log.info("define per depthimage processing pipeline")
        sink_ops = []

        position_merge_connections = []
        texcoords_merge_connections = []
        for channel in depth_streams_config:
            channel_name = channel["name"]
            camera_name = ctx_service.get_camera_name_from_port_name(channel_name)
            color_params = ctx_service.get_color_camera_model(camera_name)

            prev_op = split_op
            prev_output = channel_name
            if camera_streams_config.get("enable_temporal_filter", False):
                ditf_op = TcnDepthImageTemporalFilterOp(
                    self,
                    allocator=device_memory_pool,
                    cuda_stream_pool=cuda_stream_pool,
                    in_tensor_name="",
                    out_tensor_name="",
                    cuda_device_ordinal=cuda_device_id,
                    name=f"{camera_name}_temporal_filter",
                    **self.kwargs("depthimage_temporal_filter"),
                )
                sink_ops.append(ditf_op)
                self.add_flow(split_op, ditf_op, {
                    (channel_name, "input"),
                })
                prev_op = ditf_op
                prev_output = "output"

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
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                depth_units_per_meter=1000.0,
                near_limit_m=0.01,
                far_limit_m=10.0,
                color_image_width=color_params.dimensions.x,
                color_image_height=color_params.dimensions.y,
                color_params=ctx_service.get_depth_camera_model(camera_name),
                depth_extrinsics=ctx_service.get_depth_extrinsics(camera_name),
                color_to_depth=ctx_service.get_color_to_depth(camera_name),
                # out_tensor_name=camera_name,
                in_tensor_name="",
                out_tensor_name="output",
                enable_positions=True,
                # points visualizer does not consume texcoords
                enable_texcoords=points_visualizer is None,
                enable_depth_float=False,
                cuda_device_ordinal=cuda_device_id,
                name=f"{camera_name}_backprojection",
                )
            sink_ops.append(bp_op)

            self.add_flow(prev_op, bp_op, {
                (prev_output, "depth_image"),
            })
            self.add_flow(xylt_op, bp_op, {
                ("xy_table", "xy_table")
            })

            position_merge_connections.append((bp_op, {("positions", f"{camera_name}_positions")}))
            texcoords_merge_connections.append((bp_op, {("texcoords", f"{camera_name}_texcoords")}))

            # debug view..
            # if points_visualizer is not None:
            #     self.add_flow(bp_op, points_visualizer, {("positions", "receivers")})
            # else:
            #     sink_op = PointCloudDummySinkOp(self, name=f"{camera_name}_sink")
            #     self.add_flow(bp_op, sink_op, {
            #         ("positions", "positions"),
            #         ("texcoords", "texcoords"),
            #         })
            #     sink_ops.append(sink_op)

        # merge Pointclouds
        merge_inputs = list({list(v[1])[0][1] for v in position_merge_connections})
        log.info(f"Merge Position Streams: {merge_inputs}")
        position_merge_op = StreamMergerOp(self, cuda_stream_pool, merge_inputs, "output", "positions", True, name="point_fusion")
        for op, conn in position_merge_connections:
            self.add_flow(op, position_merge_op, conn)
        self.add_flow(position_merge_op, points_visualizer, {("output", "receivers")})


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
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
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
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("depth_holoviz"),
            )

            self.add_flow(subscriber_op, depth_visualizer, {("depth_outputs", "receivers")})
            self.add_flow(subscriber_op, depth_visualizer, {("depth_output_specs", "input_specs")})


def main(config_file=None):
    # make configurable or use holoscan debug level here too
    configure_debug = False

    if configure_debug:
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.TRACE)
        iox2.set_log_level(iox2.LogLevel.Warn)
    else:
        logging.basicConfig(level=logging.INFO)
        set_log_level(LogLevel.INFO)
        iox2.set_log_level(iox2.LogLevel.Warn)

    app = App()
    app.config(config_file)

    if configure_debug:
        scheduler = GreedyScheduler(app, name="gs", stop_on_deadlock=True)
    else:
        scheduler = EventBasedScheduler(app, worker_thread_number=24, name="ebs")

    app.scheduler(scheduler)

    with Tracker(app,
                 num_start_messages_to_skip=15,
                 num_last_messages_to_discard=15) as tracker:
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
