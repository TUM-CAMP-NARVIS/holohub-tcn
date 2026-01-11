#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import threading
import iceoryx2 as iox2

import numpy as np
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
from holoscan.resources import RMMAllocator
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler

from holoscan.pose_tree import PoseTreeManager, SO3

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
        channel_semantic_types = {
            v['name']: SemanticType(v['status']['bufferInfo']['semanticType']) for v in channels_config['ports']
        }

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
            pointclouds_output_specs = []
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

        weights_visualizer = None
        if debug_output_config.get("enable_weights", False):
            log.info("create weights debug-view")
            # configure weights debug viewer
            num_weights = len(camera_names)
            # Determine grid size (e.g., 2 for 2x2, 3 for 3x3)
            weights_grid_size = int(np.ceil(np.sqrt(num_weights))) if num_weights > 0 else 1
            weights_tile_size = 1.0 / weights_grid_size

            weights_output_specs = []
            for i, camera_name in enumerate(camera_names):
                # Compute row and column index
                row = i // weights_grid_size
                col = i % weights_grid_size

                # Compute normalized offsets (0.0 to 1.0)
                # Note: Holoviz usually uses (x, y) for offsets
                offset_x = col * weights_tile_size
                offset_y = row * weights_tile_size

                spec = HolovizOp.InputSpec(camera_name, HolovizOp.InputType.COLOR)
                views = []
                view = HolovizOp.InputSpec.View()
                view.offset_x = offset_x
                view.offset_y = offset_y
                view.width = weights_tile_size
                view.height = weights_tile_size
                views.append(view)
                spec.views = views
                #spec.color = [1.0, 0.0, 0.0, 1.0]
                weights_output_specs.append(spec)

            weights_visualizer = HolovizOp(
                self,
                name="weights_visualizer",
                tensors=weights_output_specs,
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("weights_holoviz"),
            )

        warped_color_visualizer = None
        if debug_output_config.get("enable_warped_color", False):
            log.info("create warped_color debug-view")
            # configure warped_color debug viewer
            num_warped_color = len(camera_names)
            # Determine grid size (e.g., 2 for 2x2, 3 for 3x3)
            warped_color_grid_size = int(np.ceil(np.sqrt(num_warped_color))) if num_warped_color > 0 else 1
            warped_color_tile_size = 1.0 / warped_color_grid_size

            warped_color_output_specs = []
            for i, camera_name in enumerate(camera_names):
                # Compute row and column index
                row = i // warped_color_grid_size
                col = i % warped_color_grid_size

                # Compute normalized offsets (0.0 to 1.0)
                # Note: Holoviz usually uses (x, y) for offsets
                offset_x = col * warped_color_tile_size
                offset_y = row * warped_color_tile_size

                spec = HolovizOp.InputSpec(camera_name, HolovizOp.InputType.COLOR)
                views = []
                view = HolovizOp.InputSpec.View()
                view.offset_x = offset_x
                view.offset_y = offset_y
                view.width = warped_color_tile_size
                view.height = warped_color_tile_size
                views.append(view)
                spec.views = views
                st = channel_semantic_types[f"{camera_name}_colorimage"]
                if st.content_type.get_format_type() == ImageFormatTypes.Rgba:
                    spec.image_format = holoviz._holoviz_str_to_image_format["r8g8b8a8_unorm"]
                elif st.content_type.get_format_type() == ImageFormatTypes.Bgra:
                    spec.image_format = holoviz._holoviz_str_to_image_format["b8g8r8a8_unorm"]
                else:
                    log.warning(f"Unsupported format type: {st.content_type.get_format_type()}")
                warped_color_output_specs.append(spec)

            warped_color_visualizer = HolovizOp(
                self,
                name="warped_color_visualizer",
                tensors=warped_color_output_specs,
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("warped_color_holoviz"),
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
                    cuda_stream_pool,
                    allocator=device_memory_pool,
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

            if camera_streams_config.get("enable_background_substract", False):
                dimd_op = DepthImageMaxDistanceOp(
                    self,
                    allocator=device_memory_pool,
                    cuda_stream_pool=cuda_stream_pool,
                    name=f"{camera_name}_max_distance",
                )
                sink_ops.append(dimd_op)
                self.add_flow(prev_op, dimd_op, {
                    (prev_output, "input"),
                })
                difgbg_op = DepthImageForegroundBackgroundMaskOp(
                    self,
                    allocator=device_memory_pool,
                    enable_foreground=True,
                    enable_background=False,
                    cuda_stream_pool=cuda_stream_pool,
                    name=f"{camera_name}_fg_bg_mask",
                    **self.kwargs("depthimage_fgbg_mask")
                )
                sink_ops.append(difgbg_op)
                self.add_flow(split_op, difgbg_op, {
                    (channel_name, "depth_image"),
                })
                self.add_flow(dimd_op, difgbg_op, {
                    ("output", "background_image"),
                })
                diam_op = DepthImageApplyMaskOp(
                    self,
                    allocator=device_memory_pool,
                    cuda_stream_pool=cuda_stream_pool,
                    name=f"{camera_name}_apply_mask",
                    **self.kwargs("depthimage_apply_mask")
                )
                sink_ops.append(diam_op)
                self.add_flow(split_op, diam_op, {
                    (channel_name, "depth_image"),
                })
                self.add_flow(difgbg_op, diam_op, {
                    ("foreground_mask", "mask_image"),
                })

                prev_op = diam_op
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
                cuda_stream_pool,
                allocator=device_memory_pool,
                color_image_width=color_params.dimensions.x,
                color_image_height=color_params.dimensions.y,
                color_params=ctx_service.get_color_camera_model(camera_name),
                depth_extrinsics=ctx_service.get_depth_extrinsics(camera_name),
                depth_to_color=ctx_service.get_color_to_depth_inv(camera_name),
                # out_tensor_name=camera_name,
                in_tensor_name="",
                out_tensor_name="output",
                enable_positions=True,
                enable_texcoords=camera_streams_config.get("enable_warp_colorimage", False),
                enable_depth_float=False,
                cuda_device_ordinal=cuda_device_id,
                name=f"{camera_name}_backprojection",
                **self.kwargs("depthimage_backprojection")
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

            if camera_streams_config.get("enable_compute_weights", False):
                log.info(f"create compute weights: {camera_name}")
                cp_op = TcnDepthImageWeightsOp(
                    self,
                    cuda_stream_pool,
                    allocator=device_memory_pool,
                    # out_tensor_name=camera_name,
                    in_tensor_name="",
                    out_tensor_name=camera_name,
                    cuda_device_ordinal=cuda_device_id,
                    name=f"{camera_name}_weights",
                    **self.kwargs("depthimage_weights")
                )
                sink_ops.append(cp_op)

                self.add_flow(prev_op, cp_op, {
                    (prev_output, "depth_image"),
                })
                self.add_flow(xylt_op, cp_op, {
                    ("xy_table", "xy_table")
                })
                # debug view..
                if weights_visualizer is not None:
                    self.add_flow(cp_op, weights_visualizer, {("output", "receivers")})
                else:
                    sink_op = DummySinkOp(self, name=f"{camera_name}_sink")
                    self.add_flow(cp_op, sink_op, {
                        ("output", "input"),
                        })
                    sink_ops.append(sink_op)

            if camera_streams_config.get("enable_warp_colorimage", False):
                wci_op = TcnTextureSamplerOp(
                    self,
                    cuda_stream_pool,
                    allocator=device_memory_pool,
                    in_color_tensor_name=f"{camera_name}_colorimage",
                    in_texcoord_tensor_name="output",
                    out_tensor_name=camera_name,
                    cuda_device_ordinal=cuda_device_id,
                    name=f"{camera_name}_warp_colorimage",
                )
                sink_ops.append(wci_op)
                self.add_flow(bp_op, wci_op, {
                    ("texcoords", "texcoords"),
                })
                self.add_flow(subscriber_op, wci_op, {
                    ("color_outputs", "color_image"),
                })
                # debug view..
                if warped_color_visualizer is not None:
                    self.add_flow(wci_op, warped_color_visualizer, {("output", "receivers")})
                else:
                    sink_op = DummySinkOp(self, name=f"{camera_name}_warped_color_sink")
                    self.add_flow(wci_op, sink_op, {
                        ("output", "input"),
                    })
                    sink_ops.append(sink_op)


        # merge Pointclouds
        merge_inputs = list({list(v[1])[0][1] for v in position_merge_connections})
        log.info(f"Merge Position Streams: {merge_inputs}")
        position_merge_op = StreamMergerOp(self, cuda_stream_pool, merge_inputs, "output", "positions", True, name="point_fusion")
        for op, conn in position_merge_connections:
            self.add_flow(op, position_merge_op, conn)

        flt_op = FlattenTensorOp(
            self,
            cuda_stream_pool,
            message_name="positions",
            allocator=device_memory_pool,
            name=f"flatten_pointcloud",
        )
        self.add_flow(position_merge_op, flt_op, {("output", "input")})
        self.add_flow(flt_op, points_visualizer, {("output", "receivers")})


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


        # rpc_service_name = shm_config.get("parameter_rpc_name", "holohub")
        # self.rpc_server_ = ParameterRpcServer(node, f"{rpc_service_name}/PARAMETER_RPC/Components", self)
        # self.rpc_server_.update_schema_from_fragment()
        # self.rpc_server_thread_ = threading.Thread(target=self.rpc_server_.serve_blocking)
        # self.rpc_server_thread_.start()


def main(config_file=None):
    # make configurable or use holoscan debug level here too
    configure_debug = False

    if configure_debug:
        logging.basicConfig(level=logging.DEBUG)
        set_log_level(LogLevel.DEBUG)
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

    # if hasattr(app, "rpc_server_"):
    #     app.rpc_server_.shutdown()
    #     app.rpc_server_thread_.join()


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
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_shm_receiver.yaml")
    else:
        config_file = args.config

    main(config_file=config_file)
