import os
import pathlib
from typing import Any, List
import logging
import numpy as np
import math

from holoscan.core import Subgraph
from holoscan.conditions import AsynchronousCondition, CountCondition
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3
from holoscan.resources import CudaStreamPool
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from holoscan.operators import HolovizOp
from holoscan.operators import holoviz

from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_temporal_filter import TcnDepthImageTemporalFilterOp
from holohub.tcn_shm_subscriber import TcnShmSubscriberOp as ShmSubscriberOp
from holohub.tcn_shm_subscriber._tcn_shm_subscriber import ShmSynchronizedBufferReceiver
from holohub.tcn_device_context import XYLookupTableSourceOp
from holohub.tcn_device_context._tcn_device_context import DeviceContextService
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
    RigidTransform, CameraParameters, make_rigid_transform
from holohub.tcn_stream_splitter import TcnStreamSplitterOp as StreamSplitterOp
from holohub.tcn_stream_merger import TcnStreamMergerOp as StreamMergerOp
from holohub.tcn_shm_subscriber._tcn_shm_subscriber import discover_shm

from tcnart.core.semantic_type import SemanticType
from tcnart.core.semantic_type.model import ImageFormatTypes

log = logging.getLogger(__name__)

# @todo: should go to tcn_util
def convert_rigid_transform_to_pose3(input: RigidTransform) -> Pose3:
    log.debug(f"RigidTransform translation: {input.translation} rotation: {input.rotation}")
    r = SO3.from_quaternion(input.rotation)
    return Pose3(r, input.translation)


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



class ShmConnection:

    def __init__(self, stream_name: str, cycle_time_ms: int):
        self.shm_stream_name = stream_name
        self.cycle_time_ms = cycle_time_ms
        self.receiver_config = None

    def connect_shm(self):
        # create SHM Receiver
        log.info("Discover contents in shared memory")
        try:
            self.receiver_config = discover_shm(self.shm_stream_name)
            return True
        except Exception as e:
            log.error(e)
            return False

    def get_num_streams(self):
        if self.receiver_config is None:
            raise RuntimeError("need to connect to shm first")
        channels_config = self.receiver_config["channels_config"]
        return len(channels_config["ports"])

    def get_camera_names(self):
        if self.receiver_config is None:
            raise RuntimeError("need to connect to shm first")
        channels_config = self.receiver_config["channels_config"]
        return channels_config["camera_names"]

    def get_device_contexts(self):
        if self.receiver_config is None:
            raise RuntimeError("need to connect to shm first")
        channels_config = self.receiver_config["channels_config"]
        return channels_config["device_contexts"]

    def get_max_tensor_size(self, fuse_buffers=False):
        if self.receiver_config is None:
            raise RuntimeError("need to connect to shm first")
        channels_config = self.receiver_config["channels_config"]

        depth_streams_config = []
        color_streams_config = []
        max_frame_size = 0

        for channel in channels_config["ports"]:
            max_frame_size = max(max_frame_size, channel["status"]["bufferInfo"]["frameSize"])
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)
            elif channel["status"]["portType"] == "colorimage":
                color_streams_config.append(channel)

        fused_positions_size = 0
        if fuse_buffers:
            for ch in depth_streams_config:
                fused_positions_size += ch["status"]["bufferInfo"]["width"] * ch["status"]["bufferInfo"]["height"] * 3 * 4  # sizeof(float) .. maybe use struct module here?

        max_frame_size = max(max_frame_size, fused_positions_size)
        return max_frame_size


    def get_output_specs(self):
        if self.receiver_config is None:
            raise RuntimeError("need to connect to shm first")
        channels_config = self.receiver_config["channels_config"]

        channel_semantic_types = {
            v['name']: SemanticType(v['status']['bufferInfo']['semanticType']) for v in channels_config['ports']
        }

        depth_streams_config = []
        color_streams_config = []

        for channel in channels_config["ports"]:
            if channel["status"]["portType"] == "depthimage":
                depth_streams_config.append(channel)
            elif channel["status"]["portType"] == "colorimage":
                color_streams_config.append(channel)

        color_output_specs = create_tiled_input_specs(
            [channel["name"] for channel in color_streams_config],
            semantic_types=channel_semantic_types,
        )
        depth_output_specs = create_tiled_input_specs(
            [channel["name"] for channel in depth_streams_config],
        )
        return {"color_output_specs": color_output_specs,
                "depth_output_specs": depth_output_specs
                }


class ShmSimpleBackprojectionSubgraph(Subgraph):
    """Subgraph containing the shm-receiver and backprojection pipeline."""

    def __init__(self, fragment, name,
                 stream_pool = None,
                 allocator = None,
                 shm_connection: ShmConnection = None,
                 fuse_buffers=False,
                 use_extrinsics=True):
        self.cuda_stream_pool = stream_pool
        self.allocator = allocator
        self.shm_connection = shm_connection
        self.fuse_buffers = fuse_buffers
        self.use_extrinsics = use_extrinsics
        super().__init__(fragment, name)



    def compose(self):
        log.info("Compose subgraph: ShmSimpleBackprojection")
        app = self.fragment.application

        if self.shm_connection is None:
            raise RuntimeError("needs a shm_connection")
        if self.shm_connection.receiver_config is None:
            raise RuntimeError("shm connection needs to be open")

        receiver_config = self.shm_connection.receiver_config

        shm_stream_name = self.shm_connection.shm_stream_name
        cycle_time_ms = self.shm_connection.cycle_time_ms

        # read configuration
        camera_streams_config = app.kwargs("camera_stream_processing")
        cuda_device_id = camera_streams_config.get("device_id", 0)

        shm_receiver = receiver_config["receiver"]
        camera_names = receiver_config["camera_names"]
        device_contexts = receiver_config["device_contexts"]

        # Register the ctx_service with the fragment
        log.info("Register DeviceContextService")
        ctx_service = DeviceContextService.create(device_contexts)
        app.register_service(ctx_service)

        # # create pose tree service for fragment
        log.info("Register PoseTreeManager")
        pts = PoseTreeManager(
            self,
            name="pose_tree_manager",
            **app.kwargs("pose_tree_config"),
        )
        app.register_service(pts)

        pose_tree_config = {"frames": [], "edges": []}
        pose_tree_config["frames"].append("world_origin")
        for name in camera_names:
            log.info(f"Create Reference frames for camera {name}")
            # the frames are intentionally called like the streams to simplify lookup
            depth_channel_name = f"{name}_depthimage"
            color_channel_name = f"{name}_colorimage"
            pose_tree_config["frames"].append(depth_channel_name)
            pose_tree_config["frames"].append(color_channel_name)
            pose_tree_config["edges"].append(("world_origin", depth_channel_name, convert_rigid_transform_to_pose3(ctx_service.get_depth_extrinsics(name))))
            pose_tree_config["edges"].append((color_channel_name, depth_channel_name, convert_rigid_transform_to_pose3(ctx_service.get_color_to_depth(name))))

        # # # configure pose-tree
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
        #     pts.tree.set("world_origin", depth_channel_name, 0,
        #                  convert_rigid_transform_to_pose3(ctx_service.get_depth_extrinsics(name)))
        #     pts.tree.set(color_channel_name, depth_channel_name, 0,
        #                  convert_rigid_transform_to_pose3(ctx_service.get_color_to_depth(name)))

        channels_config = receiver_config["channels_config"]

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


        log.info(f"create subscriber op {shm_stream_name}")
        shm_async_condition = AsynchronousCondition(self, name="shm_async_condition")
        subscriber_op = ShmSubscriberOp(self, self.cuda_stream_pool,
                                        allocator=self.allocator,
                                        async_condition=shm_async_condition,
                                        receiver=shm_receiver,
                                        stream_name=shm_stream_name,
                                        cycle_time_ms=cycle_time_ms,
                                        name="shm_subscriber")

        log.info("create stream_splitter op")
        split_op = StreamSplitterOp(self, self.cuda_stream_pool,
                                    channel_names=[v["name"] for v in depth_streams_config],
                                    name="stream_splitter")
        self.add_flow(subscriber_op, split_op, {("depth_outputs", "receivers")})

        log.info("define per depthimage processing pipeline")

        position_merge_connections = []
        texcoord_merge_connections = []
        for channel in depth_streams_config:
            channel_name = channel["name"]
            camera_name = ctx_service.get_camera_name_from_port_name(channel_name)
            color_params = ctx_service.get_color_camera_model(camera_name)

            prev_op = split_op
            prev_output = channel_name
            if camera_streams_config.get("enable_temporal_filter", False):
                ditf_op = TcnDepthImageTemporalFilterOp(
                    self,
                    self.cuda_stream_pool,
                    allocator=self.allocator,
                    in_tensor_name="",
                    out_tensor_name="",
                    cuda_device_ordinal=cuda_device_id,
                    name=f"{camera_name}_temporal_filter",
                    **app.kwargs("depthimage_temporal_filter"),
                )
                self.add_flow(split_op, ditf_op, {
                    (channel_name, "input"),
                })
                prev_op = ditf_op
                prev_output = "output"

            log.info(f"create xylookuptable source: {camera_name}")
            xylt_op = XYLookupTableSourceOp(self,
                                            CountCondition(self, count=1),
                                            allocator=self.allocator,
                                            name=f"xylt_loader_{camera_name}",
                                            camera_name=camera_name,
                                            )
            xylt_op.set_device_context_service(ctx_service)

            log.info(f"create backprojection: {camera_name}")
            bp_op = TcnDepthImageBackprojectionOp(
                self,
                self.cuda_stream_pool,
                allocator=self.allocator,
                color_image_width=color_params.dimensions.x,
                color_image_height=color_params.dimensions.y,
                color_params=ctx_service.get_color_camera_model(camera_name),
                depth_extrinsics=self.use_extrinsics and ctx_service.get_depth_extrinsics(camera_name) or make_rigid_transform(np.zeros(3), np.asarray([0., 0., 0., 1.])),
                depth_to_color=ctx_service.get_color_to_depth_inv(camera_name),
                in_tensor_name="",
                out_tensor_name="output",
                enable_positions=True,
                enable_texcoords=True,
                enable_depth_float=False,
                cuda_device_ordinal=cuda_device_id,
                name=f"{camera_name}_backprojection",
                **app.kwargs("depthimage_backprojection")
                )

            self.add_flow(prev_op, bp_op, {
                (prev_output, "depth_image"),
            })
            self.add_flow(xylt_op, bp_op, {
                ("xy_table", "xy_table")
            })

            position_merge_connections.append((bp_op, {("positions", f"{camera_name}_positions")}))
            texcoord_merge_connections.append((bp_op, {("texcoords", f"{camera_name}_texcoords")}))

        # merge Pointclouds
        position_merge_inputs = list({list(v[1])[0][1] for v in position_merge_connections})
        log.info(f"Merge Position Streams: {position_merge_inputs}")
        position_merge_op = StreamMergerOp(self, self.cuda_stream_pool,
                                           input_port_names=position_merge_inputs,
                                           input_message_name="output",
                                           output_message_name="positions",
                                           fuse_buffers=self.fuse_buffers,
                                           name="point_fusion")
        for op, conn in position_merge_connections:
            self.add_flow(op, position_merge_op, conn)

        texcoord_merge_inputs = list({list(v[1])[0][1] for v in texcoord_merge_connections})
        log.info(f"Merge Texcoord Streams: {texcoord_merge_inputs}")
        texcoord_merge_op = StreamMergerOp(self, self.cuda_stream_pool,
                                           input_port_names=texcoord_merge_inputs,
                                           input_message_name="output",
                                           output_message_name="texcoords",
                                           fuse_buffers=self.fuse_buffers,
                                           name="texcoord_fusion")
        for op, conn in texcoord_merge_connections:
            self.add_flow(op, texcoord_merge_op, conn)

        # Expose the relevant ports
        self.add_output_interface_port("color_outputs", subscriber_op, "color_outputs")
        self.add_output_interface_port("depth_outputs", subscriber_op, "depth_outputs")
        self.add_output_interface_port("position_outputs", position_merge_op, "output")
        self.add_output_interface_port("texcoord_outputs", texcoord_merge_op, "output")
