import os
import pathlib
from typing import Any, List
import logging

from holoscan.core import Subgraph
from holoscan.conditions import CountCondition
from holoscan.pose_tree import PoseTreeManager, SO3, Pose3
from holoscan.resources import CudaStreamPool
from holoscan.resources import BlockMemoryPool, MemoryStorageType

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

log = logging.getLogger(__name__)

# @todo: should go to tcn_util
def convert_rigid_transform_to_pose3(input: RigidTransform) -> Pose3:
    log.debug(f"RigidTransform translation: {input.translation} rotation: {input.rotation}")
    r = SO3.from_quaternion(input.rotation)
    return Pose3(r, input.translation)


class ShmSimpleBackprojectionSubgraph(Subgraph):
    """Subgraph containing the shm-receiver and backprojection pipeline."""

    def __init__(self, fragment, name, fuse_buffers=False):
        self.fuse_buffers = fuse_buffers
        super().__init__(fragment, name)

    def compose(self):
        log.info("Compose subgraph: ShmSimpleBackprojection")
        app = self.fragment.application

        # @todo: do not use app.kwargs directly, but pass the relevant dictionary or subtree to the subgraph explicitly

        # read configuration
        camera_streams_config = app.kwargs("camera_stream_processing")
        cuda_device_id = camera_streams_config.get("device_id", 0)
        block_memory_buffer_size = camera_streams_config.get("buffer_size", 8)

        shm_config = app.kwargs("shared_memory")
        shm_stream_name = shm_config.get("stream_name")
        cycle_time_ms = shm_config.get("cycle_time_ms")

        # create SHM Receiver
        log.info("Create SHM Receiver")
        node, shm_receiver = create_shm_subscriber()

        # @todo: for composability, services should be created outside and
        # populated from within the subgraphs

        log.info("Find cameras in shared memory")
        camera_names = shm_receiver.discover_devices()
        device_contexts = {}
        for camera_name in camera_names:
            log.info(f"Retrieving camera_info: {camera_name}")
            ctx = shm_receiver.retrieve_device_context(camera_name)
            if ctx is not None:
                device_contexts[camera_name] = ctx

        # Register the ctx_service with the fragment
        log.info("Register DeviceContextService")
        ctx_service = DeviceContextService(device_contexts)
        app.register_service(ctx_service)

        # # create pose tree service for fragment
        log.info("Register PoseTreeManager")
        pose_tree_config = app.kwargs("pose_tree_config")  # see pose_tree_basic.yaml
        pts = PoseTreeManager(
            self,
            name="pose_tree_manager",
            **pose_tree_config,
        )
        app.register_service(pts)

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
                    cuda_stream_pool,
                    allocator=device_memory_pool,
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
                                            name=f"xylt_loader_{camera_name}",
                                            camera_name=camera_name,
                                            )

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
        position_merge_op = StreamMergerOp(self, cuda_stream_pool, position_merge_inputs,
                                           "output", "positions",
                                           self.fuse_buffers, name="point_fusion")
        for op, conn in position_merge_connections:
            self.add_flow(op, position_merge_op, conn)

        texcoord_merge_inputs = list({list(v[1])[0][1] for v in texcoord_merge_connections})
        log.info(f"Merge Texcoord Streams: {texcoord_merge_inputs}")
        texcoord_merge_op = StreamMergerOp(self, cuda_stream_pool, texcoord_merge_inputs,
                                           "output", "texcoords",
                                           self.fuse_buffers, name="texcoord_fusion")
        for op, conn in texcoord_merge_connections:
            self.add_flow(op, texcoord_merge_op, conn)

        # Expose the relevant ports
        self.add_output_interface_port("color_outputs", subscriber_op, "color_outputs")
        self.add_output_interface_port("depth_outputs", subscriber_op, "depth_outputs")
        self.add_output_interface_port("position_outputs", position_merge_op, "output")
        self.add_output_interface_port("texcoord_outputs", texcoord_merge_op, "output")