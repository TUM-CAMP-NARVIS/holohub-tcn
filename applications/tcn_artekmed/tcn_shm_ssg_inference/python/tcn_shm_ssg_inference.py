#
# Place the license header here
#
import logging
import os
from argparse import ArgumentParser
import math

import iceoryx2 as iox2

import numpy as np
import holoscan as hs
from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_temporal_filter import TcnDepthImageTemporalFilterOp
from holohub.tcn_depthimage_weights import TcnDepthImageWeightsOp
from holohub.tcn_texture_sampler import TcnTextureSamplerOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
    RigidTransform, CameraParameters, make_rigid_transform

from holohub.tcn_shm_subscriber import TcnShmSubscriberOp as ShmSubscriberOp
from holohub.tcn_shm_subscriber._tcn_shm_subscriber import discover_shm
from holohub.tcn_device_context import XYLookupTableSourceOp
from holohub.tcn_device_context._tcn_device_context import DeviceContextService
from holohub.tcn_stream_splitter import TcnStreamSplitterOp as StreamSplitterOp
from holohub.tcn_stream_merger import TcnStreamMergerOp as StreamMergerOp
from holohub.tcn_flatten_tensor import TcnFlattenTensorOp as FlattenTensorOp
from holohub.tcn_depthimage_max_distance import TcnDepthImageMaxDistanceOp
from holohub.tcn_depthimage_fgbg_mask import TcnDepthImageFgbgMaskOp as DepthImageForegroundBackgroundMaskOp
from holohub.tcn_depthimage_apply_mask import TcnDepthImageApplyMaskOp as DepthImageApplyMaskOp
# from operators.tcn_artekmed.tcn_shm_io import (ShmSubscriberOp, DeviceContextService, XYLookupTableSourceOp,
#                                                create_shm_subscriber)
# from operators.tcn_artekmed.tcn_shm_io import ParameterRpcServer
# from operators.tcn_artekmed.tcn_util import (StreamSplitterOp, StreamMergerOp, FlattenTensorOp,
#                                              DepthImageMaxDistanceOp, DepthImageForegroundBackgroundMaskOp,
#                                              DepthImageApplyMaskOp )

from holoscan.conditions import AsynchronousCondition, CountCondition
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
        print("Starting TCN Shm SSG Inference")

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
        channels_config = receiver_config["channels_config"]

        # Register the ctx_service with the fragment
        ctx_service = DeviceContextService.create(device_contexts)
        self.register_service(ctx_service)

        # # create pose tree service for fragment
        # pts = PoseTreeManager(
        #     self,
        #     name="pose_tree_manager",
        #     **self.kwargs("pose_tree_config"),
        # )
        # self.register_service(pts)


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

        color_output_specs = create_tiled_input_specs(
            [channel["name"] for channel in color_streams_config],
            semantic_types=channel_semantic_types,
        )
        depth_output_specs = create_tiled_input_specs(
            [channel["name"] for channel in depth_streams_config],
        )

        fused_positions_size = 0
        for ch in depth_streams_config:
            fused_positions_size += ch["status"]["bufferInfo"]["width"] * ch["status"]["bufferInfo"]["height"] * 3 * 4  # sizeof(float) .. maybe use struct module here?

        max_frame_size = max(max_frame_size, fused_positions_size)

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
                                        cycle_time_ms=10,
                                        name="shm_subscriber")

        log.info("create stream_splitter op")
        split_op = StreamSplitterOp(self, cuda_stream_pool,
                                    channel_names=[v["name"] for v in depth_streams_config],
                                    name="stream_splitter")
        log.debug("Flow: subscriber_op -> split_op (depth_outputs -> receivers)")
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
            weights_output_specs = create_tiled_input_specs(camera_names)

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
            warped_color_output_specs = create_tiled_input_specs(
                camera_names,
                semantic_types=channel_semantic_types,
                semantic_key=lambda camera_name: f"{camera_name}_colorimage",
            )

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
                log.debug(f"Flow: split_op -> ditf_op [{camera_name}_temporal_filter] ({channel_name} -> input)")
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
                log.debug(f"Flow: {prev_op.name} -> dimd_op [{camera_name}_max_distance] ({prev_output} -> input)")
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
                log.debug(f"Flow: split_op -> difgbg_op [{camera_name}_fg_bg_mask] ({channel_name} -> depth_image)")
                self.add_flow(split_op, difgbg_op, {
                    (channel_name, "depth_image"),
                })
                log.debug(f"Flow: dimd_op [{camera_name}_max_distance] -> difgbg_op [{camera_name}_fg_bg_mask] (output -> background_image)")
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
                log.debug(f"Flow: split_op -> diam_op [{camera_name}_apply_mask] ({channel_name} -> depth_image)")
                self.add_flow(split_op, diam_op, {
                    (channel_name, "depth_image"),
                })
                log.debug(f"Flow: difgbg_op [{camera_name}_fg_bg_mask] -> diam_op [{camera_name}_apply_mask] (foreground_mask -> mask_image)")
                self.add_flow(difgbg_op, diam_op, {
                    ("foreground_mask", "mask_image"),
                })

                prev_op = diam_op
                prev_output = "output"


            log.info(f"create xylookuptable source: {camera_name}")
            xylt_count_cond = CountCondition(self, count=1)
            xylt_op = XYLookupTableSourceOp(self,
                                            xylt_count_cond,
                                            allocator=device_memory_pool,
                                            name=f"xylt_loader_{camera_name}",
                                            camera_name=camera_name,
                                            )
            xylt_op.set_device_context_service(ctx_service) # shouldn't this be done via service lookukp?
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

            log.debug(f"Flow: {prev_op.name} -> bp_op [{camera_name}_backprojection] ({prev_output} -> depth_image)")
            self.add_flow(prev_op, bp_op, {
                (prev_output, "depth_image"),
            })
            log.debug(f"Flow: xylt_op [xylt_loader_{camera_name}] -> bp_op [{camera_name}_backprojection] (xy_table -> xy_table)")
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

                log.debug(f"Flow: {prev_op.name} -> cp_op [{camera_name}_weights] ({prev_output} -> depth_image)")
                self.add_flow(prev_op, cp_op, {
                    (prev_output, "depth_image"),
                })
                log.debug(f"Flow: xylt_op [xylt_loader_{camera_name}] -> cp_op [{camera_name}_weights] (xy_table -> xy_table)")
                self.add_flow(xylt_op, cp_op, {
                    ("xy_table", "xy_table")
                })
                # debug view..
                if weights_visualizer is not None:
                    log.debug(f"Flow: cp_op [{camera_name}_weights] -> weights_visualizer (output -> receivers)")
                    self.add_flow(cp_op, weights_visualizer, {("output", "receivers")})
                else:
                    sink_op = DummySinkOp(self, name=f"{camera_name}_sink")
                    log.debug(f"Flow: cp_op [{camera_name}_weights] -> sink_op [{camera_name}_sink] (output -> input)")
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
                log.debug(f"Flow: bp_op [{camera_name}_backprojection] -> wci_op [{camera_name}_warp_colorimage] (texcoords -> texcoords)")
                self.add_flow(bp_op, wci_op, {
                    ("texcoords", "texcoords"),
                })
                log.debug(f"Flow: subscriber_op -> wci_op [{camera_name}_warp_colorimage] (color_outputs -> color_image)")
                self.add_flow(subscriber_op, wci_op, {
                    ("color_outputs", "color_image"),
                })
                # debug view..
                if warped_color_visualizer is not None:
                    log.debug(f"Flow: wci_op [{camera_name}_warp_colorimage] -> warped_color_visualizer (output -> receivers)")
                    self.add_flow(wci_op, warped_color_visualizer, {("output", "receivers")})
                else:
                    sink_op = DummySinkOp(self, name=f"{camera_name}_warped_color_sink")
                    log.debug(f"Flow: wci_op [{camera_name}_warp_colorimage] -> sink_op [{camera_name}_warped_color_sink] (output -> input)")
                    self.add_flow(wci_op, sink_op, {
                        ("output", "input"),
                    })
                    sink_ops.append(sink_op)


        # merge Pointclouds
        merge_inputs = list({list(v[1])[0][1] for v in position_merge_connections})
        log.info(f"Merge Position Streams: {merge_inputs}")
        position_merge_op = StreamMergerOp(self, cuda_stream_pool,
                                           input_port_names=merge_inputs,
                                           input_message_name="output",
                                           output_message_name="positions",
                                           fuse_buffers=True,
                                           allocator=device_memory_pool,
                                           name="point_fusion")
        for op, conn in position_merge_connections:
            log.debug(f"Flow: {op.name} -> position_merge_op [point_fusion] {conn}")
            self.add_flow(op, position_merge_op, conn)

        flt_op = FlattenTensorOp(
            self,
            message_name="positions",
            allocator=device_memory_pool,
            cuda_stream_pool=cuda_stream_pool,
            name=f"flatten_pointcloud",
        )
        log.debug("Flow: position_merge_op [point_fusion] -> flt_op [flatten_pointcloud] (output -> input)")
        self.add_flow(position_merge_op, flt_op, {("output", "input")})

        if points_visualizer:
            log.debug("Flow: flt_op [flatten_pointcloud] -> points_visualizer (output -> receivers)")
            self.add_flow(flt_op, points_visualizer, {("output", "receivers")})
        else:
            # need a consumer for point_dloucs
            pc_sink = DummySinkOp(self, name="point_cloud_sink")
            log.debug("Flow: subscriber_op -> pc_sink [point_cloud_sink] (color_outputs -> input)")
            self.add_flow(flt_op, pc_sink, {("output", "input")})


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
                tensors=color_output_specs,
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("color_holoviz"),
            )

            log.debug("Flow: subscriber_op -> color_visualizer (color_outputs -> receivers)")
            self.add_flow(subscriber_op, color_visualizer, {("color_outputs", "receivers")})
        else:
            # need a consumer for color_images
            ci_sink = DummySinkOp(self, name="color_image_sink")
            log.debug("Flow: subscriber_op -> ci_sink [color_image_sink] (color_outputs -> input)")
            self.add_flow(subscriber_op, ci_sink, {("color_outputs", "input")})

        if debug_output_config.get("enable_depthimage", False):
            log.info("create depth visualizer")
            depth_visualizer = HolovizOp(
                self,
                name="depth_visualizer",
                tensors=depth_output_specs,
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("depth_holoviz"),
            )

            log.debug("Flow: subscriber_op -> depth_visualizer (depth_outputs -> receivers)")
            self.add_flow(subscriber_op, depth_visualizer, {("depth_outputs", "receivers")})


        # rpc_service_name = shm_config.get("parameter_rpc_name", "holohub")
        # self.rpc_server_ = ParameterRpcServer(node, f"{rpc_service_name}/PARAMETER_RPC/Components", self)
        # self.rpc_server_.update_schema_from_fragment()
        # self.rpc_server_thread_ = threading.Thread(target=self.rpc_server_.serve_blocking)
        # self.rpc_server_thread_.start()


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

    parser = ArgumentParser(description="ARTEKMED Holoscan SHM SSG Inference.")

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
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_shm_ssg_inference.yaml")
    else:
        config_file = args.config

    main(config_file=config_file, scheduler_type=args.scheduler, log_level=args.log_level, with_tracker=args.tracking)
