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
from holohub.tcn_stream_synchronizer import TcnStreamSynchronizerOp
from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_apply_mask import TcnDepthImageApplyMaskOp
from holohub.tcn_label_sampler import TcnLabelSamplerOp
from holohub.tcn_labeled_pointcloud import TcnLabeledPointcloudOp
from holohub.tcn_instance_stats import TcnInstanceStatsOp
from holohub.tcn_stream_merger import TcnStreamMergerOp as StreamMergerOp
from holohub.tcn_device_context._tcn_device_context import XYLookupTableSourceOp

from operators.tcn_artekmed.tcn_util import RotateImage180Op
from operators.tcn_artekmed.tcn_dataset_replayer import TcnDatasetReplayerOp
from operators.tcn_artekmed.tcn_dataset_replayer._calibration import load_device_contexts
from operators.tcn_artekmed.tcn_object_tracking import (
    InstanceFusionOp, ObjectBoxRendererOp, ObjectConsoleSinkOp, ObjectTrackerOp, box_input_specs,
)


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

from operators.tcn_artekmed.tcn_depth_anything import (
    DA2MetricProcessingSubgraph, DA2PostprocessorOp,
    DA3MetricProcessingSubgraph, DA3PostprocessorOp,
)
from operators.tcn_artekmed.tcn_langsam import (
    MaskDumpOp,
    PromptedLangSamSubgraph,
    RealtimeLangSamSubgraph,
    SingleCameraLangSamSubgraph,
    TextPromptPublisher,
    build_panoptic_lut,
    class_id_map,
    mask_name,
    validate_source_cameras,
)


log = logging.getLogger(__name__)

# The vertical world axis, as an index. Spelled `axis_*` in configuration because yaml-cpp resolves a
# bare AND a quoted `y` to boolean true -- see the tcn_object_tracking README.
_UP_AXIS_INDEX = {"axis_x": 0, "axis_y": 1, "axis_z": 2, "x": 0, "y": 1, "z": 2}


# Conservative BlockMemoryPool sizing for `source: "dataset"` mode. There is no live shm
# channel to query a real `frameSize` from (see the `channels_config` synthesis in `compose`),
# so this is sized generously above the larger of the two known dataset resolutions (see
# docs/specs/2026-08-10-replay-harness-design.md "Dataset facts": k4a_capture is
# 2048x1536x3 uint8 = ~9.4 MiB). It only needs to be an upper bound, not exact -- it sizes a
# scratch pool used by ops downstream of the replayer (stream_splitter/holoviz/etc.), not the
# replayer's own device tensors, which it allocates itself via cupy.
_DATASET_MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MiB


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


class GroupCheckOp(Operator):
    """Verifies that a synchronised group really shares one acquisition timestamp.

    Stands in for the eventual mask/depth consumer. `tcn_depthimage_apply_mask` cannot be wired
    here yet -- it wants a single UNNAMED uint8 mask with the SAME element count as the depth image,
    while the panoptic map is a per-camera NAMED uint16 tensor at colour resolution (2048x1536 vs
    the depth stream's 640x576). Resizing across that gap is not registration: the colour and depth
    sensors have different intrinsics, so a naive resize would run and look plausible while being
    spatially wrong. The correct join is backprojection's `texcoords` fed to a texture sampler,
    which is what registers depth pixels into colour space -- tracked as the next step.
    """

    def __init__(self, fragment, *args, streams, **kwargs):
        self.streams = list(streams)
        self.groups = 0
        self.inconsistent = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for name in self.streams:
            spec.input(name)

    def compute(self, op_input, op_output, context):
        stamps = {}
        for name in self.streams:
            op_input.receive(name)
            stamps[name] = op_input.get_acquisition_timestamp(name)
        self.groups += 1
        distinct = {t for t in stamps.values() if t is not None}
        if len(distinct) > 1:
            self.inconsistent += 1
            log.error(f"GroupCheckOp: group {self.groups} is NOT timestamp-consistent: {stamps}")
        elif self.groups <= 3 or self.groups % 50 == 0:
            log.info(f"GroupCheckOp: group {self.groups} consistent at "
                     f"acq={next(iter(distinct)) if distinct else None} ({len(stamps)} streams)")

    def stop(self):
        # Per-group logging is rate-limited, so without this the verdict of a long run is invisible:
        # "no error lines" and "the operator never ticked" look identical in the console.
        verdict = "FAIL" if self.inconsistent else ("PASS" if self.groups else "NO GROUPS")
        log.info(f"GroupCheckOp: {verdict} -- {self.groups} groups, "
                 f"{self.inconsistent} timestamp-inconsistent")


class JoinCheckOp(Operator):
    """Reports what the mask/depth join actually produced, per camera.

    A geometric join fails quietly: wrong extrinsics still yield a full label image, just of the
    wrong pixels. Nothing here can prove the correspondence is right -- that needs the analytic gate
    in the operator's own tests -- but these counts catch the failures that produce *nothing*
    (no valid depth, no texcoord in frustum, no class selected), which otherwise look identical to
    a scene with no detections.
    """

    def __init__(self, fragment, *args, camera, stats, **kwargs):
        self.camera = camera
        self.stats = stats
        self.frames = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("labels")
        spec.input("mask")
        spec.input("masked_depth")
        spec.input("raw_depth")

    def compute(self, op_input, op_output, context):
        labels = cp.asarray(op_input.receive("labels").get(""))
        mask = cp.asarray(op_input.receive("mask").get(""))
        depth = cp.asarray(op_input.receive("masked_depth").get(""))
        raw = cp.asarray(op_input.receive("raw_depth").get(""))
        self.frames += 1
        total = int(labels.size)
        # One sync per frame for three reductions; this operator exists to be read, and the join
        # runs at the mask rate (~5 fps), so the cost is not on any hot path.
        labeled = int(cp.count_nonzero(labels))
        selected = int(cp.count_nonzero(mask))
        kept = int(cp.count_nonzero(depth))
        # Invariant: a pixel can only be selected if it had valid depth (an invalid sample yields a
        # NaN texcoord, hence no label, hence mask 0), so applying the mask must keep every selected
        # pixel. A violation means either the invalidation is not working or apply_mask is indexing
        # the mask differently from the depth image -- both silent, both wrong.
        lost = int(cp.count_nonzero((mask.reshape(-1) != 0) & (depth.reshape(-1) == 0)))
        # Splits the invariant violation in two: a selected pixel whose RAW depth is already zero
        # means the labels do not belong to this depth image (a stale or aliased buffer); one whose
        # raw depth is fine but masked depth is zero means apply_mask dropped it.
        stale = int(cp.count_nonzero((mask.reshape(-1) != 0) & (raw.reshape(-1) == 0)))
        if lost:
            log.error(f"JoinCheckOp[{self.camera}]: frame {self.frames + 1}: of {lost} lost px, "
                      f"{stale} already had zero RAW depth "
                      f"({'labels do not match this depth image' if stale else 'apply_mask dropped them'})")
        if lost:
            self.stats.setdefault(self.camera, {}).setdefault("lost", 0)
            self.stats[self.camera]["lost"] = self.stats[self.camera].get("lost", 0) + lost
            log.error(f"JoinCheckOp[{self.camera}]: frame {self.frames + 1}: {lost} px are masked "
                      f"as selected but have zero depth after apply_mask")
        s = self.stats.setdefault(self.camera, {"frames": 0, "labeled": 0, "selected": 0,
                                                "kept": 0, "total": 0})
        s["frames"] += 1
        s["labeled"] += labeled
        s["selected"] += selected
        s["kept"] += kept
        s["total"] += total
        if self.frames <= 2:
            log.info(f"JoinCheckOp[{self.camera}]: frame {self.frames} "
                     f"{labeled}/{total} px labeled ({100.0 * labeled / total:.1f}%), "
                     f"{selected} selected, {kept} depth px kept")


class CloudFusionCheckOp(Operator):
    """Reports the fused point count per class, so fusion is verified rather than assumed.

    Holoviz not erroring says nothing about whether the merge actually concatenated all cameras --
    a chain that silently forwarded one camera's cloud would look identical. The count is checkable
    against the per-camera counts (`mask_depth_join.verbose`), which must sum to it.
    """

    def __init__(self, fragment, *args, classes, stats, **kwargs):
        self.classes = list(classes)
        self.stats = stats
        self.frames = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for cls in self.classes:
            spec.input(f"class_{cls}")

    def compute(self, op_input, op_output, context):
        counts = {}
        for cls in self.classes:
            msg = op_input.receive(f"class_{cls}")
            tensor = msg.get(f"class_{cls}")
            if tensor is None:
                log.error(f"CloudFusionCheckOp: class_{cls} has no tensor named 'class_{cls}'; "
                          f"present: {sorted(k for k in msg.keys())}")
                counts[cls] = 0
                continue
            arr = cp.asarray(tensor)
            if self.frames == 0:
                log.info(f"CloudFusionCheckOp: class_{cls} arrives as shape {arr.shape}")
            # The point count is the product of every dimension but the trailing xyz, so this reads
            # correctly whether the producer hands over [N,3] or [1,N,3].
            counts[cls] = int(arr.size // arr.shape[-1])
        self.frames += 1
        for cls, n in counts.items():
            self.stats[cls] = max(self.stats.get(cls, 0), n)
        if self.frames <= 2:
            log.info(f"CloudFusionCheckOp: frame {self.frames} fused points per class: "
                     + " ".join(f"class_{c}={n}" for c, n in sorted(counts.items())))


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

        # --- frame source (design doc §2.1) --------------------------------------------
        # "shm" (default) is the live pipeline -- this branch executes the exact same
        # statements as before this toggle existed, so a live run is byte-for-byte unchanged.
        # "dataset" replays a fixed on-disk export through TcnDatasetReplayerOp for
        # deterministic correctness gating (compare two dumps with docs/compare_mask_dumps.py).
        source = str(self.from_config("source")).strip().lower()
        if source not in ("shm", "dataset"):
            raise ValueError(f"Invalid 'source': {source!r}; must be 'shm' or 'dataset'")
        log.info(f"Frame source: {source!r}")

        mask_dump_dir = str(self.from_config("mask_dump_dir")).strip()
        if mask_dump_dir and not camera_streams_config.get("enable_langsam_multicam", False):
            raise ValueError(
                "mask_dump_dir is set but camera_stream_processing.enable_langsam_multicam is "
                "False -- there is no mask output to dump."
            )
        if mask_dump_dir and source == "shm":
            # Dumping is a HARNESS facility: it exists to compare two deterministic replays. On a
            # live stream there is nothing to compare against and the output is unbounded -- one
            # panoptic map is 2048*1536*2 = 6 MiB, so 5 cameras at ~5 fps writes ~9 GiB/minute
            # until the disk fills. A mask_dump_dir left over from a harness run is therefore
            # always a mistake, and one that costs a disk rather than a run.
            raise ValueError(
                f"mask_dump_dir is set ({mask_dump_dir!r}) but source is 'shm' (live).\n"
                f"Mask dumping is for deterministic dataset replays only: on a live stream there "
                f"is no second run to compare against, and the output is unbounded -- roughly "
                f"9 GiB per minute for 5 cameras at 2048x1536.\n"
                f"Set mask_dump_dir: \"\" for live runs, or source: \"dataset\" to dump."
            )

        # Read once, up front: the source needs it (depth must be emitted for the join), the
        # synthetic dataset channel config needs it, and the join block far below needs it too --
        # all three must agree.
        join_cfg = self.kwargs("mask_depth_join") or {}
        join_enabled = bool(join_cfg.get("enabled", False))
        # Set when the tracking chain is built; the prompt publisher is wired to it later, because
        # that publisher is created in the langsam block which runs after the join block.
        self._object_tracker_op = None

        dataset_cfg = None
        dataset_cameras = None
        # The join consumes depth, so the source must emit it. Derived rather than configured
        # separately: `emit_depth: false` with the join on produces empty depth entities, which the
        # splitter reports as a missing tensor -- a confusing way to say "turn depth on".
        dataset_emit_depth = False
        if source == "dataset":
            dataset_cfg = self.kwargs("dataset_source")
            dataset_cameras = list(dataset_cfg.get("cameras") or [])
            dataset_emit_depth = bool(dataset_cfg.get("emit_depth", False)) or join_enabled
            log.info(f"dataset replayer emit_depth={dataset_emit_depth}")
            if not dataset_cameras:
                raise ValueError("dataset_source.cameras must list at least one camera id")

        if source == "shm":
            log.info("Discover contents in shared memory")
            receiver_config = discover_shm(shm_stream_name)
            shm_receiver = receiver_config["receiver"]
            device_contexts = receiver_config["device_contexts"]
            channels_config = receiver_config["channels_config"]
        else:
            shm_receiver = None
            # Calibration comes from the export's own `calibration/<camera>.json`, converted to the
            # device-context shape the SHM path produces (_calibration.py). Without it the replay
            # source has no intrinsics, so the geometric path -- xy_table, backprojection, the
            # mask/depth join -- could only ever run against a live stream, i.e. never against a
            # deterministic input. `missing_ok`: an export without calibration still replays fine
            # for everything that is not geometry, and the join refuses per camera further down
            # rather than being disabled wholesale here.
            device_contexts = load_device_contexts(
                dataset_cfg["path"], dataset_cameras, missing_ok=True)
            uncalibrated = [c for c in dataset_cameras if c not in device_contexts]
            if uncalibrated:
                log.warning(f"dataset export has no calibration for {uncalibrated}; the mask/depth "
                            f"join cannot be built for those cameras")
            else:
                log.info(f"loaded dataset calibration for {sorted(device_contexts)}")
            # A synthetic `channels_config` shaped just like discover_shm()'s
            # return value, containing only the fields actually read below, so the rest of
            # compose() (the color/depth stream split, pool sizing, ...) runs unchanged for
            # both sources.
            # Depth ports are declared only when depth is actually emitted, so what this
            # synthetic config says about the source stays true -- the join discovers its depth
            # channels from here exactly as it does from a live segment.
            channels_config = {
                "ports": [
                    {
                        "name": f"{cam}_colorimage",
                        "status": {
                            "portType": "colorimage",
                            "bufferInfo": {"frameSize": _DATASET_MAX_FRAME_BYTES},
                        },
                    }
                    for cam in dataset_cameras
                ] + ([
                    {
                        "name": f"{cam}_depthimage",
                        "status": {
                            "portType": "depthimage",
                            "bufferInfo": {"frameSize": _DATASET_MAX_FRAME_BYTES},
                        },
                    }
                    for cam in dataset_cameras
                ] if dataset_emit_depth else [])
            }

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

        # Semantic types are only reported by live shm channels; the synthetic dataset-mode
        # `channels_config` above has no `semanticType` at all (its only consumer,
        # need_convert_bgra below, is resolved without it in that mode).
        channel_semantic_types = None
        if source == "shm":
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

        # --- frame source op: ShmSubscriberOp (live) or TcnDatasetReplayerOp (deterministic
        # replay) -- design doc §2.1. Built once, wired to the exact same two ports/edges
        # (`color_outputs`, `depth_outputs`) either way, so every downstream operator is
        # unaware of which one is feeding it.
        if source == "shm":
            log.info(f"create subscriber op {shm_stream_name}")
            shm_async_condition = AsynchronousCondition(self, name="shm_async_condition")
            frame_source_op = ShmSubscriberOp(self, cuda_stream_pool,
                                            allocator=device_memory_pool,
                                            async_condition=shm_async_condition,
                                            receiver=shm_receiver,
                                            stream_name=shm_stream_name,
                                            cycle_time_ms=cycle_time_ms,
                                            name="shm_subscriber")
        else:
            dataset_frame_count = dataset_cfg.get("frame_count")
            if dataset_frame_count is None:
                raise ValueError(
                    "dataset_source.frame_count must be a positive integer in dataset mode -- "
                    "the app bounds the run with CountCondition(count=frame_count) so it "
                    "terminates on its own (design doc §2.1); it cannot discover the "
                    "dataset's frame count before compose() without decoding it."
                )
            dataset_playback = dataset_cfg.get("playback", "auto")
            log.info(
                f"create dataset replayer op: path={dataset_cfg['path']!r} "
                f"cameras={dataset_cameras} frame_count={dataset_frame_count} "
                f"loop={dataset_cfg.get('loop', True)} playback={dataset_playback!r}"
            )
            source_args = ()
            if dataset_playback == "auto":
                # Bounds the run to exactly the requested frames so a harness run exits on its
                # own instead of running forever (design doc §2.1).
                source_args = (CountCondition(self, count=int(dataset_frame_count)),)
            frame_source_op = TcnDatasetReplayerOp(
                self, *source_args,
                dataset_path=dataset_cfg["path"],
                cameras=dataset_cameras,
                frame_count=int(dataset_frame_count),
                loop=bool(dataset_cfg.get("loop", True)),
                emit_depth=dataset_emit_depth,
                playback=dataset_playback,
                allocator=device_memory_pool,
                cuda_stream_pool=cuda_stream_pool,
                name="dataset_replayer",
            )

        log.info(f"create stream_splitter op: {color_streams_config[:1]}")
        # XXX only one for now
        split_op = StreamSplitterOp(self, cuda_stream_pool,
                                    channel_names=[v["name"] for v in color_streams_config[:1]],
                                    name="stream_splitter")
        self.add_flow(frame_source_op, split_op, {("color_outputs", "receivers")})

        # Temporal synchronisation of the mask and depth streams
        # (docs/specs/2026-08-11-temporal-sync-design.md). Off by default: `temporal_sync.enabled`.
        #
        # The mask path runs ~5 fps and about two frames behind the source while depth is
        # independent and faster, so pairing "whatever depth is current" with a mask mis-registers
        # it by a varying amount. The synchroniser buffers both and emits only groups that share an
        # acquisition timestamp.
        #
        # NOTE the streams are wired but the CONSUMER is GroupCheckOp, not
        # tcn_depthimage_apply_mask -- see GroupCheckOp's docstring for why that operator cannot
        # consume a colour-resolution packed panoptic map, and what the correct join is.
        sync_cfg = self.kwargs("temporal_sync") or {}
        sync_enabled = bool(sync_cfg.get("enabled", False)) and \
            camera_streams_config.get("enable_langsam_multicam", False)
        temporal_sync = None
        if sync_enabled:
            # `streams` MUST be a constructor argument: the operator creates its ports in setup(),
            # which Holoscan runs before parameter values are applied, so this cannot come from
            # from_config() like the other settings.
            sync_streams = ["masks", "depth"]
            temporal_sync = TcnStreamSynchronizerOp(
                self,
                streams=sync_streams,
                capacities=[int(sync_cfg.get("masks_capacity", 4)),
                            int(sync_cfg.get("depth_capacity", 16))],
                reference_stream="masks",      # the laggiest stream must drive the search
                match_policy=str(sync_cfg.get("match_policy", "exact")),
                window_ns=int(sync_cfg.get("window_ns", 0)),
                verbose=bool(sync_cfg.get("verbose", False)),
                name="temporal_sync",
            )
            group_check = GroupCheckOp(self, streams=sync_streams, name="group_check")
            self.add_flow(frame_source_op, temporal_sync, {("depth_outputs", "depth")})
            for name in sync_streams:
                self.add_flow(temporal_sync, group_check, {(name, name)})
            log.info(f"Temporal sync ENABLED: streams={sync_streams} "
                     f"policy={sync_cfg.get('match_policy', 'exact')}")

            # --- the real mask/depth join (docs/specs/2026-08-12-mask-depth-join-design.md) ---
            # Per camera: depth -> backprojection -> texcoords, then the panoptic map sampled
            # through those texcoords onto the depth grid, then apply_mask on the depth image.
            # Both inputs come out of the synchroniser, so the mask and the depth belong to the
            # same captured frame -- joining unsynchronised streams is the mis-registration this
            # whole path exists to remove, which is why it is nested here rather than configurable
            # independently.
            if join_enabled:
                join_cams = [c["name"].replace("_colorimage", "") for c in color_streams_config]
                # Refuse rather than skip: a camera silently dropped from the join produces a
                # point cloud that is simply missing its labels, which reads as "no detections".
                missing_calib = [c for c in join_cams if not ctx_service.has_camera(c)]
                if missing_calib:
                    raise ValueError(
                        f"mask_depth_join is enabled but there is no calibration for "
                        f"{missing_calib}. The join needs per-camera intrinsics, distortion and "
                        f"the depth->colour transform; on the dataset source these come from the "
                        f"export's calibration/<camera>.json.")

                # Depth tensor names come from the channels the SOURCE actually declares, matched to
                # each camera by name prefix -- not built as f"{cam}_depthimage". The convention
                # happens to hold for both sources today, but a live segment that names its depth
                # channel anything else would fail at the first tick inside the splitter
                # ("input entity missing tensor"), pointing at the splitter rather than at the
                # mismatch. Refusing here names the camera and lists what the source does provide.
                depth_channel_for = {}
                for cam in join_cams:
                    candidates = [c["name"] for c in depth_streams_config
                                  if c["name"] == f"{cam}_depthimage"
                                  or c["name"].startswith(f"{cam}_")]
                    if not candidates:
                        raise ValueError(
                            f"mask_depth_join is enabled but the source declares no depth channel "
                            f"for camera {cam!r}. Depth channels found: "
                            f"{[c['name'] for c in depth_streams_config]}. The join unprojects the "
                            f"depth image, so a camera without depth cannot take part; on the "
                            f"dataset source check dataset_source.emit_depth.")
                    if len(candidates) > 1:
                        raise ValueError(
                            f"mask_depth_join found {len(candidates)} depth channels for camera "
                            f"{cam!r}: {candidates}. Cannot tell which to unproject.")
                    depth_channel_for[cam] = candidates[0]
                log.info(f"mask/depth join depth channels: {depth_channel_for}")

                # The join needs its OWN allocator. The shared device_memory_pool is a
                # BlockMemoryPool whose blocks are sized for a whole camera frame (16 MiB), and the
                # join adds dozens of small, VARIABLE-sized tensors per frame -- two per class per
                # camera, plus a merge each. Every one of those would consume a full
                # block, exhausting the pool ("Too many chunks allocated") while wasting most of
                # what it did hand out. RMM sub-allocates from a pool, which is what variable sizes
                # want.
                join_pool = RMMAllocator(
                    self,
                    name="join_pool",
                    dev_id=cuda_device_id,
                    device_memory_initial_size=str(join_cfg.get("pool_initial_size", "256MB")),
                    device_memory_max_size=str(join_cfg.get("pool_max_size", "1GB")))

                depth_split = StreamSplitterOp(
                    self, cuda_stream_pool,
                    channel_names=[depth_channel_for[c] for c in join_cams],
                    name="join_depth_splitter")
                mask_split = StreamSplitterOp(
                    self, cuda_stream_pool,
                    channel_names=[mask_name(f"{c}_colorimage") for c in join_cams],
                    name="join_mask_splitter")
                self.add_flow(temporal_sync, depth_split, {("depth", "receivers")})
                self.add_flow(temporal_sync, mask_split, {("masks", "receivers")})

                # The verification operators read every output back with GPU reductions -- one
                # sync per camera per frame. That is worth it while gating (it is what caught the
                # splitter use-after-free), and it is measurable overhead on a live run, so it is
                # switchable. Default on: a silent join is the failure mode this path is prone to.
                join_checks = bool(join_cfg.get("check", True))
                # --- object tracking (docs/specs are in the plan; see tcn_object_tracking) ------
                # Instance ids from the masks are per-camera, per-frame confidence RANKS, not
                # identities. This chain turns them into one entry per physical object with a stable
                # id: per-camera reduction -> cross-camera fusion -> temporal tracking -> console.
                track_cfg = self.kwargs("object_tracking") or {}
                track_enabled = bool(track_cfg.get("enabled", False))
                fusion_op = None
                if track_enabled:
                    fusion_op = InstanceFusionOp(
                        self,
                        iou_threshold=float(track_cfg.get("fusion_iou_threshold", 0.15)),
                        containment_threshold=float(track_cfg.get("fusion_containment", 0.6)),
                        max_centroid_distance_m=float(track_cfg.get("fusion_max_distance_m", 1.0)),
                        footprint_iou_threshold=float(track_cfg.get("fusion_footprint_iou", 0.4)),
                        max_vertical_gap_m=float(track_cfg.get("fusion_max_vertical_gap_m", 0.5)),
                        up_axis=track_cfg.get("up_axis", "axis_y"),
                        min_extent_m=float(track_cfg.get("min_extent_m", 0.0)),
                        min_points=int(track_cfg.get("fusion_min_points", 0)),
                        aggregate_containment=float(track_cfg.get("aggregate_containment", 0.7)),
                        aggregate_min_children=int(track_cfg.get("aggregate_min_children", 2)),
                        aggregate_min_volume_ratio=float(
                            track_cfg.get("aggregate_min_volume_ratio", 1.5)),
                        min_cameras=int(track_cfg.get("min_cameras", 1)),
                        min_detection_points=int(track_cfg.get("min_detection_points", 0)),
                        min_detection_extent_m=float(track_cfg.get("min_detection_extent_m", 0.0)),
                        verbose=bool(track_cfg.get("verbose", False)),
                        name="object_fusion")
                    tracker_op = ObjectTrackerOp(
                        self,
                        min_hits=int(track_cfg.get("min_hits", 3)),
                        max_age=int(track_cfg.get("max_age", 8)),
                        iou_threshold=float(track_cfg.get("track_iou_threshold", 0.1)),
                        max_centroid_distance_m=float(track_cfg.get("track_max_distance_m", 1.0)),
                        box_smoothing=float(track_cfg.get("box_smoothing", 0.5)),
                        max_yaw_smoothing_delta_rad=float(track_cfg.get("max_yaw_smoothing_delta_rad", 0.26)),
                        class_names=list((self.kwargs("text_prompts") or {}).get("prompts") or []),
                        verbose=bool(track_cfg.get("verbose", False)),
                        name="object_tracker")
                    sink_op = ObjectConsoleSinkOp(
                        self, print_every=int(track_cfg.get("print_every", 1)),
                        name="object_console")
                    self.add_flow(fusion_op, tracker_op, {("detections", "detections")})
                    self.add_flow(tracker_op, sink_op, {("objects", "objects")})
                    # A prompt change renumbers class ids, so the tracker must reset -- but the
                    # prompt publisher only exists in the langsam block further down (it needs the
                    # camera list). Deferred there, the same way the mask half of the temporal-sync
                    # join is.
                    self._object_tracker_op = tracker_op
                    log.info(f"Object tracking ENABLED: min_hits={track_cfg.get('min_hits', 3)} "
                             f"max_age={track_cfg.get('max_age', 8)}")

                join_stats = {}
                self._join_stats = join_stats          # read in run() for the end-of-run summary
                select_classes = [int(c) for c in (join_cfg.get("select_classes") or [])]

                # Classes the fused view draws, one HolovizOp InputSpec (hence one colour) each.
                # Derived from the prompts by default so they cannot drift from what LangSAM
                # actually detects -- class ids are 1-based prompt positions (class_id_map).
                prompts = list((self.kwargs("text_prompts") or {}).get("prompts") or [])
                cloud_classes = [int(c) for c in (join_cfg.get("pointcloud_classes") or [])]
                if not cloud_classes:
                    cloud_classes = sorted(class_id_map(prompts).values())
                if not cloud_classes:
                    raise ValueError(
                        "mask_depth_join is enabled but there are no classes to build point clouds "
                        "for: text_prompts.prompts is empty and mask_depth_join.pointcloud_classes "
                        "was not set.")
                cloud_merge_connections = {cls: [] for cls in cloud_classes}
                cloud_merge_ops = []
                for cam in join_cams:
                    color_model = ctx_service.get_color_camera_model(cam)

                    # Emits once (CountCondition) and backprojection's xy_table port carries no
                    # condition, so later ticks are not gated on it.
                    xylt_op = XYLookupTableSourceOp(
                        self, CountCondition(self, count=1),
                        allocator=join_pool,
                        camera_name=cam,
                        name=f"join_xylt_{cam}")
                    xylt_op.set_device_context_service(ctx_service)

                    bp_op = TcnDepthImageBackprojectionOp(
                        self, cuda_stream_pool,
                        allocator=join_pool,
                        color_image_width=color_model.dimensions.x,
                        color_image_height=color_model.dimensions.y,
                        color_params=color_model,
                        depth_extrinsics=ctx_service.get_depth_extrinsics(cam),
                        depth_to_color=ctx_service.get_color_to_depth_inv(cam),
                        in_tensor_name="",
                        out_tensor_name="",
                        # Positions are the point cloud itself (world space, via
                        # depth_extrinsics); texcoords carry the colour correspondence. The two are
                        # independent outputs of one unprojection -- texcoords used to be written
                        # only inside the positions branch, which is fixed in the kernel.
                        enable_positions=True,
                        enable_texcoords=True,
                        enable_depth_float=False,
                        cuda_device_ordinal=cuda_device_id,
                        name=f"join_bp_{cam}",
                        **self.kwargs("depthimage_backprojection"))

                    sampler_op = TcnLabelSamplerOp(
                        self,
                        allocator=join_pool,
                        cuda_device_ordinal=cuda_device_id,
                        select_classes=select_classes,
                        unlabeled_value=int(join_cfg.get("unlabeled_value", 0)),
                        name=f"join_sampler_{cam}")

                    # Now connectable: the mask lives on the depth grid, so it satisfies
                    # apply_mask's "single unnamed uint8 tensor, same element count as the depth
                    # image" contract that a colour-resolution panoptic map never could.
                    apply_op = TcnDepthImageApplyMaskOp(
                        self,
                        allocator=join_pool,
                        invert_mask=bool(join_cfg.get("invert_mask", False)),
                        name=f"join_apply_mask_{cam}")

                    check_op = None
                    if join_checks:
                        check_op = JoinCheckOp(self, camera=cam, stats=join_stats,
                                               name=f"join_check_{cam}")

                    self.add_flow(depth_split, bp_op,
                                  {(depth_channel_for[cam], "depth_image")})
                    self.add_flow(xylt_op, bp_op, {("xy_table", "xy_table")})
                    self.add_flow(mask_split, sampler_op,
                                  {(mask_name(f"{cam}_colorimage"), "labels")})
                    self.add_flow(bp_op, sampler_op, {("texcoords", "texcoords")})
                    self.add_flow(depth_split, apply_op,
                                  {(depth_channel_for[cam], "depth_image")})
                    self.add_flow(sampler_op, apply_op, {("mask_out", "mask_image")})
                    if check_op is not None:
                        self.add_flow(sampler_op, check_op, {("labels_out", "labels")})
                        self.add_flow(sampler_op, check_op, {("mask_out", "mask")})
                        self.add_flow(apply_op, check_op, {("output", "masked_depth")})
                        self.add_flow(depth_split, check_op,
                                      {(depth_channel_for[cam], "raw_depth")})
                    else:
                        # apply_mask still needs a consumer, or it back-pressures the whole join.
                        self.add_flow(apply_op, DummySinkOp(self, name=f"join_apply_sink_{cam}"),
                                      {("output", "input")})
                        self.add_flow(sampler_op,
                                      DummySinkOp(self, name=f"join_labels_sink_{cam}"),
                                      {("labels_out", "input")})

                    # Per-camera instance statistics for the tracking path. Taps the SAME two
                    # outputs the point cloud consumes, so it is additive: one row per panoptic
                    # instance with its count, centroid, box and pre-trim spread.
                    if track_enabled:
                        stats_op = TcnInstanceStatsOp(
                            self,
                            allocator=join_pool,
                            cuda_device_ordinal=cuda_device_id,
                            camera_index=join_cams.index(cam),
                            trim_percentile=float(track_cfg.get("trim_percentile", 0.02)),
                            trim_margin=float(track_cfg.get("trim_margin", 0.05)),
                            min_range_m=float(track_cfg.get("min_range_m", 0.01)),
                            up_axis=_UP_AXIS_INDEX[str(track_cfg.get("up_axis", "axis_y"))],
                            min_anisotropy=float(track_cfg.get("min_anisotropy", 1.5)),
                            min_points=int(track_cfg.get("min_points", 64)),
                            max_instances=int(track_cfg.get("max_instances", 64)),
                            verbose=bool(track_cfg.get("verbose_instances", False)),
                            name=f"join_stats_{cam}")
                        self.add_flow(bp_op, stats_op, {("positions", "positions")})
                        self.add_flow(sampler_op, stats_op, {("labels_out", "labels")})
                        self.add_flow(stats_op, fusion_op, {("instances", "receivers")})

                    # Per-camera labeled point cloud: world-space points carrying their packed
                    # (class, instance) label, split into one entity per class.
                    cloud_op = TcnLabeledPointcloudOp(
                        self,
                        allocator=join_pool,
                        classes=cloud_classes,
                        cuda_device_ordinal=cuda_device_id,
                        verbose=bool(join_cfg.get("verbose", False)),
                        name=f"join_cloud_{cam}")
                    self.add_flow(bp_op, cloud_op, {("positions", "positions")})
                    self.add_flow(sampler_op, cloud_op, {("labels_out", "labels")})
                    for cls in cloud_classes:
                        port = f"class_{cls}"
                        cloud_merge_connections[cls].append((cloud_op, {(port, f"{cam}_{port}")}))

                # --- fuse each class across cameras and show the result -------------------
                # tcn_shm_receiver's point fusion is merge -> flatten -> HolovizOp POINTS_3D, and
                # what Holoviz consumes there is [1, N, 3] -- tcn_flatten_tensor maps
                # [H, W, ...] to [1, H*W, ...], keeping the leading 1. Our per-class clouds are
                # already [1, N, 3] and the merger concatenates along dimension 1, so the flatten
                # would be an exact no-op and is left out rather than run once per class per frame.
                # One chain per class, because Holoviz colours a spec and not a vertex -- that is
                # what makes classes distinguishable in the fused view.
                lut = build_panoptic_lut(max(cloud_classes))
                cloud_specs = []
                for cls in cloud_classes:
                    tensor_name = f"class_{cls}"
                    merge_inputs = [f"{cam}_{tensor_name}" for cam in join_cams]
                    merge_op = StreamMergerOp(
                        self, cuda_stream_pool,
                        input_port_names=merge_inputs,
                        input_message_name="positions",
                        output_message_name=tensor_name,
                        fuse_buffers=True,
                        allocator=join_pool,
                        name=f"join_cloud_fusion_{cls}")
                    for op, conn in cloud_merge_connections[cls]:
                        self.add_flow(op, merge_op, conn)

                    spec = HolovizOp.InputSpec(tensor_name, HolovizOp.InputType.POINTS_3D)
                    # The class's base colour from the same LUT the 2D mask overlay uses, so a
                    # class is the same colour in the image view and in the point cloud.
                    rgba = cp.asnumpy(lut[cls << 8]).astype(float) / 255.0
                    spec.color = [float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0]
                    cloud_specs.append(spec)
                    cloud_merge_ops.append(merge_op)

                if join_checks:
                    cloud_fusion_stats = {}
                    self._cloud_fusion_stats = cloud_fusion_stats
                    fusion_check = CloudFusionCheckOp(self, classes=cloud_classes,
                                                      stats=cloud_fusion_stats,
                                                      name="cloud_fusion_check")
                    for cls, merge_op in zip(cloud_classes, cloud_merge_ops):
                        self.add_flow(merge_op, fusion_check, {("output", f"class_{cls}")})

                # Bounding-box overlay, into the same view as the points. Built here because it
                # needs `lut` and appends to `cloud_specs`, and wired from the tracker created above.
                if track_enabled and bool(track_cfg.get("render_boxes", True)):
                    box_renderer = ObjectBoxRendererOp(
                        self, classes=cloud_classes, device=cuda_device_id,
                        oriented=bool(track_cfg.get("render_oriented_boxes", True)),
                        up_axis=_UP_AXIS_INDEX[str(track_cfg.get("up_axis", "axis_y"))],
                        name="object_box_renderer")
                    self.add_flow(tracker_op, box_renderer, {("objects", "objects")})
                    cloud_specs.extend(box_input_specs(
                        cloud_classes, lut,
                        line_width=float(track_cfg.get("box_line_width", 3.0))))
                else:
                    box_renderer = None

                cloud_visualizer = HolovizOp(
                    self,
                    name="labeled_pointcloud_visualizer",
                    tensors=cloud_specs,
                    allocator=join_pool,
                    cuda_stream_pool=cuda_stream_pool,
                    **self.kwargs("labeled_pointcloud_holoviz"))
                for merge_op in cloud_merge_ops:
                    self.add_flow(merge_op, cloud_visualizer, {("output", "receivers")})
                if box_renderer is not None:
                    self.add_flow(box_renderer, cloud_visualizer, {("boxes", "receivers")})

                log.info(f"Mask/depth join ENABLED for {join_cams} "
                         f"(select_classes={select_classes or 'all non-background'}, "
                         f"pointcloud classes={cloud_classes})")
        else:
            di_sink = DummySinkOp(self, name="depth_image_sink")
            self.add_flow(frame_source_op, di_sink, {("depth_outputs", "input")})


        config_rotate_image = False
        have_camera_consumer = False

        inference_input = (split_op, "camera01_colorimage")
        camera_device_context = device_contexts.get("camera01")

        if config_rotate_image:
            log.info("rotate image enabled")
            rotate_op = RotateImage180Op(self)
            self.add_flow(inference_input[0], rotate_op, {(inference_input[1], "input")})
            inference_input = (rotate_op, "output")

        # BGRA->RGBA detection needs a live channel's reported semantic type, which only
        # exists when source == "shm" (channel_semantic_types above). TcnDatasetReplayerOp
        # always emits plain BGR (channel_order="bgr", its default -- see its docstring), so
        # there is no alpha channel to strip in dataset mode.
        need_convert_bgra = False
        if source == "shm":
            current_config = color_streams_config[0]
            channel_st = SemanticType(current_config['status']['bufferInfo']['semanticType'])
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
            # Single camera, prompts from configuration. The multi-camera promptable path is
            # PromptedLangSamSubgraph -- see the tcn_langsam README for which to pick.
            langsam_pipeline = SingleCameraLangSamSubgraph(self, "camera01_langsam_pipeline",
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

        if camera_streams_config.get("enable_langsam_multicam", False):
            all_color_cams = [c["name"] for c in color_streams_config]

            # gpu_workers.workers must name exactly the cameras this source provides -- on
            # ANY source (design doc "Dataset facts" / see _validate_source_cameras
            # docstring). Checked here, not earlier, because gpu_workers.workers is only
            # actually consumed when this subgraph is built -- validating it unconditionally
            # would newly break a live run that has enable_langsam_multicam off and a stale,
            # otherwise-harmless gpu_workers block.
            validate_source_cameras(source, all_color_cams, self.kwargs("gpu_workers"))

            # Two multi-camera variants, identical downstream: `realtime` runs Grounding DINO as a
            # prebuilt TRT engine with its vocabulary baked in; `prompted` runs it in PyTorch and
            # takes the vocabulary from a `prompts` port, so it can change while the pipeline runs.
            # Both keep the SAM TRT encoder, the batched decode and the fused panoptic paint, because
            # none of those depend on the prompt set.
            langsam_variant = str(camera_streams_config.get("langsam_variant", "realtime")).lower()
            if langsam_variant not in ("realtime", "prompted"):
                raise ValueError(
                    f"camera_stream_processing.langsam_variant must be 'realtime' or 'prompted', "
                    f"got {langsam_variant!r}")
            log.info(f"LangSAM multicam variant: {langsam_variant}")

            if langsam_variant == "prompted":
                langsam_mc = PromptedLangSamSubgraph(
                    self, "langsam_multicam", self.kwargs, all_color_cams)
                # Publish once: prompt-derived state is rebuilt on change, so a per-tick publisher
                # would be pure overhead. An application wanting interactive prompting replaces this
                # operator with its own source (UI, RPC, file watch) on the same port and message
                # shape -- {"text_prompts": [...]}.
                prompt_cfg = self.kwargs("text_prompts") or {}
                runtime_prompts = list(prompt_cfg.get("runtime_prompts")
                                       or prompt_cfg.get("prompts") or [])
                prompt_pub = TextPromptPublisher(
                    self, CountCondition(self, count=1),
                    prompts=runtime_prompts, name="text_prompt_publisher")
                self.add_flow(prompt_pub, langsam_mc, {("out", "prompts")})
                if self._object_tracker_op is not None:
                    # Same message drives the tracker's class-definition reset: class ids are prompt
                    # positions, so a vocabulary change invalidates every existing identity.
                    self.add_flow(prompt_pub, self._object_tracker_op, {("out", "prompts")})
                log.info(f"Prompt publisher will send: {runtime_prompts}")
            else:
                langsam_mc = RealtimeLangSamSubgraph(
                    self, "langsam_multicam", self.kwargs, all_color_cams)

            langsam_mc_holoviz = HolovizOp(
                self,
                allocator=device_memory_pool,
                name="langsam_multicam_holoviz",
                window_title="LangSAM Multi-Camera Masks",
                **self.kwargs("langsam_multicam_holoviz"),
            )

            # Feed the full color entity (all cameras) straight in; workers self-select.
            self.add_flow(frame_source_op, langsam_mc, {("color_outputs", "input")})
            self.add_flow(langsam_mc, langsam_mc_holoviz, {("output_viz", "receivers")})
            self.add_flow(langsam_mc, langsam_mc_holoviz, {("output_specs", "input_specs")})
            have_camera_consumer = True

            # --- mask dump (design doc §2.2) -- only built when mask_dump_dir is set, so a
            # live run pays nothing. Uses the replayer's true source frame number in dataset
            # mode; falls back to a tick counter in shm mode (correct there -- a live stream
            # has no dataset frame number to report).
            if mask_dump_dir:
                mask_dump_op = MaskDumpOp(
                    self, name="mask_dump",
                    out_dir=mask_dump_dir,
                    frame_source=frame_source_op if source == "dataset" else None,
                )
                self.add_flow(langsam_mc, mask_dump_op, {("output_masks", "masks")})
                log.info(f"Mask dump enabled: writing to {mask_dump_dir!r}")

            # The mask half of the temporal-sync join. The depth half and the operator itself were
            # created earlier (the depth branch), because the source exists by then; langsam_mc only
            # exists here. Both halves must be wired or the synchroniser starves on a required
            # stream -- which it reports, but the fix is this edge.
            if temporal_sync is not None:
                self.add_flow(langsam_mc, temporal_sync, {("output_masks", "masks")})

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
        #self.add_flow(frame_source_op, color_visualizer, {("color_output_specs", "input_specs")})


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

    _report_join_stats(app)


def _report_join_stats(app):
    """End-of-run summary for the mask/depth join, or silence when it was not built.

    JoinCheckOp rate-limits its per-frame logging, so without this a long run's outcome is invisible
    -- and "the join produced nothing" must not look like "the join was off".
    """
    fused = getattr(app, "_cloud_fusion_stats", None)
    if fused:
        log.info("Fused labeled point cloud, peak points per class: "
                 + " ".join(f"class_{c}={n}" for c, n in sorted(fused.items())))

    stats = getattr(app, "_join_stats", None)
    if not stats:
        return
    log.info("Mask/depth join summary:")
    for camera in sorted(stats):
        s = stats[camera]
        total = max(1, s["total"])
        log.info(f"  {camera}: {s['frames']} frame(s), "
                 f"{100.0 * s['labeled'] / total:.1f}% of depth px labeled, "
                 f"{100.0 * s['selected'] / total:.1f}% selected, "
                 f"{100.0 * s['kept'] / total:.1f}% depth px kept")
        if s["labeled"] == 0:
            log.error(f"  {camera}: NO depth pixel received a label -- either no depth was valid, "
                      f"no texcoord fell inside the colour frustum, or the panoptic map was empty")
        if s.get("lost"):
            log.error(f"  {camera}: FAIL -- {s['lost']} px were selected but lost their depth; the "
                      f"labels did not belong to the depth image they were applied to")
        elif s["labeled"]:
            log.info(f"  {camera}: PASS -- every selected pixel kept its depth")


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
        default="event_based",
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
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_all.yaml")
    else:
        config_file = args.config

    main(config_file=config_file, scheduler_type=args.scheduler, log_level=args.log_level, with_tracker=args.tracking)
