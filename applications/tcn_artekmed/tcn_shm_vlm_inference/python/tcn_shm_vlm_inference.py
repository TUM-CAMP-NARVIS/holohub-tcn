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
from holohub.tcn_device_context._tcn_device_context import XYLookupTableSourceOp

from operators.tcn_artekmed.tcn_util import RotateImage180Op
from operators.tcn_artekmed.tcn_dataset_replayer import TcnDatasetReplayerOp
from operators.tcn_artekmed.tcn_dataset_replayer._calibration import load_device_contexts

from langsam_helpers import mask_name

from mask_dump import MaskDumpOp

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
from langsam_multicam_fragment import LangSamMultiCamProcessingSubgraph


log = logging.getLogger(__name__)


# Conservative BlockMemoryPool sizing for `source: "dataset"` mode. There is no live shm
# channel to query a real `frameSize` from (see the `channels_config` synthesis in `compose`),
# so this is sized generously above the larger of the two known dataset resolutions (see
# docs/specs/2026-08-10-replay-harness-design.md "Dataset facts": k4a_capture is
# 2048x1536x3 uint8 = ~9.4 MiB). It only needs to be an upper bound, not exact -- it sizes a
# scratch pool used by ops downstream of the replayer (stream_splitter/holoviz/etc.), not the
# replayer's own device tensors, which it allocates itself via cupy.
_DATASET_MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MiB


def _validate_source_cameras(source, provided_cameras, gpu_workers_cfg):
    """`gpu_workers.workers` must name exactly the cameras the active source provides.

    Both directions matter, on ANY source: a camera listed in `workers` that the source
    doesn't emit gets silently-empty masks that look like a real regression (e.g. a 4-camera
    dataset export replayed against a 5-camera worker config); the reverse -- a camera the
    source emits that no worker claims -- gets silently dropped output (e.g. a 5-camera export
    against a 4-camera worker config). 4-camera and 5-camera rigs are BOTH real, supported
    deployment topologies, not "test" vs "production", so a mismatch here is always a
    misconfiguration to fix, never an expected condition to silently work around.
    """
    provided = sorted(set(provided_cameras))
    workers = (gpu_workers_cfg or {}).get("workers") or []
    configured = sorted({cam for w in workers for cam in (w.get("cameras") or [])})
    missing = sorted(set(configured) - set(provided))    # configured, source doesn't provide
    extra = sorted(set(provided) - set(configured))       # provided, no worker claims it
    if missing or extra:
        detail = [
            f"source {source!r} provides {len(provided)} camera(s): {provided}",
            f"gpu_workers.workers is configured for {len(configured)} camera(s): {configured}",
        ]
        if missing:
            detail.append(f"  missing (configured, but NOT provided by the source): {missing}")
        if extra:
            detail.append(f"  extra (provided by the source, but NO worker claims them): {extra}")
        raise ValueError(
            "gpu_workers.workers camera set does not match the cameras the active source "
            "provides.\n  " + "\n  ".join(detail) +
            "\nAdjust gpu_workers.workers to match this source's cameras -- see the "
            "commented alternative camera-count profile next to gpu_workers/dataset_source "
            "in the yaml."
        )


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

        # Read once, up front: the source needs it (depth must be emitted for the join) and the
        # join block far below needs it too, and the two must not disagree.
        join_cfg = self.kwargs("mask_depth_join") or {}

        dataset_cfg = None
        dataset_cameras = None
        if source == "dataset":
            dataset_cfg = self.kwargs("dataset_source")
            dataset_cameras = list(dataset_cfg.get("cameras") or [])
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
                ]
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
            # The join consumes depth, so the source must emit it. Derived rather than configured
            # separately: `emit_depth: false` with the join on produces empty depth entities, which
            # the splitter reports as a missing tensor -- a confusing way to say "turn depth on".
            dataset_emit_depth = (bool(dataset_cfg.get("emit_depth", False))
                                  or bool(join_cfg.get("enabled", False)))
            log.info(f"dataset replayer emit_depth={dataset_emit_depth}")
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
            if bool(join_cfg.get("enabled", False)):
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

                depth_split = StreamSplitterOp(
                    self, cuda_stream_pool,
                    channel_names=[f"{c}_depthimage" for c in join_cams],
                    name="join_depth_splitter")
                mask_split = StreamSplitterOp(
                    self, cuda_stream_pool,
                    channel_names=[mask_name(f"{c}_colorimage") for c in join_cams],
                    name="join_mask_splitter")
                self.add_flow(temporal_sync, depth_split, {("depth", "receivers")})
                self.add_flow(temporal_sync, mask_split, {("masks", "receivers")})

                join_stats = {}
                self._join_stats = join_stats          # read in run() for the end-of-run summary
                select_classes = [int(c) for c in (join_cfg.get("select_classes") or [])]
                for cam in join_cams:
                    color_model = ctx_service.get_color_camera_model(cam)

                    # Emits once (CountCondition) and backprojection's xy_table port carries no
                    # condition, so later ticks are not gated on it.
                    xylt_op = XYLookupTableSourceOp(
                        self, CountCondition(self, count=1),
                        allocator=device_memory_pool,
                        camera_name=cam,
                        name=f"join_xylt_{cam}")
                    xylt_op.set_device_context_service(ctx_service)

                    bp_op = TcnDepthImageBackprojectionOp(
                        self, cuda_stream_pool,
                        allocator=device_memory_pool,
                        color_image_width=color_model.dimensions.x,
                        color_image_height=color_model.dimensions.y,
                        color_params=color_model,
                        depth_extrinsics=ctx_service.get_depth_extrinsics(cam),
                        depth_to_color=ctx_service.get_color_to_depth_inv(cam),
                        in_tensor_name="",
                        out_tensor_name="",
                        # Texcoords only: the join needs the colour correspondence, not the point
                        # cloud. This configuration used to emit an untouched texcoord buffer --
                        # the writes were nested inside the positions branch (fixed in the kernel).
                        enable_positions=False,
                        enable_texcoords=True,
                        enable_depth_float=False,
                        cuda_device_ordinal=cuda_device_id,
                        name=f"join_bp_{cam}",
                        **self.kwargs("depthimage_backprojection"))

                    sampler_op = TcnLabelSamplerOp(
                        self,
                        allocator=device_memory_pool,
                        cuda_device_ordinal=cuda_device_id,
                        select_classes=select_classes,
                        unlabeled_value=int(join_cfg.get("unlabeled_value", 0)),
                        name=f"join_sampler_{cam}")

                    # Now connectable: the mask lives on the depth grid, so it satisfies
                    # apply_mask's "single unnamed uint8 tensor, same element count as the depth
                    # image" contract that a colour-resolution panoptic map never could.
                    apply_op = TcnDepthImageApplyMaskOp(
                        self,
                        allocator=device_memory_pool,
                        invert_mask=bool(join_cfg.get("invert_mask", False)),
                        name=f"join_apply_mask_{cam}")

                    check_op = JoinCheckOp(self, camera=cam, stats=join_stats,
                                           name=f"join_check_{cam}")

                    self.add_flow(depth_split, bp_op, {(f"{cam}_depthimage", "depth_image")})
                    self.add_flow(xylt_op, bp_op, {("xy_table", "xy_table")})
                    self.add_flow(mask_split, sampler_op,
                                  {(mask_name(f"{cam}_colorimage"), "labels")})
                    self.add_flow(bp_op, sampler_op, {("texcoords", "texcoords")})
                    self.add_flow(depth_split, apply_op,
                                  {(f"{cam}_depthimage", "depth_image")})
                    self.add_flow(sampler_op, apply_op, {("mask_out", "mask_image")})
                    self.add_flow(sampler_op, check_op, {("labels_out", "labels")})
                    self.add_flow(sampler_op, check_op, {("mask_out", "mask")})
                    self.add_flow(apply_op, check_op, {("output", "masked_depth")})
                    self.add_flow(depth_split, check_op,
                                  {(f"{cam}_depthimage", "raw_depth")})

                log.info(f"Mask/depth join ENABLED for {join_cams} "
                         f"(select_classes={select_classes or 'all non-background'})")
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

        if camera_streams_config.get("enable_langsam_multicam", False):
            all_color_cams = [c["name"] for c in color_streams_config]

            # gpu_workers.workers must name exactly the cameras this source provides -- on
            # ANY source (design doc "Dataset facts" / see _validate_source_cameras
            # docstring). Checked here, not earlier, because gpu_workers.workers is only
            # actually consumed when this subgraph is built -- validating it unconditionally
            # would newly break a live run that has enable_langsam_multicam off and a stale,
            # otherwise-harmless gpu_workers block.
            _validate_source_cameras(source, all_color_cams, self.kwargs("gpu_workers"))

            langsam_mc = LangSamMultiCamProcessingSubgraph(
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
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_shm_vlm_inference.yaml")
    else:
        config_file = args.config

    main(config_file=config_file, scheduler_type=args.scheduler, log_level=args.log_level, with_tracker=args.tracking)
