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


class App(hs.core.Application):
    def compose(self):

        # Add your operators here
        print("Loading Operators Successful")


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

    parser = ArgumentParser(description="ARTEKMED Holoscan All Placeholder.")

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
