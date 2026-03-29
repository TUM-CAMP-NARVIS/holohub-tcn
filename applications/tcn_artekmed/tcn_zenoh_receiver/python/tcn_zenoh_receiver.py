#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import logging
from argparse import ArgumentParser

import holoscan as hs

from holoscan.logger import LogLevel, set_log_level

from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.conditions import AsynchronousCondition
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.schedulers import EventBasedScheduler

from holohub.tcn_zenoh_receiver import (
    TcnZenohReceiverOp,
    ZenohStreamConfig,
    open_zenoh_session,
    discover_streams,
)
from holohub.tcn_shm_zenoh_sender import TcnShmZenohSenderOp

log = logging.getLogger(__name__)


class StatsOp(Operator):
    """Print common streaming statistics."""

    def __init__(self, app, *args, **kwargs):
        self.encode_latency = []
        self.decode_latency = []
        self.jitter_time = []
        self.fps = []
        self.first_frame_ignored = False
        self._logger = logging.getLogger(__name__)
        super().__init__(app, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        if not self.first_frame_ignored:
            self.first_frame_ignored = True
            return

        if hasattr(self, "metadata"):
            self.encode_latency.append(self.metadata.get("video_encoder_encode_latency_ms", 0))
            self.decode_latency.append(self.metadata.get("video_decoder_decode_latency_ms", 0))
            self.jitter_time.append(self.metadata.get("jitter_time", 0))
            self.fps.append(self.metadata.get("fps", 0))

    def stop(self):
        if self.encode_latency:
            self._logger.info(
                f"Encode Latency (ms) (min, max, avg): {min(self.encode_latency):.3f}, {max(self.encode_latency):.3f}, {sum(self.encode_latency) / len(self.encode_latency):.3f}"
            )
        if self.decode_latency:
            self._logger.info(
                f"Decode Latency (ms) (min, max, avg): {min(self.decode_latency):.3f}, {max(self.decode_latency):.3f}, {sum(self.decode_latency) / len(self.decode_latency):.3f}"
            )
        if self.jitter_time:
            self._logger.info(
                f"Jitter Time (ms) (min, max, avg): {min(self.jitter_time):.3f}, {max(self.jitter_time):.3f}, {sum(self.jitter_time) / len(self.jitter_time):.3f}"
            )
        if self.fps:
            self._logger.info(
                f"FPS (min, max, avg): {min(self.fps):.3f}, {max(self.fps):.3f}, {sum(self.fps) / len(self.fps):.3f}"
            )


class App(hs.core.Application):
    def compose(self):
        print("Starting TCN Zenoh Receiver (Python + C++ operators via pybind11)")

        zenoh_config = self.kwargs("zenoh")
        topic_prefix = zenoh_config.get("topic_prefix")
        capture_node = zenoh_config.get("capture_node")
        zenoh_config_file = zenoh_config.get("zenoh_config_file", "")
        stream_types = zenoh_config.get("stream_types", ["color", "depth"])

        try:
            shm_config = self.kwargs("shared_memory")
            shm_stream_name = shm_config.get("stream_name", "camera_streams")
            enable_shm_output = shm_config.get("enabled", False)
        except Exception:
            shm_stream_name = "camera_streams"
            enable_shm_output = False

        # Open a C++ Zenoh session (shared with the C++ operator).
        # Stored on self to prevent garbage collection while the operator holds a reference.
        try:
            self.zenoh_session = open_zenoh_session(zenoh_config_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open Zenoh session (config: {zenoh_config_file}): {e}"
            ) from e

        # Discover camera streams via C++ Zenoh RPC (same protocol as C++ app)
        stream_configs = discover_streams(
            self.zenoh_session, topic_prefix, capture_node, stream_types
        )
        if not stream_configs:
            raise RuntimeError("No streams discovered via Zenoh")

        print(f"Discovered {len(stream_configs)} streams:")
        for cfg in stream_configs:
            print(f"  {cfg.name} -> {cfg.topic} ({cfg.image_width}x{cfg.image_height}, "
                  f"compression={cfg.image_compression})")

        # Single composite C++ operator: subscribe + CDR decode + GPU upload
        async_cond = AsynchronousCondition(self, name="zenoh_receiver_async")
        allocator = UnboundedAllocator(self, name="zenoh_receiver_allocator")
        cuda_stream_pool = CudaStreamPool(
            self,
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=len(stream_configs),
            max_size=64,
            name="zenoh_receiver_cuda_pool",
        )

        receiver = TcnZenohReceiverOp(
            self,
            async_condition=async_cond,
            allocator=allocator,
            cuda_stream_pool=cuda_stream_pool,
            name="zenoh_receiver",
        )
        receiver.set_stream_configs(stream_configs)
        receiver.set_session(self.zenoh_session)
        receiver.init_spec()

        # Per-stream output routing: connect each dynamic output port to a sink
        for cfg in stream_configs:
            stats = StatsOp(self, name=f"stats_{cfg.name}")
            self.add_flow(receiver, stats, {(cfg.name, "input")})

            if enable_shm_output:
                shm_sender = TcnShmZenohSenderOp(
                    self,
                    stream_name=f"{shm_stream_name}/{cfg.name}",
                    input_tensor_names=[""],
                    name=f"shm_sender_{cfg.name}",
                )
                self.add_flow(receiver, shm_sender, {(cfg.name, "frame_input")})


def main(config_file=None):
    logging.basicConfig(level=logging.DEBUG)
    set_log_level(LogLevel.INFO)

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

    parser = ArgumentParser(description="ARTEKMED Holoscan Client.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "tcn_zenoh_receiver.yaml")
    else:
        config_file = args.config

    main(config_file=config_file)
