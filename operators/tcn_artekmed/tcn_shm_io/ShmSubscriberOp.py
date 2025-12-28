# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from .shm_serde import shm_transport_enum

log = logging.getLogger("TcnShmReaderOp")

from typing import Any
import queue
from concurrent.futures import Future, ThreadPoolExecutor

import holoscan as hs
import numpy as np
import cupy as cp

from holoscan.conditions import AsynchronousCondition, AsynchronousEventState
from holoscan.core import Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.operators import holoviz

from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, RigidTransform, CameraParameters, make_rigid_transform


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
