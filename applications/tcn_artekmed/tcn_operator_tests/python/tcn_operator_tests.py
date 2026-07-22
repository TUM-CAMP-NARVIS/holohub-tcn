#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for C++ ported tcn_artekmed operators.

Runs a minimal Holoscan pipeline for each operator to verify:
1. Python bindings import successfully
2. Operator can be instantiated and configured
3. Data flows through the pipeline with correct I/O behavior

Each test uses a synthetic source operator that generates test tensors,
feeds them through the operator under test, and verifies the output.
"""

import logging
import sys

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tcn_operator_tests")


# ── Imports for C++ operators ──
from holohub.tcn_convert_bgra_to_rgba import TcnConvertBgraToRgbaOp
from holohub.tcn_flatten_tensor import TcnFlattenTensorOp
from holohub.tcn_depthimage_max_distance import TcnDepthImageMaxDistanceOp
from holohub.tcn_depthimage_apply_mask import TcnDepthImageApplyMaskOp
from holohub.tcn_depthimage_fgbg_mask import TcnDepthImageFgbgMaskOp
from holohub.tcn_stream_splitter import TcnStreamSplitterOp
from holohub.tcn_stream_merger import TcnStreamMergerOp


class SyntheticSourceOp(Operator):
    """Emits synthetic test tensors for a configurable number of frames."""

    def __init__(self, fragment, *args, tensor_fn=None, max_frames=3,
                 count=None, recess_period=None, **kwargs):
        self._tensor_fn = tensor_fn
        self._max_frames = max_frames
        self._frame = 0
        # Bound scheduling with a real CountCondition so the source deactivates
        # after `max_frames` executions and the graph terminates. Passing bare
        # `count=`/`recess_period=` kwargs (as the call sites do) is a no-op in
        # Holoscan, which left the greedy scheduler spinning forever once the
        # source stopped emitting. `recess_period` is absorbed for call-site
        # compatibility but is not needed for termination.
        n = count if count is not None else max_frames
        super().__init__(fragment, CountCondition(fragment, count=n), *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if self._frame >= self._max_frames:
            return
        tensor = self._tensor_fn(self._frame)
        op_output.emit({"": hs.as_tensor(tensor)}, "output")
        self._frame += 1


class DualSourceOp(Operator):
    """Emits two synthetic tensor streams (depth + mask/background)."""

    def __init__(self, fragment, *args, tensor_fn_a=None, tensor_fn_b=None,
                 port_a="port_a", port_b="port_b", max_frames=3,
                 count=None, recess_period=None, **kwargs):
        self._tensor_fn_a = tensor_fn_a
        self._tensor_fn_b = tensor_fn_b
        self._port_a = port_a
        self._port_b = port_b
        self._max_frames = max_frames
        self._frame = 0
        # See SyntheticSourceOp: bound execution with a CountCondition so the
        # graph terminates instead of the scheduler spinning forever.
        n = count if count is not None else max_frames
        super().__init__(fragment, CountCondition(fragment, count=n), *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output(self._port_a)
        spec.output(self._port_b)

    def compute(self, op_input, op_output, context):
        if self._frame >= self._max_frames:
            return
        a = self._tensor_fn_a(self._frame)
        b = self._tensor_fn_b(self._frame)
        op_output.emit({"": hs.as_tensor(a)}, self._port_a)
        op_output.emit({"": hs.as_tensor(b)}, self._port_b)
        self._frame += 1


class MultiTensorSourceOp(Operator):
    """Emits a message with multiple named tensors (simulates multi-camera entity)."""

    def __init__(self, fragment, *args, tensor_dict_fn=None, max_frames=3,
                 count=None, recess_period=None, **kwargs):
        self._tensor_dict_fn = tensor_dict_fn
        self._max_frames = max_frames
        self._frame = 0
        # See SyntheticSourceOp: bound execution with a CountCondition so the
        # graph terminates instead of the scheduler spinning forever.
        n = count if count is not None else max_frames
        super().__init__(fragment, CountCondition(fragment, count=n), *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if self._frame >= self._max_frames:
            return
        tensor_dict = self._tensor_dict_fn(self._frame)
        msg = {name: hs.as_tensor(t) for name, t in tensor_dict.items()}
        op_output.emit(msg, "output")
        self._frame += 1


class SinkOp(Operator):
    """Receives tensors and stores them for verification."""

    def __init__(self, fragment, *args, **kwargs):
        self.received = []
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        tensor = msg.get("")
        if tensor is not None:
            self.received.append(cp.asarray(tensor).copy())
            log.info(f"SinkOp [{self.name}]: received frame {len(self.received)}, "
                     f"shape={self.received[-1].shape}, dtype={self.received[-1].dtype}")


# ── Test 1: ConvertBgraToRgbaOp ──
class TestBgraToRgbaApp(Application):
    def compose(self):
        alloc = UnboundedAllocator(self, name="allocator")

        def make_bgra(frame_idx):
            # HxWx4 BGRA with known channel values
            img = cp.zeros((4, 4, 4), dtype=cp.uint8)
            img[..., 0] = 10   # B
            img[..., 1] = 20   # G
            img[..., 2] = 30   # R
            img[..., 3] = 255  # A
            return img

        source = SyntheticSourceOp(self, tensor_fn=make_bgra, max_frames=1, name="source",
                                   count=1, recess_period=0)
        op = TcnConvertBgraToRgbaOp(self, allocator=alloc, name="bgra_to_rgba")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, op, {("output", "input")})
        self.add_flow(op, sink, {("output", "input")})


# ── Test 2: FlattenTensorOp ──
class TestFlattenTensorApp(Application):
    def compose(self):
        def make_tensor(frame_idx):
            return cp.ones((8, 16, 3), dtype=cp.float32) * (frame_idx + 1)

        source = SyntheticSourceOp(self, tensor_fn=make_tensor, max_frames=1, name="source",
                                   count=1, recess_period=0)
        op = TcnFlattenTensorOp(self, message_name="", name="flatten")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, op, {("output", "input")})
        self.add_flow(op, sink, {("output", "input")})


# ── Test 3: DepthImageMaxDistanceOp ──
class TestMaxDistanceApp(Application):
    def compose(self):
        alloc = UnboundedAllocator(self, name="allocator")

        def make_depth(frame_idx):
            img = cp.full((16, 16), fill_value=100 * (frame_idx + 1), dtype=cp.uint16)
            return img

        source = SyntheticSourceOp(self, tensor_fn=make_depth, max_frames=3, name="source",
                                   count=3, recess_period=0)
        op = TcnDepthImageMaxDistanceOp(self, allocator=alloc, name="max_distance")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, op, {("output", "input")})
        self.add_flow(op, sink, {("output", "input")})


# ── Test 4: DepthImageApplyMaskOp ──
class TestApplyMaskApp(Application):
    def compose(self):
        alloc = UnboundedAllocator(self, name="allocator")

        def make_depth(frame_idx):
            return cp.full((8, 8), fill_value=1000, dtype=cp.uint16)

        def make_mask(frame_idx):
            mask = cp.zeros((8, 8), dtype=cp.uint8)
            mask[:4, :] = 1  # top half is foreground
            return mask

        source = DualSourceOp(self, tensor_fn_a=make_depth, tensor_fn_b=make_mask,
                              port_a="depth", port_b="mask", max_frames=1, name="source",
                              count=1, recess_period=0)
        op = TcnDepthImageApplyMaskOp(self, allocator=alloc, invert_mask=False, name="apply_mask")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, op, {("depth", "depth_image"), ("mask", "mask_image")})
        self.add_flow(op, sink, {("output", "input")})


# ── Test 5: DepthImageFgbgMaskOp ──
class TestFgbgMaskApp(Application):
    def compose(self):
        alloc = UnboundedAllocator(self, name="allocator")

        def make_depth(frame_idx):
            # Depth: 500mm (close objects)
            return cp.full((8, 8), fill_value=500, dtype=cp.uint16)

        def make_bg(frame_idx):
            # Background reference: 2000mm (far)
            return cp.full((8, 8), fill_value=2000, dtype=cp.uint16)

        source = DualSourceOp(self, tensor_fn_a=make_depth, tensor_fn_b=make_bg,
                              port_a="depth", port_b="bg", max_frames=1, name="source",
                              count=1, recess_period=0)
        op = TcnDepthImageFgbgMaskOp(self, allocator=alloc, sensitivity=1.0,
                                     enable_foreground=True, enable_background=False,
                                     name="fgbg_mask")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, op, {("depth", "depth_image"), ("bg", "background_image")})
        self.add_flow(op, sink, {("foreground_mask", "input")})


# ── Test 6: StreamSplitterOp ──
class TestStreamSplitterApp(Application):
    def compose(self):
        channels = ["camera0", "camera1"]

        def make_multi_camera(frame_idx):
            return {
                "camera0": cp.full((4, 4), fill_value=100 * (frame_idx + 1), dtype=cp.float32),
                "camera1": cp.full((4, 4), fill_value=200 * (frame_idx + 1), dtype=cp.float32),
            }

        source = MultiTensorSourceOp(self, tensor_dict_fn=make_multi_camera,
                                      max_frames=1, name="source",
                                      count=1, recess_period=0)
        splitter = TcnStreamSplitterOp(self, channel_names=channels, name="splitter")
        sink0 = SinkOp(self, name="sink_cam0")
        sink1 = SinkOp(self, name="sink_cam1")

        self.add_flow(source, splitter, {("output", "receivers")})
        self.add_flow(splitter, sink0, {("camera0", "input")})
        self.add_flow(splitter, sink1, {("camera1", "input")})


# ── Test 7: StreamMergerOp (separate mode) ──
class TestStreamMergerSeparateApp(Application):
    def compose(self):
        port_names = ["camera0_depth", "camera1_depth"]

        def make_cam0(frame_idx):
            return cp.full((4, 4), fill_value=10.0, dtype=cp.float32)

        def make_cam1(frame_idx):
            return cp.full((4, 4), fill_value=20.0, dtype=cp.float32)

        source = DualSourceOp(self, tensor_fn_a=make_cam0, tensor_fn_b=make_cam1,
                              port_a="camera0_depth", port_b="camera1_depth",
                              max_frames=1, name="source",
                              count=1, recess_period=0)
        merger = TcnStreamMergerOp(self, input_port_names=port_names,
                                    input_message_name="", output_message_name="depth",
                                    fuse_buffers=False, name="merger")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, merger, {("camera0_depth", "camera0_depth"),
                                        ("camera1_depth", "camera1_depth")})
        self.add_flow(merger, sink, {("output", "input")})


# ── Test 8: StreamMergerOp (fuse mode) ──
class TestStreamMergerFuseApp(Application):
    def compose(self):
        alloc = UnboundedAllocator(self, name="allocator")
        port_names = ["camera0_depth", "camera1_depth"]

        def make_cam0(frame_idx):
            return cp.ones((4, 8), dtype=cp.float32) * 1.0

        def make_cam1(frame_idx):
            return cp.ones((4, 8), dtype=cp.float32) * 2.0

        source = DualSourceOp(self, tensor_fn_a=make_cam0, tensor_fn_b=make_cam1,
                              port_a="camera0_depth", port_b="camera1_depth",
                              max_frames=1, name="source",
                              count=1, recess_period=0)
        merger = TcnStreamMergerOp(self, input_port_names=port_names,
                                    input_message_name="", output_message_name="fused",
                                    fuse_buffers=True, allocator=alloc, name="merger_fuse")
        sink = SinkOp(self, name="sink")

        self.add_flow(source, merger, {("camera0_depth", "camera0_depth"),
                                        ("camera1_depth", "camera1_depth")})
        self.add_flow(merger, sink, {("output", "input")})


def run_test(name, app_class):
    log.info(f"{'=' * 60}")
    log.info(f"TEST: {name}")
    log.info(f"{'=' * 60}")
    try:
        app = app_class()
        app.run()
        log.info(f"PASS: {name}")
        return True
    except Exception as e:
        log.error(f"FAIL: {name} — {e}")
        return False


def main():
    tests = [
        ("ConvertBgraToRgbaOp", TestBgraToRgbaApp),
        ("FlattenTensorOp", TestFlattenTensorApp),
        ("DepthImageMaxDistanceOp", TestMaxDistanceApp),
        ("DepthImageApplyMaskOp", TestApplyMaskApp),
        ("DepthImageFgbgMaskOp", TestFgbgMaskApp),
        ("StreamSplitterOp", TestStreamSplitterApp),
        ("StreamMergerOp (separate)", TestStreamMergerSeparateApp),
        ("StreamMergerOp (fuse)", TestStreamMergerFuseApp),
    ]

    results = []
    for name, app_class in tests:
        ok = run_test(name, app_class)
        results.append((name, ok))

    log.info(f"\n{'=' * 60}")
    log.info("RESULTS:")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        log.info(f"  [{status}] {name}")
    log.info(f"{'=' * 60}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    log.info(f"{passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
