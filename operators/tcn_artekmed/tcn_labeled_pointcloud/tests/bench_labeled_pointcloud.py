"""Micro-benchmark for TcnLabeledPointcloudOp: per-tick cost at a realistic depth-grid size.

    PYTHONPATH=<build>/python/lib python3 tests/bench_labeled_pointcloud.py [iters]

Exists to A/B the compaction strategy. The operator's cost is dominated by host synchronisation, not
arithmetic, so a wall-clock-per-tick figure at the real grid size and class count is the number that
matters; the 640x576 grid and 5 classes match the live 5-camera configuration.
"""
import sys
import time

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import CudaStreamPool, RMMAllocator

from holohub.tcn_labeled_pointcloud import TcnLabeledPointcloudOp

H, W = 576, 640                      # the live depth grid
CLASSES = [1, 2, 3, 4, 5]            # five prompts, as configured live
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# Pick an IDLE device: the host GPU 0 also runs the shm publisher, whose ~12% load
# swamps the difference being measured here.
DEV = int(sys.argv[2]) if len(sys.argv) > 2 else 0

rng = np.random.default_rng(7)
# ~16% of pixels labeled, matching the live measurement, spread over the classes so the per-class
# counts are realistic rather than all-or-nothing.
labels = np.zeros((H, W), np.uint16)
n_lab = int(0.165 * H * W)
idx = rng.choice(H * W, size=n_lab, replace=False)
cls = rng.choice(CLASSES, size=n_lab, p=[0.80, 0.02, 0.13, 0.02, 0.03])
inst = rng.integers(1, 6, size=n_lab).astype(np.uint16)
labels.reshape(-1)[idx] = (cls.astype(np.uint16) << 8) | inst
positions = rng.random((H, W, 3), dtype=np.float32)


class SourceOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        with cp.cuda.Device(DEV):
            self.pos = cp.asarray(positions)
            self.lab = cp.asarray(labels)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("positions")
        spec.output("labels")

    def compute(self, op_input, op_output, context):
        op_output.emit({"pos": hs.as_tensor(self.pos)}, "positions")
        op_output.emit({"lab": hs.as_tensor(self.lab)}, "labels")


class SinkOp(Operator):
    def __init__(self, fragment, *args, classes, timings, **kwargs):
        self.classes = list(classes)
        self.timings = timings
        self.prev = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for cls_id in self.classes:
            spec.input(f"class_{cls_id}")

    def compute(self, op_input, op_output, context):
        for cls_id in self.classes:
            op_input.receive(f"class_{cls_id}")
        now = time.perf_counter()
        if self.prev is not None:
            self.timings.append((now - self.prev) * 1000.0)
        self.prev = now


class Bench(Application):
    def __init__(self, timings):
        self.timings = timings
        super().__init__()

    def compose(self):
        pool = RMMAllocator(self, name="pool", dev_id=DEV,
                            device_memory_initial_size="512MB",
                            device_memory_max_size="2GB")
        # Must be on DEV. Without an explicit pool Holoscan hands out a stream belonging to device 0,
        # and launching on it while DEV is current fails with "invalid device ordinal".
        streams = CudaStreamPool(self, name="streams", dev_id=DEV, stream_flags=0,
                                 stream_priority=0, reserved_size=1, max_size=8)
        src = SourceOp(self, CountCondition(self, count=ITERS), name="src")
        op = TcnLabeledPointcloudOp(
            self, streams, allocator=pool, classes=CLASSES, cuda_device_ordinal=DEV,
            in_positions_tensor_name="pos", in_labels_tensor_name="lab", name="cloud")
        sink = SinkOp(self, classes=CLASSES, timings=self.timings, name="sink")
        self.add_flow(src, op, {("positions", "positions"), ("labels", "labels")})
        for cls_id in CLASSES:
            self.add_flow(op, sink, {(f"class_{cls_id}", f"class_{cls_id}")})


if __name__ == "__main__":
    timings = []
    Bench(timings).run()
    if len(timings) < 20:
        print(f"only {len(timings)} samples; increase iters")
        sys.exit(1)
    warm = timings[10:]                   # drop first-tick allocation and CUB sizing
    warm_sorted = sorted(warm)
    print(f"dev{DEV}: grid {H}x{W} = {H*W} points, {len(CLASSES)} classes, "
          f"{100.0*n_lab/(H*W):.1f}% labeled, {len(warm)} warm samples")
    print(f"  mean   {sum(warm)/len(warm):7.3f} ms/tick")
    print(f"  median {warm_sorted[len(warm_sorted)//2]:7.3f} ms/tick")
    print(f"  p90    {warm_sorted[int(0.9*len(warm_sorted))]:7.3f} ms/tick")
    print(f"  min    {warm_sorted[0]:7.3f} ms/tick")
