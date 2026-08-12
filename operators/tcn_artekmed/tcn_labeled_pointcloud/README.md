# tcn_labeled_pointcloud

`TcnLabeledPointcloudOp` — turns a labeled depth grid into one compacted point cloud per class.

Design: [`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md`](../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md)

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `positions` | in | device `[H, W, 3]` float32 — world-space points from `tcn_depthimage_backprojection` |
| `labels` | in | device `[H, W]` or `[H, W, 1]` uint16 — packed panoptic labels for the **same** grid |
| `class_<id>` | out | one per entry in `classes`; each entity holds `positions` `[1, N, 3]` float32 **and** `labels` `[1, N, 1]` uint16 |

Positions and labels must index the same depth grid — the operator refuses a mismatch, because
otherwise every point would be given some other pixel's label.

Each output carries the **packed** label, so the instance id survives into the data product rather
than being collapsed to the class.

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | Prefer an `RMMAllocator`: these outputs are numerous and variable-sized. |
| `classes` | `[]` | Class ids to emit, one `class_<id>` port each. **Must be a constructor argument.** Empty emits a single `class_all` port carrying every non-background class. |
| `cuda_device_ordinal` | `0` | |
| `in_positions_tensor_name` / `in_labels_tensor_name` | `""` | |
| `out_positions_tensor_name` | `"positions"` | Must match the downstream merger's `input_message_name`. |
| `out_labels_tensor_name` | `"labels"` | |
| `verbose` | `false` | Log per-class point counts every frame. |

## Why one port per class

`HolovizOp` colours `POINTS_3D` per `InputSpec`, not per vertex. A port per class is therefore what
lets a fused view show classes in different colours through the existing viewer. Per-point colour
would mean adding a colour vertex buffer to `tcn_slang_renderer`'s pointcloud renderable.

## Fusing across cameras

```
per camera:  positions + labels ──► tcn_labeled_pointcloud ──► class_1, class_2, ...
                                                                 │
             one tcn_stream_merger per class (all cameras) ◄──────┘
                                                                 │
                                    HolovizOp POINTS_3D spec per class
```

The merger concatenates along dimension 1, so per-frame point counts may differ between cameras. No
`tcn_flatten_tensor` is needed: it maps `[H, W, ...]` to `[1, H*W, ...]`, i.e. to exactly the
`[1, N, 3]` this operator already emits.

## Semantics

**Background is never a point.** Label 0 is background, and it is also what `tcn_label_sampler`
writes for a depth pixel with no colour correspondence.

**A class with no points still emits**, because a downstream merger needs every input every frame. It
emits a single **NaN** point, which the rasteriser culls — an empty tensor would starve the merger and
a zero position would draw a stray point at the origin. `N` therefore never reaches 0.

**Compaction preserves source order.** `cub::DeviceSelect::Flagged` is a stable scan, so identical
input yields an identical point order and a byte-comparison gate is meaningful. An atomic append
would be marginally faster and would reorder points run to run.

**One synchronisation per tick, independent of the class count.** `Flagged` writes its result count to
*device* memory, so every class is selected and compacted with the stream still running; a single copy
brings all K counts back and one `cudaStreamSynchronize` waits for them. The host round-trip cannot be
removed — each output tensor is sized to its point count before allocation — only paid once instead of
K times. (A per-class `thrust::copy_if` returns a host-side iterator, and reading it cost two blocking
calls per class.)

One selection buffer serves all classes: the per-class select and its compaction are issued on the
same stream, so stream ordering prevents the next class from overwriting flags the previous compaction
has not read. Indices are per class, because all K must remain readable after the single
synchronisation when the gathers are issued.

## Performance note

Reducing the synchronisation count cut this operator's median from 12.29 to 5.24 ms/tick live, and
changed end-to-end throughput not at all: these operators run concurrently on worker threads and were
never on the critical path — their time was spent blocked on the GPU queue *in parallel with* the real
bottleneck. Worth knowing before optimising here again: **a duration measured inside a blocking call
is queue depth, not this operator's cost.**

## Tests

```bash
PYTHONPATH=<build>/python/lib python3 tests/test_labeled_pointcloud.py      # 6 cases
PYTHONPATH=<build>/python/lib python3 tests/bench_labeled_pointcloud.py 1000 1   # micro-benchmark, device 1
```

The benchmark takes an iteration count and a device index; pick an idle device, since a few percent of
foreign load swamps what it measures.
