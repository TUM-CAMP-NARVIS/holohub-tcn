# tcn_stream_merger

`TcnStreamMergerOp` — merges several input entities into one output entity, optionally concatenating
their tensors into a single buffer.

## Ports

| port | dir | notes |
|---|---|---|
| `<input_port_names[i]>` | in | one port per entry; each must carry a tensor named `input_message_name` |
| `output` | out | one entity holding the merged result as `output_message_name` |

Every input is required every tick.

## Parameters

| parameter | default | notes |
|---|---|---|
| `input_port_names` | — | Input ports to create. **Must be a constructor argument** (ports are created in `setup()`). |
| `input_message_name` | — | Tensor name to read from each input entity. |
| `output_message_name` | — | Tensor name to write in the output entity. |
| `fuse_buffers` | `false` | `true` concatenates the inputs into one tensor; `false` collects them as separate tensors in one entity. |
| `allocator` | `nullptr` | Required when `fuse_buffers` is set, since fusing allocates. |
| `cuda_stream_pool` | — | |

## Fusing

Concatenation is along **dimension 1**. All other dimensions must match across inputs, so
`[H, W1, C]` and `[H, W2, C]` fuse to `[H, W1+W2, C]` — per-frame widths may differ, which is what
lets variable-sized point clouds be fused.

Two shapes this is used for:

- **Per-camera images** `[H, W, C]` → `[H, W*n, C]`, as in `tcn_shm_receiver`'s point fusion.
- **Point clouds** `[1, N_i, 3]` → `[1, ΣN_i, 3]`, which is already the shape `HolovizOp`
  `POINTS_3D` expects, so no flatten step is needed afterwards.

The input entities are kept alive until the emit, since the merged tensor references their memory.

## Usage

```python
from holohub.tcn_stream_merger import TcnStreamMergerOp as StreamMergerOp

merge = StreamMergerOp(self, cuda_stream_pool,
                       input_port_names=[f"{c}_class_1" for c in cameras],
                       input_message_name="positions",
                       output_message_name="class_1",
                       fuse_buffers=True,
                       allocator=pool,
                       name="cloud_fusion_1")
```
