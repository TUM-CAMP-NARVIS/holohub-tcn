# tcn_depthimage_max_distance

`TcnDepthImageMaxDistanceOp` — per-pixel running maximum over successive depth images.

## Ports

| port | dir | notes |
|---|---|---|
| `input` | in | device uint16 depth, tensor named `in_tensor_name` |
| `output` | out | per-pixel maximum so far, tensor named `out_tensor_name` |

## Parameters

| parameter | default |
|---|---|
| `allocator` | — |
| `in_tensor_name` | `""` |
| `out_tensor_name` | `""` |

## Usage

The intended use is learning a **background reference**: run it over a stretch of frames with nothing
in the foreground, and each pixel converges on the furthest surface it has seen — the room. Feed the
result to `tcn_depthimage_fgbg_mask` as its `background_image`.

The state accumulates for the operator's lifetime, so anything transient that appears during learning
is baked in permanently. Restart the application (or the operator) to relearn.
