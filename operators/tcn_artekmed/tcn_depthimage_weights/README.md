# tcn_depthimage_weights

`TcnDepthImageWeightsOp` — per-pixel confidence weights for a depth image, from surface incidence
angle and depth limits.

Used when fusing several cameras' point clouds: a point seen at a grazing angle, or near the sensor's
range limits, is less trustworthy than one seen face-on at mid range.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `depth_image` | in | device `[H, W]` uint16 |
| `xy_table` | in | device `[H, W, 2]` float32 — **no condition**, so it need only arrive once |
| `output` | out | per-pixel weights, tensor named `out_tensor_name` |

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `depth_units_per_meter` | `1000.0` | As for backprojection: a property of the capture format. |
| `angle_reject_limit` | `π/9` (20°) | Incidence angle beyond which a point is rejected. |
| `angle_reject_envelope` | `1.0` | Softness of the angular rejection. |
| `offset_envelope` | `1.0` | Softness of the depth-limit rejection. |
| `depth_near_limit` | `0.1` | metres |
| `depth_far_limit` | `8.0` | metres |
| `in_tensor_name` / `out_tensor_name` | `""` | |
| `cuda_device_ordinal`, `cuda_stream_pool` | | |

Note the limits here are **independent** of `tcn_depthimage_backprojection`'s `near_limit_m` /
`far_limit_m` and default differently (0.1/8.0 vs the backprojection config's typical 0.01/10.0).
Keep them consistent deliberately: a point that backprojection accepts but this operator rejects gets
a position and a zero weight, which is a valid but easily surprising combination.
