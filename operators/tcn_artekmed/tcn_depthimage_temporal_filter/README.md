# tcn_depthimage_temporal_filter

`TcnDepthImageTemporalFilterOp` — temporal noise filter for depth images, combining a persistence
window with an exponential moving average.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `input` | in | device `[H, W]` uint16 |
| `output` | out | filtered depth, tensor named `out_tensor_name` |

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `persistence` | `8` | How many recent frames a pixel may be filled in from when its current sample is invalid. Higher values fill more holes and smear more motion. |
| `delta` | `30` | Depth-difference threshold (in depth units) beyond which a new sample is treated as a real change rather than noise, so the EMA is reset instead of blended. |
| `alpha` | `0.15` | EMA weight for the new sample. Lower is smoother and laggier. |
| `in_tensor_name` / `out_tensor_name` | `""` | |
| `cuda_device_ordinal`, `cuda_stream_pool` | | |

## Trade-off

The filter trades temporal noise for motion lag, and both knobs point the same way: raising
`persistence` or lowering `alpha` gives a visibly cleaner point cloud and visibly smeared moving
objects. For a pipeline whose purpose is segmenting *people*, prefer conservative values and let the
segmentation handle the rest.

`delta` is what keeps a genuine depth discontinuity (an object arriving) from being averaged away —
tune it before reaching for `alpha`.
