# tcn_depthimage_backprojection

`TcnDepthImageBackprojectionOp` — unprojects a depth image into world-space points and, independently,
into colour-image texture coordinates.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `depth_image` | in | device `[H, W]` uint16 |
| `xy_table` | in | device `[H, W, 2]` float32 — normalised ray directions; **no condition**, so it need only arrive once |
| `positions` | out | device `[H, W, 3]` float32 — world space (depth camera → `depth_extrinsics`) |
| `texcoords` | out | device `[H, W, 2]` float32 — normalised colour-image coordinates per depth pixel |
| `depth_float` | out | device `[H, W]` float32 — depth in metres |

Each output is allocated and emitted only when its `enable_*` parameter is set.

The `xy_table` comes from `XYLookupTableSourceOp` (see `tcn_device_context`), which emits once under a
`CountCondition`; this port carries `ConditionType::kNone` so later ticks are not gated on it.

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `depth_units_per_meter` | `1000.0` | Property of the capture format (Azure Kinect / Orbbec write millimetres). A wrong value scales every point linearly. |
| `near_limit_m` / `far_limit_m` | — | Depth samples outside the range are invalid; see below. |
| `color_image_width` / `color_image_height` | `1920` / `1080` | Used to normalise the projected pixel into `[0,1]`. |
| `depth_extrinsics` | — | Depth camera → world. |
| `depth_to_color` | — | Depth camera → colour camera. `DeviceContextService::get_color_to_depth_inv()` supplies it. |
| `color_params` | — | `nvidia::gxf::CameraModel` for the colour camera, including distortion. |
| `in_tensor_name` / `out_tensor_name` | `""` | `out_tensor_name` names the tensor in **every** enabled output entity. |
| `enable_positions` / `enable_texcoords` | `true` | Independent. |
| `enable_depth_float` | `false` | |
| `cuda_device_ordinal`, `cuda_stream_pool` | | |

## Geometry

```
depth pixel (x,y) + depth d  ──xy_table──►    3D point, depth-camera space
                             ──depth_extrinsics──►  positions   (world space)
                             ──depth_to_color──►    point, colour-camera space
                             ──project (distortion-aware)──►  texcoords (normalised)
```

Both outputs come from the same unprojected point but are computed independently — the two
`enable_*` flags do not gate each other. (They used to: texcoords were written only inside the
positions branch, so `enable_positions=false, enable_texcoords=true` emitted an untouched buffer.)

## Invalid depth writes NaN texcoords

A pixel whose depth is NaN/inf or outside `[near_limit_m, far_limit_m]` gets `positions = (0,0,0)`
and `texcoords = NaN`, **not** `(0,0)`. `(0,0)` is a legitimate texcoord — the colour image's
top-left pixel — so an invalid pixel used to be indistinguishable from one that genuinely projects
into the corner, and a label sampler would hand it that corner's value.

This is bit-identical for the bilinear colour path: `tcn_texture_sampler` clamps with
`fminf(fmaxf(u,0),1)`, and IEEE `fmaxf(NaN,0)` returns `0`, so a NaN texcoord clamps to pixel `(0,0)`
exactly as `(0,0)` did. Only a consumer that checks for NaN sees a difference —
`tcn_label_sampler` does, and treats it as "no label".

## Consumers

- `positions` → `tcn_labeled_pointcloud`, or fused with `tcn_stream_merger` for an unlabeled cloud.
- `texcoords` → `tcn_texture_sampler` (colour) or `tcn_label_sampler` (labels).
