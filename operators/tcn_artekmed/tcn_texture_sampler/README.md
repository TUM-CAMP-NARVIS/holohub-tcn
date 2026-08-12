# tcn_texture_sampler

`TcnTextureSamplerOp` — samples a colour image through per-pixel texture coordinates, bilinearly.

Pairs with `tcn_depthimage_backprojection`: its `texcoords` output says where each depth pixel lands
in the colour image, so sampling through them produces a colour image registered to the **depth**
grid ("warped colour").

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `color_image` | in | device `[Hc, Wc, 4]` uint8 — RGBA/BGRA; 4 channels are required |
| `texcoords` | in | device `[Hd, Wd, 2]` float32 |
| `output` | out | device `[Hd, Wd, 4]` uint8 |

## Parameters

| parameter | default |
|---|---|
| `allocator` | — |
| `in_color_tensor_name` | `""` |
| `in_texcoord_tensor_name` | `""` |
| `out_tensor_name` | `""` |
| `cuda_device_ordinal` | |
| `cuda_stream_pool` | |

## Filtering and edge behaviour

Sampling is **bilinear**, and `uv` is clamped into `[0,1]`, so a depth pixel projecting outside the
colour frustum takes the border colour (edge extension). Both are reasonable for colour.

**Neither is acceptable for label data.** Interpolating packed `(class << 8) | instance` ids produces
ids denoting none of the classes involved, and edge extension smears a border object across
everything beside it. Use `tcn_label_sampler` for panoptic maps — it is nearest-neighbour and rejects
out-of-frustum coordinates rather than clamping them.

A NaN texcoord (which backprojection writes where depth is invalid) clamps to pixel `(0,0)`, since
IEEE `fmaxf(NaN,0)` is `0`. That is deliberate and unchanged behaviour, but it means this operator
cannot distinguish "no depth" from "projects into the corner".

The `[0,1] → [0, N-1]` mapping is shared with `tcn_label_sampler`, so colour and label sampling
address the same pixel grid.
