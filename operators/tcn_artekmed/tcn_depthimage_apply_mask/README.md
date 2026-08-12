# tcn_depthimage_apply_mask

`TcnDepthImageApplyMaskOp` — zeroes the depth pixels that a binary mask does not select.

## Ports

| port | dir | notes |
|---|---|---|
| `depth_image` | in | device uint16, single **unnamed** tensor |
| `mask_image` | in | device uint8, single **unnamed** tensor |
| `output` | out | masked depth, tensor named `out_tensor_name` |

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `invert_mask` | `false` | `true` keeps the background instead of the foreground. |
| `out_tensor_name` | `""` | |

## The input contract is strict

Both inputs are read as the entity's **unnamed** tensor, and the mask is indexed by the depth image's
element count — there is **no resampling**. So the mask must:

- be `uint8`,
- be the entity's unnamed tensor (an upstream operator emitting a named tensor will not be found),
- have the **same element count** as the depth image.

That last point is why a colour-resolution panoptic map cannot be fed here directly: it is a named
`uint16` tensor at a different resolution. Produce a depth-grid mask first —
`tcn_label_sampler`'s `mask_out` is exactly that, and leaving its `out_mask_tensor_name` empty
satisfies the unnamed requirement.

## Invariant worth checking

If the mask was derived from the same depth image (via backprojection texcoords), then every selected
pixel had valid depth, so masking must **keep** every selected pixel. A pixel that is selected but
zero after masking means the mask does not belong to this depth image — a stale or aliased buffer.
That check caught a real use-after-free in `tcn_stream_splitter`; the VLM application implements it as
`JoinCheckOp`.
