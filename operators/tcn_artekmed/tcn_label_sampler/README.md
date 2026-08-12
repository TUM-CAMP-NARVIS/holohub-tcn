# tcn_label_sampler

`TcnLabelSamplerOp` — gives every depth pixel the panoptic label of the scene point it observes.

Design: [`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md)

## Why

Masks are produced at colour resolution; the depth map is a different sensor with different
resolution, intrinsics, distortion and optical centre. **Rescaling a mask onto the depth grid is not
registration** — it runs, looks plausible, and is spatially wrong in a depth-dependent way that no
test on aggregate statistics catches.

The correct correspondence already exists as `tcn_depthimage_backprojection`'s `texcoords`:
normalised colour-image coordinates per depth pixel, computed by unprojecting with the depth
intrinsics, transforming into colour-camera space, and projecting with the colour intrinsics and
distortion. This operator samples the panoptic map through them.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `labels` | in | device `[Hc, Wc]` or `[Hc, Wc, 1]` uint16 — the packed panoptic map |
| `texcoords` | in | device `[Hd, Wd, 2]` float32 — from backprojection |
| `labels_out` | out | device `[Hd, Wd, 1]` uint16 — packed label per depth pixel |
| `mask_out` | out | device `[Hd, Wd, 1]` uint8 — 255 where a selected class was hit, else 0 |

Both outputs are always emitted.

`mask_out` is what makes `tcn_depthimage_apply_mask` connectable: that operator needs a single
**unnamed** uint8 mask with the **same element count** as the depth image, which a colour-resolution
panoptic map can never be. Leave `out_mask_tensor_name` empty for that consumer.

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | For the two outputs. |
| `cuda_device_ordinal` | `0` | See the collection README on device selection. |
| `in_labels_tensor_name` | `""` | Tensor to read from the `labels` entity. |
| `in_texcoord_tensor_name` | `""` | Tensor to read from the `texcoords` entity. |
| `out_labels_tensor_name` | `""` | |
| `out_mask_tensor_name` | `""` | `""` is what `tcn_depthimage_apply_mask` requires. |
| `select_classes` | `[]` | Class ids `mask_out` marks. Each must be in `[0, 255]`; out of range raises at `start()`. Empty selects every non-background class. |
| `unlabeled_value` | `0` | Written where a depth pixel has no colour correspondence. Must fit uint16. |

## Semantics

**Nearest neighbour, never interpolation.** Averaging four packed `(class << 8) | instance` ids
produces an id denoting none of the classes involved, so bilinear filtering would invent a label at
every object boundary. Label images admit exactly one filter. The `[0,1] → [0, N-1]` mapping matches
`tcn_texture_sampler`, so colour and label sampling address the same pixel grid; the sample is
rounded rather than floored.

**Out-of-frustum texcoords are rejected, not clamped.** Edge extension is reasonable for colour and
wrong for labels — it would smear a border object across everything beside it.

**Two rejections, both yielding `unlabeled_value` and mask 0:** a non-finite texcoord (backprojection
writes NaN where depth is invalid or outside the near/far limits) and a texcoord outside `[0,1]`.
Rejection is tracked separately from the sampled label, so a non-zero `unlabeled_value` that collides
with a real class still masks as unselected.

## Tests

```bash
PYTHONPATH=<build>/python/lib python3 tests/test_label_sampler.py     # 5 cases
```

Covers exact pixel addressing, nearest-not-bilinear at a midpoint, NaN and out-of-range rejection,
`select_classes` filtering only the mask, and the colliding `unlabeled_value` case.
