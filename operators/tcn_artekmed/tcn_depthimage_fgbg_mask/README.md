# tcn_depthimage_fgbg_mask

`TcnDepthImageFgbgMaskOp` — foreground and background masks, by comparing a depth image against a
background reference.

## Ports

| port | dir | notes |
|---|---|---|
| `depth_image` | in | device uint16 |
| `background_image` | in | device uint16 — the reference, typically from `tcn_depthimage_max_distance` |
| `foreground_mask` | out | emitted when `enable_foreground` |
| `background_mask` | out | emitted when `enable_background` |

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `sensitivity` | `1.0` | Error sensitivity factor: how far from the background a pixel must be to count as foreground. |
| `enable_foreground` | `true` | |
| `enable_background` | `false` | |

## Usage

Pair with `tcn_depthimage_max_distance` to learn the background: run the max-distance operator over a
scene with no foreground to accumulate a per-pixel far surface, then feed that as
`background_image` here. The resulting mask is a cheap, purely geometric alternative to a learned
segmentation — it separates "something is in front of the room" from "the room", and cannot tell you
*what* the something is.
