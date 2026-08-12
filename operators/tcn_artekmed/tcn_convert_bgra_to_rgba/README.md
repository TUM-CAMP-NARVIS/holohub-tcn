# tcn_convert_bgra_to_rgba

`TcnConvertBgraToRgbaOp` — swaps the red and blue channels of a 4-channel 8-bit image.

## Ports

| port | dir | notes |
|---|---|---|
| `input` | in | device `[H, W, 4]` uint8, tensor named `in_tensor_name` |
| `output` | out | device `[H, W, 4]` uint8, tensor named `out_tensor_name` |

## Parameters

| parameter | default |
|---|---|
| `allocator` | — |
| `in_tensor_name` | `""` |
| `out_tensor_name` | `""` |

## When it is needed

Only when the source reports a BGRA semantic type, which is a property of the **live** camera
channel. Insert it based on the channel's declared semantic type, never unconditionally: a stream
that is already RGB/BGR-3-channel or that downstream consumers already swap themselves will be
corrupted by a second swap, and the result looks plausible enough to survive casual inspection.

`tcn_dataset_replayer` emits 3-channel BGR by design (matching what the live subscriber delivers), so
the dataset path needs no conversion.
