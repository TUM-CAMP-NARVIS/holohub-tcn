# tcn_flatten_tensor

`TcnFlattenTensorOp` — reshapes a tensor from `[H, W, ...]` to `[1, H*W, ...]` without copying.

## Ports

| port | dir | notes |
|---|---|---|
| `input` | in | entity holding a tensor named `message_name` |
| `output` | out | entity holding the reshaped view under the same name |

## Parameters

| parameter | default | notes |
|---|---|---|
| `message_name` | `""` | Tensor name, used for **both** the input lookup and the output. |
| `allocator` | — | |
| `cuda_stream_pool` | — | |

## Note the exact shape

The result **keeps a leading 1**: `[H, W, C]` becomes `[1, H*W, C]`, not `[H*W, C]`. That is the
shape `HolovizOp` `POINTS_3D` consumes in this codebase, which is why the point-cloud path in
`tcn_shm_receiver` ends with this operator.

A producer that already emits `[1, N, C]` gains nothing here — the reshape is an exact no-op. Check
the producer's shape before inserting this operator.

Inputs with rank < 2 are rejected with a warning and no output.

## Caveat: zero-copy view

The output wraps the input's device memory. The operator holds a reference to the source entity in
the view's release callback; without it, the allocation could be reused by a later frame while a
consumer was still reading the "flattened" view. See the collection README.
