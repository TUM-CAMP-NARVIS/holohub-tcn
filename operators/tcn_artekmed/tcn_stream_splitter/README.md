# tcn_stream_splitter

`TcnStreamSplitterOp` — routes a multi-camera entity (a map of channel name → tensor) to separate
named outputs, one per channel.

## Ports

| port | dir | notes |
|---|---|---|
| `receivers` | in | one entity holding a tensor per channel |
| `<channel_name>` | out | one port per entry in `channel_names`, each an entity with a single **unnamed** tensor |

## Parameters

| parameter | default | notes |
|---|---|---|
| `channel_names` | — | Channels to extract. **Must be a constructor argument** — the output ports are created in `setup()`, which runs before parameter values are applied. |
| `cuda_stream_pool` | — | The input's stream is propagated to every output port. |

## Behaviour

Outputs are **zero-copy views** of the input's device memory (`wrapMemory`), not copies, and each
output tensor is **unnamed** regardless of the source channel name. Downstream operators therefore
read with an empty `in_tensor_name`, which is the default for the C++ operators in this collection.

A channel named in `channel_names` that is absent from the input entity is an error, not a skip: it
means the wiring and the actual stream disagree, and continuing would silently drop a camera.

## Caveat: the source entity is kept alive deliberately

Because the outputs wrap the input's memory, the operator holds a reference to the source entity in
the release callback of each view. Without that, the allocation could be reused by a later frame
while a consumer was still reading — nondeterministic, affecting one consumer on some frames, and
invisible until a view is buffered across ticks or read by more than one consumer. See the
collection README.

## Usage

```python
from holohub.tcn_stream_splitter import TcnStreamSplitterOp as StreamSplitterOp

split = StreamSplitterOp(self, cuda_stream_pool,
                         channel_names=[f"{c}_depthimage" for c in cameras],
                         name="depth_splitter")
self.add_flow(source, split, {("depth_outputs", "receivers")})
self.add_flow(split, consumer, {("camera01_depthimage", "depth_image")})
```

Derive the channel names from what the source actually declares (the discovered channel config)
rather than by string construction — a live segment that names a channel differently otherwise fails
at the first tick inside this operator, pointing here rather than at the mismatch.
