# tcn_zenoh_receiver

`TcnZenohReceiverOp` — composite source operator: Zenoh subscription, CDR decode and GPU tensor
output in one step.

## Ports

One output port **per configured stream**, named `<sensor_name>_<type>` (e.g.
`camera01_colorimage`), each carrying a device tensor entity. The ports are created from the stream
configuration, so they exist before any message arrives.

## Parameters

| parameter | notes |
|---|---|
| `async_condition` | `AsynchronousCondition` driven by the subscription callbacks. |
| `allocator` | Device allocator for the decoded frames. |
| `cuda_stream_pool` | |

## Stream discovery

Each stream's configuration comes from the sensor's **descriptor message**, which supplies the topic,
`image_width`, `image_height`, `image_step`, `image_format`, `image_compression` and `frame_rate`.
Where a descriptor is unavailable, the operator falls back to the message's own metadata fields for
topic and dimensions.

This matters when wiring: the tensor names and shapes are determined by what the publisher declares,
not by local configuration, so a publisher change alters this operator's output contract.

## Relationship to the SHM path

This is the network-transport sibling of `tcn_shm_subscriber`. The SHM path is the one the ARTEKMED
pipeline uses for local cameras; this one is for streams arriving over Zenoh. `tcn_shm_zenoh_sender`
goes the other way, republishing decoded frames into a shared-memory segment.
