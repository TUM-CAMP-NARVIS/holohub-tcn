# tcn_shm_zenoh_sender

`TcnShmZenohSenderOp` — publishes decoded frames into an iceoryx2 shared-memory segment.

## Ports

| port | dir | notes |
|---|---|---|
| `frame_input` | in | entity holding the tensors named in `input_tensor_names` |

A sink: it has no outputs.

## Parameters

| parameter | notes |
|---|---|
| `stream_name` | SHM stream to publish on, e.g. `"camera_streams"`. |
| `input_tensor_names` | Which tensors of the input entity to publish, in order. Their names become the segment's channel names. |

## Role in the pipeline

The counterpart of `tcn_shm_subscriber`: it makes frames that arrived over the network (or were
produced locally) available to any process attached to the segment, which is how a decoded stream is
handed to the ARTEKMED pipeline without a second network hop.

Because `input_tensor_names` becomes the channel naming that downstream subscribers discover, keep it
consistent with the `<camera>_colorimage` / `<camera>_depthimage` convention documented in the
collection README — a splitter downstream addresses channels by exactly these names.
