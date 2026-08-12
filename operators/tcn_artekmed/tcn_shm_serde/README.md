# tcn_shm_serde

**Library, not an operator.** Shared-memory message types and the parameter RPC used by the
iceoryx2 camera transport.

## Contents

| file | purpose |
|---|---|
| `shm_types.hpp` | The SHM message/buffer type definitions |
| `shm_serde.hpp` | Serialisation helpers for those types |
| `shm_rpc.hpp` / `.cpp` | Parameter RPC — request/response for camera configuration and calibration |
| `generated/` | Generated type definitions; do not edit by hand |

## Role in the pipeline

`tcn_shm_subscriber` uses these types to read camera frames out of the segment, and the parameter RPC
is how `discover_shm()` retrieves each camera's device context (calibration) and the channel
configuration before any frame is received.

Because the layouts here are the wire format between the camera publisher and this pipeline, changing
them requires changing both sides together. The `generated/` directory is the authority for the
current layout.
