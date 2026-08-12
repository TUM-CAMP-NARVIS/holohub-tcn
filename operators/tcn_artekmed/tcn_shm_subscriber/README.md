# tcn_shm_subscriber

`TcnShmSubscriberOp` — live multi-camera ingest from an iceoryx2 shared-memory segment.

## Ports

| port | dir | notes |
|---|---|---|
| `color_outputs` | out | one entity holding a tensor per colour channel, keyed `<camera>_colorimage` |
| `depth_outputs` | out | one entity holding a tensor per depth channel, keyed `<camera>_depthimage` |

Both entities are emitted every tick. If the segment carries no depth channels, the depth entity is
emitted empty — a downstream splitter will then report a missing tensor, which is the intended,
loud failure.

## Parameters

| parameter | notes |
|---|---|
| `allocator` | Device allocator for the uploaded frames. |
| `async_condition` | `AsynchronousCondition`; the receiver thread flips it to `EVENT_DONE` when a frame is ready and back to `EVENT_WAITING` after the emit. |
| `receiver` | The receiver handle from `discover_shm()`. |
| `stream_name` | SHM stream to attach to, e.g. `"camera_streams"`. |
| `cycle_time_ms` | Receiver poll interval. Default `1`. |

## Discovery

`discover_shm(stream_name)` is a free function in the same module and returns everything the
application needs to build the graph before any frame arrives:

```python
from holohub.tcn_shm_subscriber._tcn_shm_subscriber import discover_shm

cfg = discover_shm("camera_streams")
cfg["receiver"]          # pass to the operator
cfg["camera_names"]      # discovered cameras
cfg["device_contexts"]   # per-camera calibration -> DeviceContextService.create(...)
cfg["channels_config"]   # {"ports": [{"name", "status": {"portType", "bufferInfo"}}, ...]}
```

`channels_config` is the authority on what the segment actually provides: `portType` is
`"colorimage"` or `"depthimage"`, and `bufferInfo.frameSize` sizes the memory pool. Derive channel
names from it rather than constructing them, so a differently-named channel fails at compose time
with a clear message instead of at the first tick.

`device_contexts` is keyed by camera name and shaped for
`DeviceContextService.create()` — see `tcn_device_context`.

## Frame lifetime and the SHM release barrier

SHM data pointers are valid only while the frame handle is alive, and the CUDA DMA engine reads from
them asynchronously. The operator therefore synchronises its copy stream **before** releasing the
segment. Do not move the release earlier: the corruption it would cause is asynchronous, partial and
frame-dependent.

## Acquisition timestamps

The operator attaches a `nvidia::gxf::Timestamp` to **both** entities, with `acqtime` taken from the
publisher's frame timestamp and `pubtime` from the local steady clock. This is the origin of frame
identity for the whole pipeline — nothing in Holoscan creates one — and `tcn_stream_synchronizer`
depends on it. The epoch is the publisher's clock and is only comparable against itself.

If the publisher never sets a timestamp, every frame carries the same value; the synchroniser then
rejects non-advancing timestamps and publishes nothing, reporting `non-monotonic` rising with
`published 0 groups`. That is the first thing to check when a live run produces no groups.
