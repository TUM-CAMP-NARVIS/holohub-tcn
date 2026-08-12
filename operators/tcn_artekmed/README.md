# TCN ARTEKMED operators

Holoscan operators for the ARTEKMED multi-camera pipeline: shared-memory camera ingest, depth
unprojection, vision-language segmentation, temporal frame grouping, and point-cloud fusion.

Each operator has its own README with ports, parameters and caveats. This page covers the
**conventions they share**, which is the part you cannot discover by reading one operator.

## Operators

### Frame sources and camera metadata

| operator | purpose |
|---|---|
| [tcn_shm_subscriber](tcn_shm_subscriber/README.md) | Live camera ingest from an iceoryx2 shared-memory segment; emits colour and depth entities |
| [tcn_dataset_replayer](tcn_dataset_replayer/README.md) | Deterministic drop-in replacement for the subscriber, replaying an on-disk export |
| [tcn_device_context](tcn_device_context/README.md) | Camera calibration service (`DeviceContextService`) and the xy-lookup-table source operator |

### Frame grouping

| operator | purpose |
|---|---|
| [tcn_stream_synchronizer](tcn_stream_synchronizer/README.md) | Groups entities arriving on several ports at different rates into timestamp-consistent frame groups |

### Stream plumbing

| operator | purpose |
|---|---|
| [tcn_stream_splitter](tcn_stream_splitter/README.md) | Multi-tensor entity → one entity per named channel |
| [tcn_stream_merger](tcn_stream_merger/README.md) | Several entities → one, optionally concatenating their buffers |
| [tcn_flatten_tensor](tcn_flatten_tensor/README.md) | Reshapes `[H, W, ...]` to `[1, H*W, ...]` without copying |
| [tcn_convert_bgra_to_rgba](tcn_convert_bgra_to_rgba/README.md) | Channel swap for camera streams that report BGRA |

### Depth processing

| operator | purpose |
|---|---|
| [tcn_depthimage_backprojection](tcn_depthimage_backprojection/README.md) | Unprojects a depth image to world-space points and colour texcoords |
| [tcn_depthimage_weights](tcn_depthimage_weights/README.md) | Per-point confidence weights from surface angle and depth limits |
| [tcn_depthimage_apply_mask](tcn_depthimage_apply_mask/README.md) | Zeroes depth pixels outside (or inside) a binary mask |
| [tcn_depthimage_fgbg_mask](tcn_depthimage_fgbg_mask/README.md) | Foreground/background masks by comparison against a background depth image |
| [tcn_depthimage_max_distance](tcn_depthimage_max_distance/README.md) | Per-pixel running maximum, for building a background reference |
| [tcn_depthimage_temporal_filter](tcn_depthimage_temporal_filter/README.md) | Persistence/EMA temporal filter for depth noise |

### Models and inference sub-flows

Python sub-flows: whole pipelines packaged as `Subgraph`s, reusable across applications.

| package | purpose |
|---|---|
| [tcn_langsam](tcn_langsam/README.md) | Open-vocabulary segmentation (Grounding DINO + SAM 2) — a TensorRT multi-camera realtime path with baked prompts, and a PyTorch path with runtime prompting |
| [tcn_depth_anything](tcn_depth_anything/README.md) | Monocular metric depth, Depth-Anything V2 and V3 |

### Segmentation and point clouds

| operator | purpose |
|---|---|
| [tcn_panoptic_map](tcn_panoptic_map/README.md) | Paints per-detection masks into one packed panoptic map |
| [tcn_label_sampler](tcn_label_sampler/README.md) | Samples a panoptic map through texcoords onto the depth grid |
| [tcn_labeled_pointcloud](tcn_labeled_pointcloud/README.md) | Compacts a labeled depth grid into one point cloud per class |
| [tcn_texture_sampler](tcn_texture_sampler/README.md) | Samples a colour image through texcoords (bilinear) |
| [tcn_instance_stats](tcn_instance_stats/README.md) | Reduces a labeled point grid to one row per instance: count, centroid, box, spread |
| [tcn_object_tracking](tcn_object_tracking/README.md) | Cross-camera fusion, persistent object ids over time, console dump and box overlay |

### Transport

| operator | purpose |
|---|---|
| [tcn_zenoh_subscriber](tcn_zenoh_subscriber/README.md) | Zenoh topic → raw CDR payload |
| [tcn_zenoh_publisher](tcn_zenoh_publisher/README.md) | Raw CDR payload → Zenoh topic |
| [tcn_zenoh_receiver](tcn_zenoh_receiver/README.md) | Composite: Zenoh subscribe + CDR decode + GPU tensor output |
| [tcn_cdr_decoder](tcn_cdr_decoder/README.md) | Decodes CDR messages via the type registry |
| [tcn_cdr_serde](tcn_cdr_serde/README.md) | Library: CDR serialisation and the type registry (no operator) |
| [tcn_shm_serde](tcn_shm_serde/README.md) | Library: shared-memory types and the parameter RPC (no operator) |
| [tcn_shm_zenoh_sender](tcn_shm_zenoh_sender/README.md) | Publishes decoded frames into an iceoryx2 SHM segment |

### Python-side collections

| directory | purpose |
|---|---|
| [tcn_util](tcn_util/README.md) | Python operators: rotate, channel convert, flatten, split/merge, depth helpers; plus `frame_identity` (acquisition timestamps and tensor-map filtering) |
| [tcn_shm_io](tcn_shm_io/README.md) | Python SHM receiver, RPC and subscriber operator |
| [tcn_processing](tcn_processing/README.md) | Python reference implementations (e.g. simple backprojection) |
| [tcn_slang_renderer](tcn_slang_renderer/README.md) | slangpy-based renderers for point clouds and meshes |
| [tcn_ui](tcn_ui/README.md) | Parameter editor and application window |

## Shared conventions

### Tensor naming

Multi-camera entities key their tensors by channel name, which is how the splitter and the
synchroniser address them:

- `<camera>_colorimage` — e.g. `camera01_colorimage`
- `<camera>_depthimage`
- `<camera>_mask` — the packed panoptic map (`langsam_helpers.mask_name()` owns this mapping)

Single-tensor entities use the **unnamed** tensor (`""`). Most C++ operators expose
`in_tensor_name` / `out_tensor_name` and default to `""`; `tcn_stream_splitter` always emits unnamed
tensors, and `tcn_depthimage_apply_mask` only reads unnamed ones. When wiring operators, a
"missing tensor" error almost always means one side expected a name and the other emitted `""`.

### Packed panoptic labels

A panoptic map is `uint16` with `value = (class_id << 8) | instance_id`, and `0` means background.
Class ids are 1-based positions in the prompt list (`langsam_helpers.class_id_map`), so a class id
fits one byte and there are at most 256 of them.

The convention is encoded in exactly **two** places, and they must not drift:
`applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py` (which packs) and
`cuda/tcn_label_sampler_kernel.cuh` (which unpacks). `tcn_panoptic_map` receives already-packed
values and never unpacks, so it is not a third place.

### Acquisition timestamps

Frame identity travels as a `nvidia::gxf::Timestamp` component on the entity. Nothing in Holoscan
creates one, so:

- **`tcn_shm_subscriber` writes it**, from the publisher's clock, on both the colour and depth
  entities. `tcn_dataset_replayer` does the same from the export's capture times.
- **Every intermediate stage must forward it.** In C++, `op_output.emit(entity, port, acq_timestamp)`;
  in Python, `emit(..., acq_timestamp=...)`. A stage that forgets loses frame identity *silently*,
  and the only symptom is that downstream grouping never matches.
- **`tcn_stream_synchronizer` consumes it.** It requires timestamps to *advance*, since a timestamp
  is a frame's identity — see its README for what that means for looping replay.

Reading it back: `InputContext::get_acquisition_timestamp(port)` searches the entity for any
component of that type, so the component's *name* is not significant. Note that a received tensor
map therefore exposes an extra `"timestamp"` key whose value is not a tensor — Python consumers that
discover tensors by iterating `msg.keys()` must filter it (`langsam_common.tensor_names()`).

### Memory: choosing an allocator

- **`BlockMemoryPool`** — fixed-size blocks sized for a whole camera frame. Right for per-frame
  image tensors, wrong for many small tensors: each allocation consumes a full block, so a stage
  emitting dozens of small buffers per frame exhausts it ("Too many chunks allocated") while wasting
  most of what it hands out.
- **`RMMAllocator`** — sub-allocates from a pool. Right for variable-sized or numerous small outputs,
  e.g. the per-class point clouds of `tcn_labeled_pointcloud`.
- **`UnboundedAllocator`** — `cudaMalloc` per allocation. Fine for tests, not for a hot path.

### Zero-copy views must keep their source alive

Several operators produce a *view* of an upstream tensor with `nvidia::gxf::Tensor::wrapMemory`
(`tcn_stream_splitter`, `tcn_flatten_tensor`). `wrapMemory` does **not** take ownership. Passing a
null release callback leaves the output pointing at memory owned solely by the input entity, so once
`compute()` returns, that allocation can be reused by a later frame while a downstream consumer is
still reading it.

Both operators now hold a reference to the source entity in the release callback. If you add another
`wrapMemory` producer, do the same. The failure mode is worth recognising: nondeterministic, affects
one consumer on some frames, and is invisible while a single consumer reads the view immediately
after it is produced — it only appears once the tensor is buffered across ticks or read by more than
one consumer.

### Dynamic ports come from constructor arguments

`setup()` runs **before** parameter values are applied. An operator whose ports depend on
configuration (`tcn_stream_splitter`'s `channel_names`, `tcn_stream_merger`'s `input_port_names`,
`tcn_stream_synchronizer`'s `streams`, `tcn_labeled_pointcloud`'s `classes`) must read that list from
`args()` in `setup()`, and callers must pass it to the **constructor** rather than via
`from_config()`. Passing it as a config parameter yields an operator with no ports and an error that
names the port map rather than the cause.

### CUDA device selection

`cuda_device_ordinal` selects the device, but two things are easy to get wrong:

- The current device is **per thread**, and the scheduler runs `compute()` on any free worker.
  Retaining the primary context does not make the device current, so an operator that allocates or
  launches must scope itself to the configured device explicitly.
- A non-zero ordinal also needs a **`CudaStreamPool` on that device**. Holoscan otherwise supplies a
  device-0 stream, and launching on it reports `invalid device ordinal`.

### Holoviz geometry shapes

`HolovizOp` `POINTS_3D` consumes `[1, N, 3]` in this codebase. `tcn_flatten_tensor` maps
`[H, W, ...]` to `[1, H*W, ...]` — it *keeps* the leading 1 — which is how `tcn_shm_receiver` gets a
fused cloud into that shape. A producer already emitting `[1, N, 3]` needs no flatten step.

Colour for geometry is per `InputSpec`, not per vertex, so showing several classes in different
colours means one spec (and one upstream chain) per class.

## Building

Operators are gated behind a per-operator CMake option:

```cmake
add_holohub_operator(tcn_label_sampler)          # operators/tcn_artekmed/CMakeLists.txt
```

That option is only turned **ON** by an application listing the operator under
`DEPENDS OPERATORS`:

```cmake
add_holohub_application(tcn_shm_vlm_inference DEPENDS OPERATORS
        tcn_label_sampler
        ...)
```

Adding the operator to the application's `target_link_libraries` is **not** sufficient — that is a
link-time relationship resolved after configure, so the subdirectory is never added and the Python
module is silently not built. A missing module then surfaces as an `ImportError` at app start, or
worse, as a silent fallback to a slower path.

TensorRT engines are locked to both the TRT version and the GPU architecture; see the repo-root
`docs/trt11-upgrade-runbook.md` for the two-stage export procedure (ONNX on the host, engine built
inside the container).

## Testing

- **Host tests** run without holoscan/cupy and cover pure logic — frame-plan and timestamp
  arithmetic (`tcn_dataset_replayer/tests/`), ring buffer and matcher
  (`tcn_stream_synchronizer/tests/`). Run them directly with `python3` or `g++`.
- **Container tests** instantiate the real operator in a small Holoscan application
  (`tcn_label_sampler/tests/`, `tcn_labeled_pointcloud/tests/`). They need the built Python module:

  ```bash
  PYTHONPATH=<build>/python/lib python3 tests/test_label_sampler.py
  ```

Expectations in these tests are written against the operator's documented contract, computed
independently (hand-written values, or numpy) rather than by restating the implementation.
