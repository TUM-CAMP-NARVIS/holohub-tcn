# tcn_shm_vlm

Process Camera Images with Dino to extract features

## Overview

This application is built using Holoscan SDK version 3.7.0 and supports the following platforms:


## Prerequisites

- Holoscan SDK 3.7.0
- CUDA (if using GPU acceleration)
- Docker (for containerized deployment)

## Installation

1. Clone this repository

2. Install dependencies:

3. Build the application:

## Usage

### Running the Application

```bash
./holohub run tcn_shm_receiver
```

By default, the `./holohub build` and `./holohub run` commands will build and run the application in a containerized environment using the `standard` mode.

For local development without containers, use the `--local` flag:

```bash
./holohub run tcn_shm_receiver --local
```

Note that for the `--local` flag, the relevant custom dependencies (e.g. `requirements.txt` for Python) will be ignored and need to be installed manually.

### For containerized deployment

The application includes a Dockerfile for containerized deployment:

```bash
# Build the container
./holohub build-container tcn_shm_receiver

# Run the containerized application
./holohub run-container tcn_shm_receiver
```

For custom Docker builds:

```bash
# Build with custom base image
./holohub build-container tcn_shm_receiver --base-image nvcr.io/nvidia/clara-holoscan/holoscan:v3.7.0-dgpu

# Run with specific GPU type
./holohub run-container tcn_shm_receiver --gpu-type dgpu
```

## Models

### Depth-Anything-3 ONNX

The DA3 fragment needs an ONNX model exported with a **channels-last (NHWC)** input to match
the Holoscan `FormatConverterOp` / DepthAnything-V2 contract — the stock Depth-Anything-3
exporter produces an NCHW graph that TensorRT silently misreads. See
[`docs/da3_onnx_export.md`](docs/da3_onnx_export.md) for the how-to and
[`docs/da3_export.py`](docs/da3_export.py) for the ready-to-run patched exporter.

## Segmented point clouds (mask/depth join)

Gives every depth pixel the panoptic label of the scene point it observes, then emits one point cloud
per class and fuses them across cameras into a single view. Design:
[docs/specs/2026-08-12-mask-depth-join-design.md](docs/specs/2026-08-12-mask-depth-join-design.md).

Per camera the chain is

```
depth ─► backprojection ─┬─► texcoords ─┐
                         └─► positions ─┼─► label_sampler ─┬─► labels ─► labeled_pointcloud ─┐
masks (panoptic map) ────────────────────┘                 └─► mask ─► apply_mask            │
                                                                                             ▼
                                            one merger per class (fuses all cameras) ─► HolovizOp
```

Sampling the panoptic map through backprojection's texcoords is the only correct correspondence:
the depth and colour sensors differ in resolution, intrinsics, distortion and optical centre, so
rescaling a mask onto the depth grid is spatially wrong in a depth-dependent way that no aggregate
test detects.

### Enabling it

```yaml
temporal_sync:      { enabled: true }     # required -- see below
mask_depth_join:    { enabled: true }
camera_stream_processing: { enable_langsam_multicam: true }
```

`temporal_sync` is not optional. The mask path runs at roughly a third of the source rate and a
couple of frames behind it, so joining a mask to "whatever depth is current" mis-registers it by a
varying amount — which is the error this whole path exists to remove.

Classes come from `text_prompts.prompts` (ids are 1-based prompt positions) unless
`mask_depth_join.pointcloud_classes` overrides them, and each class takes its colour from the same
LUT the 2D mask overlay uses, so a class looks the same in both views.

`mask_depth_join.check` (default on) builds the verification operators. They read every output back
with GPU reductions — one sync per camera per frame — and assert that a pixel selected by the mask
kept its depth. That invariant caught a use-after-free in `tcn_stream_splitter`; keep it on while
gating and turn it off for a latency-sensitive run.

### Running on a live shm stream

Works on `source: "shm"` with no extra configuration: calibration comes from the device contexts
`discover_shm()` returns, the subscriber already emits `depth_outputs`, and the join discovers each
camera's depth channel from the segment rather than assuming its name. Preconditions:

- **Engines must match the active worker split.** A 5-camera rig needs engines built for its own
  batch sizes (see [docs/trt11-upgrade-runbook.md](../../../docs/trt11-upgrade-runbook.md)).
- **The publisher must set acquisition timestamps.** The subscriber forwards `frame.timestamp`; if it
  is never set, every frame carries the same value, the synchroniser rejects non-advancing timestamps
  and publishes nothing. The shutdown line says so directly — `non-monotonic` rising with
  `published 0 groups`.
- **`temporal_sync.depth_capacity` must exceed the mask/depth skew.** If it does not, the
  synchroniser reports `forced drops` rather than silently losing frames.
- **`depthimage_backprojection.depth_units_per_meter` is a yaml constant** (1000.0 = millimetres).
  The live device context carries the real value but `DeviceContextService` exposes no getter for it,
  so a camera using different units would scale every point linearly.

### Measured cost

Live 5-camera run, 529 frames (`/tmp/tcn/vlm_inference_profile.nsys-rep`, 2026-08-12): the period
went from 197.2 ms to **209.8 ms (4.77 fps)** — the whole geometric path costs **+12.6 ms/frame**,
about 6%. Per camera per frame:

| stage | avg |
|---|---|
| backprojection | 0.8–1.0 ms |
| label sampler | 1.1–1.4 ms |
| apply_mask | 0.6–1.7 ms |
| **labeled_pointcloud** | **24.4–24.8 ms** |
| fusion (5 mergers, once per frame) | ~25 ms total |
| point-cloud Holoviz | 68.8 ms/tick, asynchronous |

`tcn_labeled_pointcloud` is 87% of the per-camera cost, and 32% of its time is measured inside
`cudaStreamSynchronize`: it runs one `thrust::copy_if` per class and each one ends with a host-side
count, so a 5-class configuration synchronises five times per camera per frame. Computing all class
counts in one pass — one kernel, one device-to-host copy of K counts, then per-class scatter — would
cut that to one sync. Not done yet.

## Development

### Project Structure

```
tcn_shm_receiver/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   └── main.py
├── include/
├── tests/
└── docs/
```

### Adding New Operators

1. Create a new operator class in `src/operators/`
2. Include the operator in `src/main.py`
3. Update the pipeline configuration

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- Ulrich Eck - TU Munich

## Acknowledgments

- NVIDIA Holoscan Team
- Open source community contributors
