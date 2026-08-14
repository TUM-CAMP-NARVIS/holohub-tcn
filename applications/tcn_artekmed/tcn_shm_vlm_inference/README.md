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
[`../tcn_all/docs/da3_onnx_export.md`](../tcn_all/docs/da3_onnx_export.md) for the how-to and
[`../tcn_all/docs/da3_export.py`](../tcn_all/docs/da3_export.py) for the ready-to-run patched exporter.

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

`tcn_labeled_pointcloud` is 87% of the per-camera cost, and 32% of its time was measured inside
`cudaStreamSynchronize`.

That has since been reduced from K synchronisations per camera per frame to exactly one:
`cub::DeviceSelect::Flagged` writes its result count to *device* memory, so every class can be
selected and compacted with the stream still running, and a single copy brings all K counts back.
(The previous `thrust::copy_if` returns a host-side iterator, and reading it forces a synchronise per
class.) CUB's variant is also stable, so the point order — and therefore the output — is unchanged.

Measured in isolation on an idle GPU at the live grid size (576×640, 5 classes, 16.5% labeled,
3 × 1000 ticks, `tests/bench_labeled_pointcloud.py`): median **0.703 → 0.600 ms/tick (−15%)**, mean
−19%, p90 −22%, with non-overlapping run-to-run ranges.

Re-traced live (5 cameras, 522 frames, 2026-08-12): the operator's median went **12.29 → 5.24 ms**
(trimmed mean 24.70 → 9.75, p90 36.14 → 11.89), consistently across all five cameras, and its
`Synchronize` calls per tick went **10.00 → 1.00** — `thrust::copy_if` was costing two blocking calls
per class, not one.

**It bought no throughput.** The period went 209.8 → 211.8 ms (4.77 → 4.72 fps) and dev1 utilisation
75.1% → 74.9% — flat, marginally worse, inside run-to-run variation. The freed time reappeared
elsewhere in the same run (`shm_subscriber` median 14.94 → 21.37 ms, the mask collector 15.01 →
20.12 ms) while gdino/sam each got slightly faster.

The reason is structural: these five operators run concurrently on 24 worker threads and were never
on the critical path. Their 25 ms was spent blocked on the GPU queue *in parallel with* the real
bottleneck, GDINO/SAM on dev1. Freeing it releases worker threads that then contend with everything
else, so latency redistributes instead of disappearing — and the one remaining sync still absorbs 61%
of the operator's now-much-shorter wall time, i.e. it is still waiting on queued work rather than its
own.

Worth keeping anyway, for what it does buy: 45 fewer blocking sync points per frame, ~75 ms/frame of
worker-thread occupancy freed, and a 3× tighter tail on this operator — headroom for more cameras or
classes. Not fps. **A duration measured inside a blocking call is queue depth, not the callee's
cost**, and an operator off the critical path can be made nearly free without moving the period.

## Development

**[docs/development-loop.md](../tcn_all/docs/development-loop.md)** — the edit → test-in-container → verify loop
used to build this application: how to run checks inside the already-running container, how to build a
gate whose output distinguishes *passed* from *did not run*, why a deterministic replay source is a
prerequisite for any correctness claim, and how to tier gates so the fast ones run on every edit. Worth
reading before changing anything here; most of the subtle defects in this pipeline were found by a gate
rather than by reading code.


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
