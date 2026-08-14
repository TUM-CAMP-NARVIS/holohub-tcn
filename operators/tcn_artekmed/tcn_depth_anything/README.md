# tcn_depth_anything

Monocular metric-depth sub-flows built on Depth-Anything V2 and V3.

| | `DA2MetricProcessingSubgraph` | `DA3MetricProcessingSubgraph` |
|---|---|---|
| module | `da2.py` | `da3.py` |
| postprocessor | `DA2PostprocessorOp` | `DA3PostprocessorOp` |

Each subgraph is self-contained: format conversion → TensorRT inference → postprocess to metric
depth, with the colour-camera intrinsics from the device context used to recover metric scale where
the model provides only relative depth.

The two share no code, deliberately. They differ in output layout and in how metric scale is
recovered, and an earlier attempt to collapse them into one parameterised subgraph made both harder
to follow than keeping them apart.

## Usage

```python
from operators.tcn_artekmed.tcn_depth_anything import DA3MetricProcessingSubgraph

da3 = DA3MetricProcessingSubgraph(self, "camera01_da3_pipeline", self.kwargs, device_context)
self.add_flow(source, da3, {("camera01_colorimage", "input")})
```

A missing device context is handled by disabling metric scaling rather than failing, so the subgraph
still runs on a source without calibration (e.g. a dataset export lacking it) — the depth is then
relative.

Imports are lazy: pulling in one variant does not construct the other's TensorRT stack.

## Models and engines

**Neither variant runs out of the box**: each needs an ONNX model on disk, which Holoscan's
`InferenceOp` then builds into a TensorRT engine on first use. There is no bundled weight file and no
download-on-demand.

Configured in the application's YAML:

```yaml
da2_inference_config:
  model_path: /srv/models/active/depth_anything_v2/depth_anything_v2_vits.onnx
da3_inference_config:
  model_path: /srv/models/active/depth_anything_v3/<hash>.onnx
  depth_near: 0.3        # metric display range
  depth_far: 10.0
da3_inference:
  backend: "trt"
  is_engine_path: false  # the path above is ONNX; InferenceOp caches an .engine beside it
```

Reference config and the export tooling live in the example application
[`applications/tcn_artekmed/tcn_shm_vlm_inference`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference):

| what | where |
|---|---|
| reference config | `python/tcn_shm_vlm_inference.yaml` → `da2_inference_config`, `da3_inference_config`, `da2_preprocessor`, `da3_inference` |
| DA3 export tool | `docs/da3_export.py` + [`docs/da3_onnx_export.md`](../../../applications/tcn_artekmed/tcn_all/docs/da3_onnx_export.md) |
| enabling the paths | `camera_stream_processing.enable_da2` / `enable_da3` (both **off** by default) |

### The layout requirement, and why a wrong export looks right

**The ONNX graph must accept NHWC `[1, H, W, 3]` in `[0, 1]`.** That is exactly what
`FormatConverterOp` produces upstream, and it is *not* what the upstream Depth-Anything code exports:
the stock model's `forward()` takes NCHW `(B, 3, H, W)` and expects ImageNet-normalised input.

`da3_export.py` wraps the model so the graph itself does the permute and the normalisation:

| stock | this export |
|---|---|
| `forward(image)` expects NCHW `(B,3,H,W)` | permutes internally — accepts NHWC `(B,H,W,3)` in `[0,1]` |
| caller normalises (ToTensor + Normalize) | normalisation is baked into the graph |

A stock NCHW export **will still load and still produce a depth map**. The numbers will be wrong in a
way that looks like a plausible depth image, which is the whole reason this is documented rather than
left to inference. Verify the export before trusting it:

```
Input image: [1, 518, 518, 3]      # NHWC -- matches the DA2 contract
```

`H` and `W` must both be divisible by **14** (the ViT patch size); `da3_export.py` enforces it.

### After re-exporting, delete the cached engine

`InferenceOp` caches a built engine next to the ONNX file (e.g. `<model>.onnx.engine.fp32`). It keys
that cache on the **path**, not the file's contents, so re-exporting the ONNX in place leaves the stale
engine in use — and the symptom is that your fix appears to have no effect at all.

```bash
rm -f /srv/models/active/depth_anything_v3/*.engine*   # then re-run; first run rebuilds
```

The same applies after a TensorRT upgrade: an engine built by another TRT version fails to
deserialise (`Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed`). Unlike the
LangSAM engines, these are cached artifacts rather than gated build products, so deleting them is
always safe — the cost is one slow first frame.

### Container requirements

Nothing beyond a Holoscan runtime with TensorRT: no fork, no separate venv, no patched SDK. Both
subgraphs work on the stock image. `da3_export.py` runs on the **host** (it needs the
Depth-Anything-3 checkout and its own venv — keep it separate from the GroundingDINO one), and only
the resulting `.onnx` has to reach the container.

If a metric-scaled result is needed, the subgraph takes the colour camera's intrinsics from
`DeviceContextService`; without a device context it disables metric scaling and produces relative
depth rather than failing, so a source lacking calibration still runs.

### Status in the example application

`enable_da2` and `enable_da3` are both **false** in the reference config — the LangSAM path is what
that application currently exercises, and the depth used by the mask/depth join comes from the
cameras, not from these models. The paths are kept working and expected to be available; if you enable
one, expect to export its ONNX first.
