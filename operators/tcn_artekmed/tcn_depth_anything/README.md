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

## ONNX export

The V3 model is exported by `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/da3_export.py`.
The export must be **NHWC** to match `FormatConverterOp`'s output layout; a stale `.engine` built
from an NCHW export produces plausible-looking but wrong depth, so delete cached engines after
re-exporting.
