# Exporting Depth-Anything-3 to ONNX for the DA3 fragment

The `operators/tcn_artekmed/tcn_depth_anything/da3.py` pipeline runs Depth-Anything-3 through the Holoscan `InferenceOp`
(TensorRT backend). The ONNX model it consumes must follow the **same input contract as
the Depth-Anything-V2 model**, otherwise the depth output is garbage.

The upstream Depth-Anything-3 `export.py` does **not** produce a compatible model, so a
small patched exporter is provided here: [`da3_export.py`](./da3_export.py).

## Why a custom export

The DA3 subgraph feeds inference via `FormatConverterOp`, which emits a **channels-last
`[1, H, W, 3]` (NHWC)** image tensor scaled to `[0, 1]` — this is exactly what the
DA2 ONNX model was exported to consume. The stock DA3 exporter produces an **NCHW
`[1, 3, H, W]`** graph. Because the element counts match (`3·H·W == H·W·3`), TensorRT
copies the buffer without a shape error and silently reinterprets the interleaved RGB
pixels as three planes → **scrambled input → broken depth**.

`da3_export.py` fixes this by making the graph accept **NHWC `[0, 1]`** input and doing the
transpose-to-NCHW + ImageNet/DINOv2 normalization *inside* the graph (matching the DA2
contract). The key differences from the upstream script:

| Upstream `export.py` | `da3_export.py` |
| --- | --- |
| `forward(image)` expects NCHW `(B,3,H,W)` | `image = image.permute(0,3,1,2)` first — accepts NHWC `(B,H,W,3)` in `[0,1]`, then normalizes |
| `dummy_input = zeros(B, 3, H, W)` | `dummy_input = zeros(B, H, W, 3)` |
| `output_names=["depth", "sky"]` | `output_names=["depth"]` (monocular model has no `sky` output) |
| demo double-normalizes (ToTensor + Normalize) and feeds NCHW | demo feeds un-normalized NHWC `[0,1]` (normalization is baked into the graph) |

The DA3 model outputs **focal-normalized depth**, not metres (it is exported with
`intrinsics=None`). `DA3PostprocessorOp` converts it to metric depth at runtime using the
camera focal length: `metric[m] = raw · (focal / 300)` — see `da3_inference_config`
(`depth_near`/`depth_far`) in `tcn_shm_vlm_inference.yaml`.

## Prerequisites

1. **Clone Depth-Anything-3** and set up its environment (tested at commit `3fe327a`):

   ```bash
   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
   cd Depth-Anything-3
   # follow the repo's install instructions (a .venv with torch, onnx, onnxruntime)
   pip install -r requirements.txt onnx onnxruntime
   ```

2. **Get a checkpoint.** The metric model used here is
   [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE);
   `DepthAnything3.from_pretrained` will pull it from the Hugging Face Hub (or pass a local
   snapshot path). The ONNX filename in the config is that checkpoint's HF snapshot commit
   hash.

## Export steps

1. **Copy the patched exporter** into the Depth-Anything-3 checkout (next to the original
   `export.py`):

   ```bash
   cp <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/da3_export.py \
      Depth-Anything-3/da3_export.py
   ```

2. **Run the export** (518×518 to match the `da3_preprocessor` resize; both dims must be
   divisible by 14):

   ```bash
   cd Depth-Anything-3
   python da3_export.py \
     --model-dir ~/.cache/huggingface/hub/models--depth-anything--DA3METRIC-LARGE/snapshots/<commit> \
     --height 518 --width 518 \
     --output-dir /data/models/active/depth_anything_v3/ \
     --demo-image assets/examples/SOH/000.png
   ```

   The ONNX filename is derived from the `--model-dir` name, so it becomes
   `<commit>.onnx` (matching `da3_inference_config.model_path` in the YAML — note the host
   path `/data/models` maps to `/srv/models` inside the container). A `.onnx.data`
   sidecar (external weights) is written alongside it.

3. **Delete the stale TensorRT engine.** Holoscan's `InferenceOp` (`is_engine_path: false`)
   builds and caches a `*.engine.fp32` next to the ONNX. After re-exporting the ONNX you
   **must** delete the cached engine or it silently reuses the old (NCHW) graph:

   ```bash
   rm /data/models/active/depth_anything_v3/*.engine.fp32
   ```

   The engine is rebuilt from the new ONNX on the next app launch (adds a minute or two).

## Verify

The export prints the ONNX I/O shapes and a depth sanity check. A correct export shows:

```
Input image: [1, 518, 518, 3]      # NHWC, matches the DA2 contract
Output depth: [1, 1, 518, 518]
[DEMO] Depth stats: min=~2.2, max=~10.8, mean=~5.8   # positive, varied
```

If the input shows `[1, 3, 518, 518]` you are running the upstream exporter, not
`da3_export.py`.
