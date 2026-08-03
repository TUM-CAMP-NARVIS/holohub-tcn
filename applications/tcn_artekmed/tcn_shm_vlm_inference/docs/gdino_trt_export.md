# Grounding DINO → ONNX → TensorRT export

Builds the TensorRT engine + baked text tensors used by the `gdino_backend: "trt"` path in
`langsam_inference` (see the runtime `GDinoTrtDetector`). Companion tool:
[`gdino_trt_export.py`](./gdino_trt_export.py).

## Why this exists

The PyTorch Grounding DINO forward is **launch-bound** (~230 ms, GIL-serialized — see the nsys
analysis). A TensorRT engine collapses those launches (~38 ms TF32 @512×672 in the spike, ~4×)
and runs from C++/one enqueue, so it also escapes the GIL. Detection quality is preserved
(spike parity IoU 0.9997).

## Environment (dedicated venv — NOT the Depth-Anything-3 venv)

The exporter needs the **IDEA-Research GroundingDINO** source and a **transformers 4.x** env
(5.x removes `BertModel.get_head_mask` and breaks GroundingDINO). Do not reuse the DA3 venv.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git   # or reuse an existing checkout
cd GroundingDINO
python3 -m venv .venv-gdino-export && . .venv-gdino-export/bin/activate
pip install torch torchvision                       # CUDA build matching your driver
pip install "transformers==4.44.2" "tokenizers<0.20" addict yapf timm \
            opencv-python onnx pycocotools tensorrt
# IMPORTANT: do NOT `pip install -e .` — that compiles GroundingDINO's CUDA op (_C) and fails
# on any CUDA-toolkit-vs-torch version mismatch (e.g. system nvcc 12.9 vs torch cu13.0). The op
# is NOT needed: export runs on CPU with the pure-PyTorch deformable-attention fallback, and the
# engine/parity run via TensorRT. Just make the package importable via PYTHONPATH:
export PYTHONPATH=$PWD
# checkpoint (Swin-T):
mkdir -p weights && curl -fsSL -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Notes / gotchas (found during the spike):
- `torch.onnx.export(..., dynamo=False)` — the legacy tracer handles GDINO's data-dependent
  shapes; the dynamo exporter fails.
- The traced ONNX is **fixed to its traced resolution** despite dynamic axes → the engine's
  `(H, W)` and token length `L` are baked. Re-run per resolution / prompt change.
- **TensorRT 11** dropped the `FP16`/`EXPLICIT_BATCH` builder flags (strongly-typed networks);
  the tool uses `TF32` (the validated default) and `create_network(0)`.

## Usage

Copy `gdino_trt_export.py` into the GroundingDINO checkout root and run:

```bash
export PYTHONPATH=$PWD            # so `import groundingdino` resolves (no pip install -e)
python gdino_trt_export.py \
  --checkpoint weights/groundingdino_swint_ogc.pth \
  --config groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --prompts floor person \
  --hw 512 672 \
  --out /data/models/active/groundingdino \
  --parity-image .asset/cats.png     # any existing image; the gate checks engine==pytorch agreement
```

Outputs (into `--out`):
- `gdino_swint_512x672_tf32.engine`
- `gdino_swint_prompts.npz` — `input_ids, attention_mask, position_ids, token_type_ids,
  text_token_mask, token_class_ids (256,), prompts`

The run ends with a **parity gate**: it compares the PyTorch (CPU) and engine (GPU) top-box
and **aborts if IoU < 0.99** (`--min-iou`). The parity image only needs to exist — it checks
engine-vs-PyTorch *agreement*, not detection quality, so any image works.

## Wiring into the app

Point the YAML at the artifacts and flip the backend (host `/data/models` → container
`/srv/models`):

```yaml
langsam_inference:
  gdino_backend: "trt"
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_tf32.engine"
  gdino_trt_text:   "/srv/models/active/groundingdino/gdino_swint_prompts.npz"
  gdino_trt_hw: [512, 672]     # must match the engine build size
```

**The prompts in `text_prompts.prompts` must match the engine's baked prompts** (order
included) — the runtime verifies this and fails fast otherwise. Changing prompts or the input
resolution requires re-running this tool.

## FP16 (optional, later)

`--fp16` is experimental: on TRT 11 (no global FP16 flag) it needs an fp16 ONNX / strongly-typed
build. TF32 is already quality-matched and ~4×; only pursue FP16 if you re-run the parity gate
and it stays ≥ 0.99.
