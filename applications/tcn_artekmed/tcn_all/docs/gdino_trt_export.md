# Grounding DINO → ONNX → TensorRT export

Builds the TensorRT engine + baked text tensors used by the `gdino_backend: "trt"` path in
`langsam_inference` (see the runtime `GDinoTrtDetector`). Companion tool:
[`gdino_trt_export.py`](gdino_trt_export.py).

## Why this exists

The PyTorch Grounding DINO forward is **launch-bound** (~230 ms, GIL-serialized — see the nsys
analysis). A TensorRT engine collapses those launches (~38 ms TF32 @512×672 in the spike, ~4×)
and runs from C++/one enqueue, so it also escapes the GIL. Detection quality is preserved
(spike parity IoU 0.9997).

## Two stages, two environments

A serialized TensorRT engine is **version-locked**: it only deserializes in the TRT that built
it. Building on the host produces an engine the runtime container refuses to load —

```
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 1: Serialization (Serialization assertion
      stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match.
      Note: Current Version: 239, Serialized Engine Version: 243)
```

— and the host cannot simply use the container's TRT, because the export half needs the
GroundingDINO fork + checkpoint that the container does not have. So the tool has two stages:

| Stage | Where | Needs | Produces |
|---|---|---|---|
| `--stage export` | host, wingdzero checkout | torch, groundingdino, opencv, checkpoint | `gdino_swint_prompts.npz`, `…_b<N>_tf32.onnx`, `…_parity_ref.npz` |
| `--stage build` | **inside the holohub runtime container** | tensorrt, torch | `…_b<N>_tf32.engine` (+ gates) |

The export stage deliberately builds **no** engine — a host-built engine is unusable, so building
one would only waste minutes. It instead saves a PyTorch parity reference (preprocessed image,
top score, top box) so the build stage can run the *same* faithfulness gate without torch weights
or the GroundingDINO checkout. The full model → ONNX → engine chain stays gated even though no
single environment can run all of it.

Re-run `--stage build` after any SDK/container bump that changes TensorRT. The ONNX and the npz
files are version-independent and do not need re-exporting.

## Environment (dedicated venv — NOT the Depth-Anything-3 venv)

**Use the wingdzero fork, NOT stock IDEA-Research GroundingDINO.** This tool feeds the model
six pre-tokenized tensors `(img, input_ids, attention_mask, position_ids, token_type_ids,
text_token_mask)`. The stock `GroundingDINO.forward(self, samples, targets=None)` tokenizes the
caption internally and rejects that interface (`forward() takes from 2 to 3 positional arguments
but 7 were given`). The **wingdzero fork** modifies `groundingdino/models/GroundingDINO/
groundingdino.py` to accept the flat tensors — that modified forward is what the export/engine
consume, so the exporter must import *that* source. Also needs a **transformers 4.x** env (5.x
removes `BertModel.get_head_mask` and breaks GroundingDINO). Do not reuse the DA3 venv.

```bash
git clone https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference.git
cd GroundingDINO-TensorRT-and-ONNX-Inference
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

Notes / gotchas (found during the spike + first real export):
- **Checkpointing off.** `load_model` sets both `use_checkpoint=False` and
  `use_transformer_ckpt=False`. The config defaults both to `True`, which wraps encoder layers in
  `torch.utils.checkpoint.checkpoint` — untraceable (`RuntimeError: unordered_map::at`).
- **`dynamic_axes` is required.** Without it the legacy tracer constant-folds the image branch and
  drops `img` as a graph input; the engine output becomes image-independent (bakes the dummy
  image). The tool passes the full `dynamic_axes` dict; engine profiles still pin fixed shapes.
- **Clear the feature cache before tracing.** `GroundingDINO.forward` caches backbone features in
  `self.features`/`self.poss` and only recomputes them if absent, so any prior forward (e.g. the
  parity reference) makes the trace bake stale features and ignore `img`. The tool calls
  `model.unset_image_tensor()` before export. Symptom if this regresses: engine gives identical
  output for different images (test `np.allclose(engine(imgA), engine(imgB))` — must be False).
- **Capture the parity reference before export.** The same cache means the *first* PyTorch forward
  after `torch.onnx.export` returns wrong values; the tool computes the reference before exporting.
- `torch.onnx.export(..., dynamo=False)` — the legacy tracer handles GDINO's data-dependent
  shapes; the dynamo exporter fails.
- The traced ONNX is **fixed to its traced resolution / token length** → the engine's `(H, W)` and
  `L` are baked. Re-run per resolution / prompt change.
- **TensorRT 11** dropped the `FP16`/`EXPLICIT_BATCH` builder flags (strongly-typed networks);
  the tool uses `TF32` (the validated default) and `create_network(0)`.
- **The parity image must contain the prompted classes** (`--min-detect`, default 0.30). On an
  image lacking them every query is low-confidence noise and TF32 rounding flips the argmax box,
  so top-1 IoU is meaningless; the tool aborts with `PARITY IMAGE UNUSABLE` in that case.

## Usage

### Stage 1 — export (host)

**Do NOT copy the tool into the checkout.** Run it **in place** from this repo, with the
GroundingDINO checkout as the working directory. Copying it breaks `--from-config`: the tool
locates `tcn_langsam.helpers` relative to its own file (`<tool dir>/../python`), so a copy sitting in
the checkout root looks for `~/develop/vision/python` and fails with
`ModuleNotFoundError: No module named 'tcn_langsam.helpers'`.

```bash
cd ~/develop/vision/GroundingDINO-TensorRT-and-ONNX-Inference   # relative --config/--checkpoint/
                                                               # --parity-image resolve from here
PYTHONPATH=$PWD \                       # so `import groundingdino` resolves (no pip install -e)
~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python \
  <repo>/applications/tcn_artekmed/tcn_all/docs/gdino_trt_export.py --stage export \
  --checkpoint weights/groundingdino_swint_ogc.pth \
  --config groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --prompts floor person \
  --hw 512 672 \
  --batch 3 \
  --out /data/models/active/groundingdino \
  --parity-image images/in/person.jpg  # MUST contain the prompted classes (floor/person here)
```

Why each piece is needed:

| piece | reason |
|---|---|
| `cd` into the checkout | `--config`, `--checkpoint` and `--parity-image` above are relative paths |
| `PYTHONPATH=$PWD` | resolves `import groundingdino` to the wingdzero fork |
| the `.venv-gdino-export` python | has torch, `transformers==4.44.2` and tensorrt |
| the tool's **absolute repo path** | lets its own `sys.path` insert find `tcn_langsam.helpers` for `--from-config` |
| `--out` is a **host** path | this is the host stage; `/srv/models` only exists in the container |

`--batch N` is the batch this artifact set is for — set it to the busiest LangSAM worker's
camera count. **The engine's batch is baked at ONNX trace time**, so a camera-count change is a
re-export, not just a rebuild (see "Why a single fixed `--batch`" below). Both stages must pass
the SAME `N`.

The fork only ships `images/in/car_1.jpg`; supply your own image that clearly shows the prompted
classes (a person on a floor for `floor person`) and pass it as `--parity-image`.

Outputs (into `--out`):
- `gdino_swint_512x672_b3_tf32.onnx`
- `gdino_swint_prompts.npz` — `input_ids, attention_mask, position_ids, token_type_ids,
  text_token_mask, token_class_ids (256,), prompts` — batch-independent, shared by every `--batch`
- `gdino_swint_512x672_parity_ref.npz` — `image` (preprocessed `(1,3,H,W)`), `score`,
  `box_cxcywh`, `image_path`, `hw` — always the batch-1 PyTorch reference, also shared across
  every `--batch`

This stage aborts with `PARITY IMAGE UNUSABLE` if the PyTorch top score is below `--min-detect`
(0.30), i.e. the image does not contain the prompted classes — see the gotchas above for why
that makes the IoU meaningless. It fails *before* the ONNX export so you do not wait for it.

### Why a single fixed `--batch`

A previous revision made `--batch` a MIN/OPT/MAX optimization profile so one engine could serve
any camera count up to MAX. Hardware testing proved that does not work: a `Where` node in the
encoder fusion attention is only broadcast-conformable at the batch the ONNX was traced at.
Building the profile at any other batch fails outright ("broadcast dimensions must be
conformable"), and a multi-size profile silently specialises to a static shape instead of
actually staying dynamic. So `--batch` is one integer, used identically by both stages: export
traces at it, build pins `min = opt = max` to it. Changing camera count means re-running
**both** stages at the new `N` — there is no way to widen an already-built engine.

### `--from-config`

Instead of `--batch N`, pass `--from-config /path/to/tcn_all.yaml` to build one
engine per **distinct worker camera count** in the app's `gpu_workers` node (mutually exclusive
with `--batch`). Both stages need the **same** `--from-config`:

```bash
# host
python gdino_trt_export.py --stage export --prompts floor person \
  --hw 512 672 --out /data/models/active/groundingdino \
  --parity-image images/in/person.jpg \
  --from-config /path/to/tcn_all.yaml

# container
python3 /workspace/holohub/applications/tcn_artekmed/tcn_all/docs/gdino_trt_export.py \
  --stage build --hw 512 672 --out /srv/models/active/groundingdino \
  --from-config /path/to/tcn_all.yaml
```

With a `gpu_workers.workers` list of `[{cameras: [a, b]}, {cameras: [c, d, e]}]` this builds
`gdino_swint_512x672_b2_tf32.*` and `gdino_swint_512x672_b3_tf32.*` — reading the *same*
`gpu_workers` node the application reads at startup is the point: the engines that exist cannot
drift from the GPU split that actually runs. The prompts npz and the parity-reference npz are
batch-independent and are written **once** per export run, not once per batch. `--batch N` still
works unchanged for building a single engine by hand.

### Stage 2 — build (inside the runtime container)

The repo is mounted at `/workspace/holohub` and `/data/models` at `/srv/models`, so the tool and
the stage-1 artifacts are already visible:

```bash
# Inside the already-running container:
./holohub run --local --cuda 13 tcn_all
python3 /workspace/holohub/applications/tcn_artekmed/tcn_all/docs/gdino_trt_export.py \
  --stage build --hw 512 672 --out /srv/models/active/groundingdino \
  --batch 3
```

`--batch N` must match the value passed to `--stage export` — the ONNX was traced at that batch
and the build stage pins the optimization profile's `min = opt = max` to the same `N`. Artifacts
carry `_b<N>_` in the filename precisely so a mismatched export/build pair cannot be combined by
accident.

Output: `gdino_swint_512x672_b3_tf32.engine` (overwrites any earlier engine at that `_b<N>_`
path — the filename carries no TRT version so the YAML never changes across TRT bumps).

The build then runs two checks:
- **slice-consistency gate (blocking).** Replays the parity image at batch `N` and requires
  every slice to agree with slice 0 (top-box IoU >= 0.999, score delta <= 0.01). This is what
  actually proves batching is correct: identical inputs across the batch must give identical
  output, and the failure mode of a broken batch would be silently wrong output on slices
  1..N-1 — indistinguishable from random per-camera detection dropouts. Skipped (prints and
  returns) when `N == 1`.
- **PyTorch fidelity (reported only).** Compares slice 0 against the stage-1 PyTorch reference.
  Non-blocking by default: every container TRT 10.9 engine currently deviates from PyTorch for
  reasons unrelated to batching (boxes stay close, IoU ~0.96, but confidence scores are
  depressed) — a batch-1 engine deviates identically, so blocking on it by default would fail
  every build for a pre-existing, separately-tracked issue. It is printed loudly either way; pass
  `--strict-parity` to make a deviation fatal.

## Customizing: prompts, resolution, model

**Golden rule: the four artifacts are one matched set.** The ONNX bakes the resolution `(H, W)`,
the traced batch `N`, and the token length `L`; the npz bakes the prompt→class mapping. Never
regenerate one without the others: always re-run `--stage export` *and* `--stage build`, in that
order, with the SAME `--batch N`, then update the YAML.

### Changing the prompts

`--prompts` is order-sensitive: prompt *i* (0-based, left to right) becomes **class id i+1**
(0 is background), and that id drives the panoptic label map and the mask colors.

1. **Pick a parity image that contains the new classes.** The gate compares a top-1 box; on an
   image without a prompted class every query is low-confidence noise and TF32 rounding flips
   which one wins, so the IoU is meaningless. `--stage export` refuses to continue when the
   PyTorch top score is below `--min-detect` (0.30).
2. Re-run **stage 1** with the new `--prompts` (this changes `L`, so the ONNX must be rebuilt —
   a prompt set of a different length is not a drop-in npz swap):
   ```bash
   python gdino_trt_export.py --stage export --prompts floor person robot \
     --hw 512 672 --out /data/models/active/groundingdino \
     --parity-image images/in/person.jpg
   ```
   Check the printed `caption=` line. `token_class_ids nonzero=` counts BERT *tokens* covered by
   the prompts, not prompts — a multi-word prompt like `operating table` spans several tokens, so
   this is ≥ the prompt count (`floor person` → 2). The line that actually signals trouble is
   `WARNING: N caption categories vs M prompts`: it means the caption did not split into one
   category per prompt (usually a prompt containing a `.`). Do not ship a build that prints it.
3. Re-run **stage 2** in the container (same command as above — the filenames do not change).
4. Update `text_prompts.prompts` in `tcn_all.yaml`. It does not need to be an exact
   copy of what you just baked: at runtime `GDinoTrtDetector.set_prompts` (`build_prompt_remap`
   in `../../../../operators/tcn_artekmed/tcn_langsam/helpers.py`) accepts **any subset and/or reordering** of the engine's baked
   prompts and renumbers the class ids to match, with no rebuild. Only a term that was never
   baked into the engine at all raises, e.g.:
   `prompts ['robot'] are not baked into the GDINO TRT engine (baked: ['floor', 'person'])`.
   That case needs the re-export + rebuild shown above (with the new term included).

> **Filename caveat:** `gdino_swint_prompts.npz` encodes neither the prompts nor the resolution,
> and the parity ref encodes only `HxW` (it's always the batch-1 PyTorch reference — shared
> across every `--batch`). The ONNX/engine encode `HxW` and `_b<N>_` but not the prompts. Two
> different prompt sets in the same `--out` therefore overwrite each other regardless of
> resolution or batch. Use one `--out` directory per prompt set if you want to keep several
> around, and point `gdino_trt_engine`/`gdino_trt_text` at that directory.

### Changing the input resolution

`--hw H W` is the resolution the engine is built for; `detect()` resizes each camera frame
straight to it with no letterboxing.

- **Match the camera aspect ratio.** The TCN cameras are 2048×1536 (4:3), which is why the
  default is 512×672 (672/512 = 1.3125 ≈ 4:3). A mismatched ratio distorts the frame and costs
  detection quality.
- Larger = better on small objects, slower. GDINO cost is dominated by token count (deformable
  attention + fusion), so it scales roughly with H×W, not with the backbone size.
- The filenames carry the resolution and batch (`gdino_swint_512x672_b3_tf32.*`), so several
  resolutions (and batch sizes) can coexist in one `--out`.

After re-running both stages, update **two** YAML keys:

```yaml
  gdino_trt_engine: ".../gdino_swint_<H>x<W>_b<N>_tf32.engine"   # filename carries the new size
  gdino_trt_hw: [<H>, <W>]                                       # must match the engine build size
```

`gdino_input_size` is **not** used by the TRT path — it only resizes for the PyTorch backend.
Leave it alone unless you also test `gdino_backend: "pytorch"`.

### Changing the model variant or precision

- **Swin-B instead of Swin-T:** pass the matching `--checkpoint` and `--config`. Note the artifact
  tag is hard-coded `gdino_swint_` in `main()`, so a Swin-B build silently reuses Swin-T
  filenames — either edit `tag` or (simpler) export into a separate `--out` directory.
- **`--fp16`:** experimental, and it changes the tag to `gdino_swint_<H>x<W>_b<N>_fp16.*`, so
  `gdino_trt_engine` must be repointed. On TRT 11 there is no global FP16 flag (strongly-typed
  networks), so this needs an fp16 ONNX; TF32 is the validated default. Only adopt it if the
  fidelity report still reports IoU ≥ 0.99.

### Re-running after a container/SDK bump

TensorRT changed → **`--stage build` only**. The ONNX, prompts npz and parity ref are
version-independent; there is no need to touch the host or the YAML.

## Wiring into the app

Point the YAML at the artifacts and flip the backend (host `/data/models` → container
`/srv/models`):

```yaml
langsam_inference:
  gdino_backend: "trt"
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_b3_tf32.engine"
  gdino_trt_text:   "/srv/models/active/groundingdino/gdino_swint_prompts.npz"
  gdino_trt_hw: [512, 672]     # must match the engine build size
```

**`text_prompts.prompts` may be any subset and/or reordering of the engine's baked prompts.**
`GDinoTrtDetector.set_prompts` (`build_prompt_remap` in `../../../../operators/tcn_artekmed/tcn_langsam/helpers.py`) renumbers the
class ids to match at runtime — no rebuild needed. Only a prompt term the engine never baked
raises (`ValueError: prompts [...] are not baked into the GDINO TRT engine ...`), and only that
case requires re-running this tool (both stages), with the new term included. Changing the input
resolution, or the camera count driving `--batch`, always requires re-running this tool (both
stages, same `N`), since `(H, W)` and the batch are baked into the ONNX.

## FP16 (optional, later)

`--fp16` is experimental: on TRT 11 (no global FP16 flag) it needs an fp16 ONNX / strongly-typed
build. TF32 is already quality-matched and ~4×; only pursue FP16 if you re-run the fidelity
report and it stays ≥ 0.99.

## Build-stage gates

The build stage runs three checks, in this order:

1. **Image independence** (blocking). Mirrors the parity image and requires the logits to change.
   `GroundingDINO.forward` caches backbone features, so a trace taken after any prior forward can
   bake them in as constants and drop `img` from the graph — producing an engine that returns the
   same detections for every frame. This matters especially with `--from-config`, where one
   process traces the same model once per batch. It is deliberately independent of the PyTorch
   comparison below, so the container's TRT-vs-PyTorch deviation can neither mask it nor trip it.
2. **Slice consistency** (blocking). Every slice of a batched run must agree with slice 0.
3. **PyTorch fidelity** (reported, non-blocking). See below.
