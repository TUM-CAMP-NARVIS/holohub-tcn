# tcn_langsam

Open-vocabulary segmentation (Grounding DINO + SAM 2) as reusable sub-flows, in two variants that
share their model wrappers, prompt/class mapping and panoptic packing.

## The variants

| | `RealtimeLangSamSubgraph` | `PromptedLangSamSubgraph` | `SingleCameraLangSamSubgraph` |
|---|---|---|---|
| module | `realtime.py` | `prompted.py` | `single_camera.py` |
| cameras | many, split across GPU workers | **many, split across GPU workers** | one |
| Grounding DINO | TensorRT, prompts baked in | **PyTorch** (or TRT, restricted) | PyTorch |
| SAM encoder | TensorRT, batched | **TensorRT, batched** | PyTorch |
| batched SAM decode | yes | **yes** | no |
| fused CUDA panoptic | yes | **yes** | no |
| vocabulary | fixed at export time | **any term, any time** | fixed at construction |
| throughput | ~4.7 fps, 5 cameras (measured, A40 pair) | lower, in the detector only | lowest |

**The promptable path is not a reduced version of the realtime one.** It is the same pipeline —
same worker split, same batching, same SAM and panoptic optimisations — with only Grounding DINO's
text branch made dynamic. That is possible because nothing else depends on the prompt set:

- the **SAM image encoder**'s only input is the image; prompts never enter it
- the **SAM mask decoder** is prompted by *boxes*, not text, and box counts already vary per frame
  (which is what the batched decode pads for)
- the **fused CUDA panoptic paint** writes the packed `uint16` values it is handed; the class mapping
  is computed host-side from the current prompt list
- **batching** is over images, not prompts

So making the vocabulary dynamic costs a slower *detector*, not a slower pipeline. SAM does not have
to become dynamic, which is the question worth settling before assuming otherwise.

### One vocabulary, all cameras

`PromptedLangSamSubgraph` takes a single prompt set and applies it to every camera. That is what keeps
the batched forward batched: `GDINO.predict_gpu_batch` encodes the caption **once** and repeats it
across the image batch, so per-camera vocabularies would mean one text encode per camera and would
undo the batching. The subgraph's `prompts` interface port fans out to every worker and to the
colouriser.

### Three tiers of prompt dynamism

| tier | configuration | what can change at runtime |
|---|---|---|
| fixed | `RealtimeLangSamSubgraph` | nothing; re-export to change the vocabulary |
| restricted | `PromptedLangSamSubgraph` + `prompted_gdino_backend: trt` | any **subset or reordering** of the baked vocabulary, at full engine speed (`build_prompt_remap` renumbers token class ids — no tokenizer, no re-export) |
| free | `PromptedLangSamSubgraph` (default `pytorch`) | **any term**, including ones never baked |

The middle tier is genuinely useful: switching from "person, bed, device, hololens, pipes" to just
"person" costs nothing and keeps the TRT detector.

Note `prompted_gdino_backend` is a **separate key** from `gdino_backend`. Inheriting the latter would
let a config written for the realtime path silently restrict the prompted sub-flow to the engine's
baked vocabulary — you would select the promptable variant and only discover the restriction when an
update was rejected.

## Layout

```
tcn_langsam/
  helpers.py       prompt/class mapping, panoptic packing, worker+engine planning, postprocess,
                   batch planning, validate_source_cameras   (pure; no torch, no holoscan)
  models.py        SAM, GDINO, SamTrtEncoder, GDinoTrtDetector, panoptic map/LUT builders
  realtime_ops.py  GdinoOp, SamOp, PanopticOp -- the split, pipelinable per-stage operators
  realtime.py      LangSamBatchOp (monolithic per-worker op, promptable), MaskCollectorOp,
                   LabelMapColorizeOp, RealtimeLangSamSubgraph
  prompted.py      PromptedLangSamSubgraph -- same operators, dynamic vocabulary
  single_camera.py SingleCameraLangSamSubgraph -- the original one-camera reference path
  prompted_ops.py  LangSAM2Operator, TextPromptPublisher, LangSamPostprocessorOp
  _viz.py          private drawing/debug helpers used by the single-camera path
  mask_dump.py     MaskDumpOp -- writes per-frame .npy panoptic maps for byte-comparison gates
  tests/           host tests (no holoscan/cupy required)
```

Both multi-camera subgraphs build the **same** `LangSamBatchOp`; `promptable=True` adds its
conditionless `prompts` port. Nothing is duplicated between them.

`helpers.py` deliberately imports nothing heavy: the offline engine builders
(`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/{gdino,sam}_trt_export.py`) import it to
derive the same worker/batch split the application runs, so the engines that exist always match the
split that uses them.

Everything is imported **lazily** through `__init__.py`. Touching this package must not construct a
TensorRT stack — the prompted path does not need one, and the engine builders import the helpers
without any model at all.

## Usage

```python
from operators.tcn_artekmed.tcn_langsam import RealtimeLangSamSubgraph, PromptedLangSamSubgraph

# many cameras, fixed vocabulary, TRT detector
langsam = RealtimeLangSamSubgraph(self, "langsam_multicam", self.kwargs, all_color_cams)
self.add_flow(source, langsam, {("color_outputs", "input")})
self.add_flow(langsam, viz, {("output_viz", "receivers")})
# also: output_masks (packed panoptic maps), output_specs

# many cameras, vocabulary editable while running -- identical outputs
langsam = PromptedLangSamSubgraph(self, "langsam_multicam", self.kwargs, all_color_cams)
self.add_flow(source, langsam, {("color_outputs", "input")})
self.add_flow(prompt_source, langsam, {("out", "prompts")})   # {"text_prompts": [...]}
```

Any operator emitting `{"text_prompts": [str, ...]}` works as the prompt source — `TextPromptPublisher`
reads them from configuration, but a UI, an RPC or a file watcher substitutes directly. Publish **on
change**, not per tick: prompt-derived state is rebuilt when the set differs, and the port is
conditionless so frames keep flowing when no update is pending.

In `tcn_shm_vlm_inference`, pick the variant with
`camera_stream_processing.langsam_variant: realtime | prompted`.

Two behaviours worth knowing:

- **A prompt change is not atomic across workers.** Each worker applies the update on its own next
  tick, so for one frame two workers can label with different class ids and the collector will merge
  them — one frame of mixed colours. Treat a prompt change as a scene change, not a per-frame control.
- **The colour LUT only grows.** A class that disappears and comes back keeps its colour, which
  matters more than reclaiming a few unused LUT rows.

Per-stage operators are exported too, for an application that wants to wire them itself:

```python
from operators.tcn_artekmed.tcn_langsam import GdinoOp, SamOp, PanopticOp
```

## Outputs

`RealtimeLangSamSubgraph` emits:

- `output_masks` — one packed panoptic map per camera, keyed `<camera>_mask`, `uint16`
  `(class_id << 8) | instance_id`, 0 = background
- `output_viz` — RGBA colourisation of those maps for display
- `output_specs` — matching `HolovizOp` input specs

The packed encoding is shared with `tcn_label_sampler`; see the collection README for where the two
halves of that convention live.

## Models, engines and the container

**This package cannot run out of the box.** Both variants need model weights, and every configuration
except pure PyTorch needs TensorRT engines that must be built **for your TensorRT version, your GPU
architecture and your camera split**. There is no fallback that silently downgrades: a missing or
mismatched engine is a hard failure at `start()`, by design.

Everything below is driven by `langsam_inference` in the application's YAML. The reference
configuration and the export tooling live in the example application
[`applications/tcn_artekmed/tcn_shm_vlm_inference`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference):

| what | where |
|---|---|
| reference config | `python/tcn_shm_vlm_inference.yaml` → `langsam_inference`, `gpu_workers`, `text_prompts` |
| GDINO export tool | `docs/gdino_trt_export.py` + [`docs/gdino_trt_export.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md) |
| SAM export tool | `docs/sam_trt_export.py` + [`docs/sam_trt_export.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.md) |
| container rebuild | [`docs/trt11-upgrade-runbook.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/trt11-upgrade-runbook.md), `docs/trt11_build_test.sh` |
| mask correctness gate | `docs/compare_mask_dumps.py` (fed by `MaskDumpOp`) |

### What each backend needs

| configuration | needs | fails how |
|---|---|---|
| `gdino_backend: pytorch` | HF weights (`gdino_model_id`, downloaded on first use) | slow, but runs anywhere |
| `gdino_backend: trt` | `gdino_trt_engine` **and** `gdino_trt_text` (the baked prompt tensors), matching `gdino_trt_hw` | `RuntimeError` at `start()`: engine or npz missing |
| `sam_backend: pytorch` | SAM 2 checkpoint (`sam_type`, `sam_ckpt_path`) | runs anywhere |
| `sam_backend: trt` | `sam_trt_engine` for **each distinct worker batch** | `RuntimeError` at `start()` |
| `panoptic_backend: cuda` | the `tcn_panoptic_map` pybind module **built** | falls back to cupy with a warning |

`panoptic_backend` is the one soft failure, and it is a trap: the module is only built if an
application lists `tcn_panoptic_map` under `DEPENDS OPERATORS` (see the collection README). Without it
you get the slow path and a single warning line, which is easy to miss in a long log.

### Engines are triply locked

An engine file is only valid for the combination of:

1. **TensorRT version.** A TRT 10.9 engine will not load in TRT 11.2 — the failure is
   `Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed`. Engines must therefore
   be built **inside the container that will run them**, which is why the GDINO tool is split into
   two stages.
2. **GPU architecture.** An engine built for `sm_86` (A40/A6000) does not run on `sm_120` (Blackwell).
   Moving to different hardware means rebuilding every engine.
3. **Batch size = a worker's camera count.** Engines are fixed-batch. `gpu_workers.workers` defines
   the split, and `worker_engine_path()` substitutes `{batch}` into the configured template:

   ```yaml
   sam_trt_engine: /srv/models/active/sam2/sam2.1_hiera_tiny_encoder_b{batch}_fp16.engine
   gdino_trt_engine: /srv/models/active/groundingdino/gdino_swint_512x672_b{batch}_tf32.engine
   ```

   A 2/3 camera split needs engines for batch **2 and 3**; changing to 4/1 needs batch **4 and 1**.
   A template without `{batch}` formats to itself, so a single-engine setup keeps working — but then
   every worker must have the same camera count.

Both export tools accept `--from-config <the app yaml>` and derive the distinct batch sizes from
`gpu_workers` themselves, which is the only way to keep engines and the running split in step. Use it
in preference to passing `--batch` by hand.

### GDINO: two stages, two environments

The export cannot happen in the container (no checkpoint, no GroundingDINO source) and the engine
build cannot happen on the host (the engine must match the container's TRT). Hence:

| stage | where | produces |
|---|---|---|
| `--stage export` | host, inside a **wingdzero GroundingDINO fork** checkout | `gdino_swint_prompts.npz`, `…_b<N>.onnx`, `…_parity_ref.npz` |
| `--stage build` | **inside the runtime container** | `…_b<N>_tf32.engine` + gates |

```bash
# host, in the fork checkout
python3 gdino_trt_export.py --stage export --prompts person bed device hololens pipes \
        --hw 512 672 --from-config <app>/python/tcn_shm_vlm_inference.yaml
# container
python3 gdino_trt_export.py --stage build --hw 512 672 --out /srv/models/active/groundingdino
```

Three prerequisites that are not optional, each of which cost real time to discover:

- **The wingdzero fork, not stock IDEA-Research GroundingDINO.** The tool needs a 6-tensor forward
  signature; stock raises `takes 5 positional arguments but 7 were given`.
- **Do NOT `pip install -e .`** in that checkout. It compiles GroundingDINO's `_C` CUDA op, which
  fails on any nvcc/torch mismatch, and the op is *not needed* — export runs on CPU with the
  pure-PyTorch deformable-attention fallback. Make the package importable with `PYTHONPATH` instead.
- **Run from the checkout as working directory**, and use a dedicated venv (not the Depth-Anything
  one).

**The prompts are baked into the engine at this point.** `--prompts` fixes the tokens in the engine's
`input_ids`; the class mapping lives in `token_class_ids` inside the npz. Consequences:

- `RealtimeLangSamSubgraph` can never use a term that was not exported.
- `PromptedLangSamSubgraph` with `prompted_gdino_backend: trt` can use any **subset or reordering** of
  the baked terms at runtime — `build_prompt_remap` renumbers, no re-export.
- A genuinely new term needs a re-export **and** a rebuild. The error message says so and prints both
  commands.
- The parity image must actually contain the prompted classes, or the export's faithfulness gate is
  meaningless (`--min-detect`, default 0.30).

### SAM: one stage, three gates

SAM's encoder exports from the same environment the container already has, so it is a single stage
that can run entirely inside the container:

```bash
python3 sam_trt_export.py --sam-type sam2.1_hiera_tiny \
        --from-config <app>/python/tcn_shm_vlm_inference.yaml --out /srv/models/active/sam2
```

All three gates always run, and the artifact is moved into place **only if every one passes** — so a
present engine file is a passed engine file:

1. **Feature fidelity** — the engine's three output tensors against the PyTorch reference.
2. **Slice consistency** — at `--batch N ≥ 2`, identical images must produce identical slices.
3. **Mask IoU** — the unchanged PyTorch decoder run on both sets of features; masks must agree to
   IoU ≥ 0.99. This is the gate that proves the swap is safe end to end.

Precision: FP16 by default (`--tf32` to compare). On TensorRT 11 this required a **strongly-typed**
export, because TRT 11 removed `BuilderFlag::kFP16`; a build that silently skipped the flag produced a
file named `_fp16` that ran at TF32 speed. If you rebuild, check the reported timing, not the filename.

Note SAM's encoder takes **only the image** — no prompts, no classes — which is why it stays a fixed
TensorRT engine even in the promptable variant.

### The container

The reference deployment runs **TensorRT 11.2 on a locally rebuilt Holoscan SDK image**, because the
stock image ships TRT 10.x and SAM's FP16 encoder measured ~2.2× faster on TRT 11. That rebuild is a
deliberate, documented deviation, not a supported configuration:

- It patches the SDK's `holoinfer` for TRT 11 (`kFP16` and `kPREFER_PRECISION_CONSTRAINTS` were
  removed from `nvinfer1::BuilderFlag`) — see `docs/patches/`.
- **Every engine must be rebuilt** after the switch, and engines for the old TRT are kept in a
  separate tree so the previous setup remains a rollback.
- `docs/trt11_build_test.sh` exists so the multi-stage rebuild is approved and run **once** rather
  than as a series of prompts. It asserts the resulting TensorRT version, because a build that
  silently ignored `--build-arg` produced a working-looking image with the wrong TRT.

The full procedure — SDK image, app containers, engines, mount switch, verification, rollback — is
[`docs/trt11-upgrade-runbook.md`](../../../applications/tcn_artekmed/tcn_shm_vlm_inference/docs/trt11-upgrade-runbook.md).

If you do **not** want a modified container: set `gdino_backend: pytorch` and `sam_backend: pytorch`.
Everything still runs, at a substantially lower frame rate, with no engines and no rebuild. That is
also the fastest way to check that a problem is not engine-related.

### Verifying a rebuild changed nothing it should not

Engines are the one dependency whose replacement can silently change output. The application supports
a byte-comparison gate for exactly this: set `mask_dump_dir` to write one `.npy` panoptic map per
camera per frame from a deterministic replay run, do it before and after, then

```bash
python3 docs/compare_mask_dumps.py <before_dir> <after_dir>
```

A structural problem (missing directory, differing frame counts) exits with a distinct code from an
ordinary mask difference, so "the gate could not run" cannot be mistaken for "the masks differ".

## Prompts and class ids

Class ids are **1-based positions in the prompt list** (`class_id_map`), so prompt order is part of
the wire format: reordering `text_prompts.prompts` renames every class downstream, including in the
point-cloud colours and any `select_classes` configuration. For the realtime path it also invalidates
the exported engines.

## Tests

Host tests, no holoscan or cupy needed — run them directly:

```bash
cd operators/tcn_artekmed/tcn_langsam/tests
python3 test_gpu_workers.py        # worker/engine-batch planning        (8 checks)
python3 test_prompt_remap.py       # prompt token remapping             (11)
python3 test_gdino_postprocess.py  # detection postprocess, batched     (7)
python3 test_panoptic_paint.py     # packing and paint order            (15)
python3 test_langsam_multicam.py   # multicam wiring and batch padding  (19)
```

Correctness of the produced masks is gated end to end by the replay harness plus
`docs/compare_mask_dumps.py` in the application, using `MaskDumpOp` — see the application README.
