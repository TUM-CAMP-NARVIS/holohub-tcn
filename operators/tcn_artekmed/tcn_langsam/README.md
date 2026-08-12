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
