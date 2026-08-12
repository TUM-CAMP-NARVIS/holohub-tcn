# tcn_langsam

Open-vocabulary segmentation (Grounding DINO + SAM 2) as reusable sub-flows, in two variants that
share their model wrappers, prompt/class mapping and panoptic packing.

## The two variants

| | `RealtimeLangSamSubgraph` | `PromptedLangSamSubgraph` |
|---|---|---|
| module | `realtime.py` (+ `realtime_ops.py`) | `prompted.py` (+ `prompted_ops.py`) |
| cameras | many, split across GPU workers | one |
| GDINO / SAM encoder | prebuilt TensorRT engines, batched per worker | PyTorch |
| prompts | **baked into the engines at export time** | **changed at runtime** (`TextPromptPublisher`) |
| panoptic paint | fused CUDA kernel (`tcn_panoptic_map`), cupy fallback | numpy/cupy |
| throughput | ~4.7 fps for 5 cameras (measured, A40 pair) | substantially lower |

Choose by what the application needs to vary: a fixed vocabulary at speed, or a vocabulary an
operator can retype while the pipeline runs.

They are two sub-flows rather than one switch on purpose. The engine-batch planning that makes the
realtime path fast — one engine per worker, sized to that worker's camera count, with the prompt
tokens embedded — is exactly what makes its prompts static. A single parameterised subgraph would
have to carry both worlds and would make the prompt lifecycle ambiguous at the call site.

## Layout

```
tcn_langsam/
  helpers.py       prompt/class mapping, panoptic packing, worker+engine planning, postprocess,
                   batch planning, validate_source_cameras   (pure; no torch, no holoscan)
  models.py        SAM, GDINO, SamTrtEncoder, GDinoTrtDetector, panoptic map/LUT builders
  realtime_ops.py  GdinoOp, SamOp, PanopticOp -- the split, pipelinable per-stage operators
  realtime.py      LangSamBatchOp (monolithic per-worker op), MaskCollectorOp,
                   LabelMapColorizeOp, RealtimeLangSamSubgraph
  prompted_ops.py  LangSAM2Operator, TextPromptPublisher, LangSamPostprocessorOp
  prompted.py      PromptedLangSamSubgraph
  _viz.py          private drawing/debug helpers used only by the prompted path
  mask_dump.py     MaskDumpOp -- writes per-frame .npy panoptic maps for byte-comparison gates
  tests/           host tests (no holoscan/cupy required)
```

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

# many cameras, fixed prompts, TRT engines
langsam = RealtimeLangSamSubgraph(self, "langsam_multicam", self.kwargs, all_color_cams)
self.add_flow(source, langsam, {("color_outputs", "input")})
self.add_flow(langsam, viz, {("output_viz", "receivers")})
# also: output_masks (packed panoptic maps), output_specs

# one camera, prompts editable at runtime
langsam = PromptedLangSamSubgraph(self, "langsam_single", self.kwargs, allocator, ...)
```

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
