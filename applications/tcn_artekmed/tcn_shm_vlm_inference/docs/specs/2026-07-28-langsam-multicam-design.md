# Multi-camera LangSAM pipeline — design

Date: 2026-07-28
App: `applications/tcn_artekmed/tcn_shm_vlm_inference`
Status: approved design, pending implementation plan

## 1. Goal

Run the LangSAM (Grounding DINO + SAM2) segmentation pipeline over **all color cameras**
(typically 4–5) instead of a single hard-wired camera, batching inference and spreading it
across the machine's **two NVIDIA A40 (48 GB)** GPUs for best-effort maximum throughput.

The pipeline consumes the shm-subscriber's temporally-aligned color output and produces a
**composite mask entity** with one full-resolution per-camera class-label map, which serves
both downstream processing (e.g. masking depth/pointclouds) and a tiled monitoring view.

## 2. Requirements & success criteria

- Process every color camera per frame through GDINO + SAM2.
- **Success criterion: best-effort maximize throughput** — no hard fps target; take what
  batching + both GPUs give.
- Split cameras across both GPUs (two model instances batching in parallel), with a
  **config-driven assignment**. Single-GPU operation must be selectable via config.
- Output a composite entity of **full-resolution per-camera `uint8` class-label maps**
  (source of truth), plus a tiled Holoviz view (label maps colorized via a class LUT).
- Reuse the already-optimized GDINO/SAM code paths (GPU preprocessing, GPU mask output,
  bf16, optional torch.compile, per-stage timing).

## 3. Input & output contracts

### Input
The subscriber emits `color_outputs`: a single GXF entity holding **named GPU tensors**,
one per color port — `camera01_colorimage` … `cameraNN_colorimage`, each `uint8`
`[H, W, 4]` **BGRA** (2048×1536), temporally aligned, allocated on **GPU 0**
(`camera_stream_processing.device_id`).

### Output
`output_masks`: a composite entity of named tensors `camera0X_mask`, each a full-resolution
`[H, W]` label map.

> **Update (2026-07-29): panoptic (class, instance).** The map is now **`uint16`** packing
> `(class_id << 8) | instance_id` (`0` = background). Query class with `value >> 8`, instance
> with `value & 0xFF`. Instances are numbered per class by descending score (instance 1 =
> most confident) and are **per-frame** (not temporally tracked). The class-only `uint8`
> form below is superseded.

Original class-only contract:
- `0` = background.
- `i` = class `i`, where class ids follow the `text_prompts.prompts` order (`prompt[0]`→1,
  `prompt[1]`→2, …). This mapping is the documented contract for downstream consumers.
- Overlapping detections are resolved to a single class per pixel: detections are painted in
  **ascending mask-score order** so the highest-confidence class wins.

`output_viz` (+ Holoviz input specs): per-camera RGBA tiles produced by colorizing the label
maps through a class LUT, laid out via the existing `create_tiled_input_specs`.

## 4. Configuration

New YAML section. Single- vs multi-GPU is expressed purely by how many workers are declared
(a worker = one model instance on one device processing a camera subset):

```yaml
langsam_multicam:
  workers:                              # one entry per model instance
    - device: 0
      cameras: ["camera01_colorimage", "camera02_colorimage"]
    - device: 1
      cameras: ["camera03_colorimage", "camera04_colorimage", "camera05_colorimage"]
```

- **Default (the `workers` key missing or empty): a single worker `{device: 0, cameras:
  <all discovered colorimage ports>}`.** All streams are processed on GPU 0.
- Single-GPU by choice: declare one worker listing all cameras.
- Reuses the existing `langsam_inference` block (sam_type, gdino_model_id, gdino_input_size,
  box/text thresholds, `gdino_gpu_preprocess`, `sam_gpu_output`, `*_compile`, timing) for
  every worker, and `text_prompts` (shared prompts define the class ids).

## 5. Components

- **`LangSamBatchOp(cameras, device, **langsam_cfg)`** — a batched, device-parameterized
  refactor of today's langsam operator. Builds its own GDINO + SAM on `device`. In: the full
  `color_outputs` entity. Out: an entity of named `uint8` label maps for *its* cameras, on
  `device`.
- **`MaskCollectorOp`** — dynamic `receivers` input (one connection per worker). Consolidates
  every worker's label maps onto GPU 0 and emits one composite entity `{camera0X_mask}`.
  Pass-through (device-normalizing) when there is a single worker.
- **`LabelMapColorizeOp`** — colorizes each label map via a class LUT → per-camera RGBA for a
  tiled Holoviz. Viz only; does not alter `output_masks`.
- **`LangSamMultiCamProcessingSubgraph`** — wires the workers + collector + colorize; exposes
  `input` (color entity), `output_masks` (label maps), `output_viz`/specs.

## 6. Data flow

```
subscriber.color_outputs ─┬─▶ LangSamBatchOp(cuda:0, [cam01,02]) ─┐
   (one entity, all cams,  │                                       ├─▶ MaskCollectorOp ─┬─▶ output_masks (downstream)
    named GPU tensors) ─────┴─▶ LangSamBatchOp(cuda:1, [cam03..05])┘   (→ all on GPU0)   └─▶ LabelMapColorizeOp ─▶ Holoviz (tiled)
```

The subscriber output **fans out** to every worker (Holoscan one-to-many); each worker
**self-selects** its cameras from the shared entity — no stream-splitter needed. The cuda:1
worker copies its cameras' raw images to GPU 1 on entry.

## 7. Batched inference (per worker)

**Extract + preprocess.** For each of the worker's cameras, pull the named GPU tensor
(`[H,W,4]` BGRA, cuda:0). If `worker.device ≠ cuda:0`, copy to the worker's device. Convert
BGRA→RGB via channel reorder `img[..., [2,1,0]]`. Then build:
- **GDINO input:** stack the N RGB frames → NHWC batch; run the existing GPU preprocess
  (resize to `gdino_input_size` + ImageNet normalize) → `pixel_values (N,3,H',W')`. All
  cameras share the same size, so this is one batched tensor. Text is shared: tokenize once,
  tile `input_ids/attention_mask/token_type_ids` to N. One forward →
  `post_process_grounded_object_detection(..., target_sizes=[(H,W)]*N)` → N per-image
  `{boxes, labels, scores}`.
- **SAM input:** `SAM.predict_batch_gpu` already batches N images (`set_image_batch` over N,
  then per-image decode → per-image cupy masks). **Caveat (unchanged from today):** SAM2's
  `set_image_batch` requires numpy, so this keeps one `.get()` per camera (host bounce for
  SAM only). Accepted for this design; a full-GPU SAM encoder path is out of scope.

Reuse: generalize `GDINO.predict_gpu` to a batch (stack + tile); feed `predict_batch_gpu`
the N-image lists (already supported).

**Label-map construction (per camera).** Build `class_id[normalize(prompt)] = index+1` from
`text_prompts.prompts`. For camera k with masks `(Mk,H,W)`, labels, scores:
`label_map = zeros(H,W, uint8)`; paint detections in ascending mask-score order:
`label_map[mask_j > 0] = class_id(label_j)`. Label→class matching reuses the exact/substring
logic from the current postprocessor (`_color_index_for_label`), returning a 1-based class id.
Emit `{camera0X_mask: label_map}` on the worker's device (port `camera01_colorimage` →
tensor `camera01_mask`).

## 8. Device handling

- `LangSamBatchOp.compute` runs inside `torch.cuda.device(self.device)` +
  `cp.cuda.Device(self.device.index)` so all torch/cupy allocations and kernels target the
  right GPU; models are constructed on `self.device`.
- Cross-GPU copies use torch `.to(...)` (P2P when available, host-staged otherwise — correct
  either way): raw image → GPU 1 on entry; label maps → GPU 0 in the collector
  (1 byte/px, ~3 MB/cam).
- Two model instances (GDINO-tiny ~230 MB + SAM-tiny ~150 MB each) fit comfortably on 48 GB.

## 9. Error handling & defaults

- **Missing/empty `workers` → single worker `{device: 0, cameras: <all discovered
  colorimage ports>}`** (derived from `color_streams_config`).
- Camera listed for a worker but absent from the entity → warn + omit from output.
- No detections for a camera → all-background label map at full res (still emitted).
- `device` index ≥ available GPU count → fail fast at construction with a clear message.
- Per-worker inference exception → log and skip that worker's output for the frame; the
  pipeline keeps running.

## 10. Testing

- **Pure-function unit tests** (no models, fast):
  - label-map builder: class-id mapping + higher-score-wins overlap resolution.
  - config resolver: empty/missing `workers` → single GPU-0 worker with all cameras.
  - collector merge: two fake worker outputs → one composite entity, consolidated device.
- **Integration smoke test:** synthetic color entity with 2 random camera tensors → one
  `LangSamBatchOp` on cuda:0 → assert named `uint8` label maps, correct shapes, values in
  `[0, num_classes]`.
- **Manual/perf:** live run; reuse per-worker GDINO/SAM timing; verify tiled viz + downstream
  label maps; compare aggregate throughput single-GPU vs 2-GPU split.

## 11. Code organization & migration

- Extract the reusable `GDINO`, `SAM`, and label/class helpers from `langsam2operator.py`
  into a shared module (e.g. `langsam_common.py`), imported by the new
  `langsam_multicam_fragment.py`. The existing single-camera fragment stays intact and can
  import from the shared module too.
- Wire the app (`tcn_shm_vlm_inference.py`) to feed `subscriber.color_outputs` straight into
  `LangSamMultiCamProcessingSubgraph` and add a tiled Holoviz for the colorized masks. The
  existing single-camera path/config is retained until the multi-camera path is validated.

## 12. Staged delivery

1. **Stage 1 — batched single-GPU:** `LangSamBatchOp` (device-parameterized) processing all
   cameras on GPU 0; label-map output; collector as pass-through; tiled viz. Validate
   correctness + measure batched throughput.
2. **Stage 2 — GPU split:** instantiate a second `LangSamBatchOp` on GPU 1 per the `workers`
   config; activate the collector's cross-GPU consolidation. Validate parity + measure split
   throughput.

## 13. Out of scope

- Full-GPU SAM encoder (bypassing SAM2's numpy `set_image_batch`) and SAM/GDINO TRT engines.
- Dynamic load balancing / frame-level work stealing across GPUs (assignment is static config).
- Changes to the subscriber's allocation device or the depth/pointcloud downstream ops.
