# SAM 2 image encoder → TensorRT (design)

Replace the PyTorch SAM 2 Hiera image encoder with a prebuilt FP16 TensorRT engine, keeping the
GPU-native preprocessing and the PyTorch decode loop unchanged.

## Motivation (measured)

After the GDINO batching work, **SAM is the largest stage on the bottleneck worker**
(2026-08-05 nsys, 5 cameras, 2 GPUs):

| worker | gdino | **sam** | panoptic | total | period |
|---|---|---|---|---|---|
| GPU 1, 3 cameras | 69.0 | **93.3** | 9.8 | 172.1 ms | 194.5 ms |
| GPU 0, 2 cameras | 95.5 | 72.4 | 7.5 | 175.4 ms | (5.14 fps) |

Splitting the SAM stage using an isolated benchmark (eager encode at batch 3 = 45.5 ms): the
**encoder is ~45.5 ms and the per-image decode loop ~47.8 ms** — roughly half each. This spec
targets the encoder half only.

`torch.compile` was measured as the alternative and rejected: it delivers only **+4.5%**
(period 194.5 -> 186.2 ms) because it accelerates the encoder but not the decode loop, and it
costs **146 s of startup compilation** during which both workers trace *concurrently* — exactly
the condition that has deadlocked this app before. See the `sam_compile` comment in
`tcn_shm_vlm_inference.yaml` for the full measurement.

A TensorRT engine removes Dynamo from the picture entirely: no runtime compilation, no tracer,
no deadlock class, deterministic startup.

## Why this is simpler than the GDINO export

The container already has the `sam2` package (the app imports it) and the checkpoints under
`/srv/models/active/sam2/`. So **both export and build run in the container**, in one command:

- No host/container split, no wingdzero-style fork requirement.
- No TensorRT version-lock problem — we build where we run.

That removes most of the machinery `gdino_trt_export.py` needed.

## Design

### 1. The encoder wrapper

A `nn.Module` wrapping the SAM 2 model to expose exactly the three tensors the predictor needs.
Derived from tier4's `sam2_pytorch2onnx/export_sam2_onnx.py` (Apache-2.0, attributed in the
tool), which was verified line by line against our `SAM._set_image_batch_gpu`:

| wrapper step | equivalent in `_set_image_batch_gpu` |
|---|---|
| `image_encoder(img)` then `conv_s0`/`conv_s1` on `backbone_fpn[0..1]` | `model.forward_image(batch)` — our config sets `use_high_res_features_in_sam: true`, so `forward_image` applies exactly these convs |
| flatten/permute to `vision_feats` | `model._prepare_backbone_features(backbone_out)` |
| `vision_feats[-1] += no_mem_embed` | `if model.directly_add_no_mem_embed:` |
| reshape per level, reversed | the `feats = [...]` comprehension |
| returns `high_res_feats_0, high_res_feats_1, image_embed` | `_features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}` |

Input `(B, 3, 1024, 1024)`; outputs, for `hiera_tiny`, `(B,32,256,256)`, `(B,64,128,128)`,
`(B,256,64,64)`.

**The wrapper must assert** `use_high_res_features_in_sam` and `directly_add_no_mem_embed` on
the loaded model rather than assuming them — a config where either is false would make the
engine silently wrong.

### 2. Fixed batch, as with GDINO

The GDINO work established that **the traced batch is the engine's batch** for that model. SAM 2's
Hiera encoder has no text fusion and may genuinely support a dynamic batch — but the design does
not depend on finding out. `--batch N` is a single integer used for both the trace and the
profile (`min = opt = max = N`), and the artifact is named `_b<N>_`. If the encoder turns out to
be batch-dynamic, that is a free future simplification, not a prerequisite.

N = the largest worker's camera count (3 today). Workers with fewer cameras pad and discard,
reusing `plan_batch_padding` from the GDINO work.

### 3. FP16

The eager path already runs the encoder under bf16 `torch.autocast`, so an FP16 engine **matches
the precision we accept today** rather than lowering it. The TF32 caution that applied to GDINO
(where the PyTorch reference was FP32) does not apply here.

### 4. Gates

Three, in increasing end-to-end scope:

- **Feature fidelity (blocking).** Run the same preprocessed image through the PyTorch encoder
  and the engine; require per-output relative error within tolerance. Compare with cosine
  similarity >= 0.999 per output plus max relative error, not bit-exactness — FP16 vs bf16 will
  differ in the last bits and that is expected.
- **Slice consistency (blocking).** At batch N, every slice of a replicated image must produce
  identical features. Catches a baked-batch defect, exactly as for GDINO.
- **End-to-end mask IoU (blocking).** Feed both feature sets through the *unchanged* PyTorch
  decoder with the same boxes, and require mask IoU >= 0.99. This is the check that actually
  matters: feature-space error is only meaningful insofar as it changes masks.

Artifacts are written to a temp path and moved into place **only after all gates pass** — the
GDINO builder wrote before gating and left an unvalidated engine on disk for a day.

### 5. Runtime

`SamTrtEncoder` in `langsam_common.py`, mirroring `GDinoTrtDetector`:

- loads engine once, reads `engine_batch` from the profile, exposes `batch_error(n)`
- `encode(batch_gpu)` takes the already-resized/normalised `(n,3,1024,1024)` tensor, pads to
  `engine_batch`, runs one `execute_async_v3`, and returns the three feature tensors sliced back
  to `n`
- outputs stay on the GPU as torch tensors — they are assigned straight into `p._features`

`SAM._set_image_batch_gpu` gains a branch: when a TRT encoder is configured, replace the
`forward_image` / `_prepare_backbone_features` / no_mem_embed / reshape block with one
`encode()` call. **The GPU resize/normalize and the `_features` assignment are unchanged**, so
the decode path is untouched.

### 6. Config

```yaml
langsam_inference:
  sam_backend: "pytorch"        # "pytorch" | "trt"
  sam_trt_engine: "/srv/models/active/sam2/sam2.1_hiera_tiny_encoder_b3_fp16.engine"
```

Default `pytorch`, so the engine is opt-in exactly as `gdino_backend` was.

## Testing

| test | where | gate |
|---|---|---|
| existing host suites | host, numpy | 11/11, 7/7, 9/9 unchanged |
| feature fidelity, slice consistency, mask IoU | container, in the tool | as above |
| end-to-end masks | container | detections/colours unchanged |
| `sam` NVTX stage | container, nsys | vs 93.3 ms (GPU 1) / 72.4 ms (GPU 0) |

Success threshold: the `sam` stage on GPU 1 must beat **80.0 ms** — what `torch.compile`
achieved — or the engine is not worth its complexity.

## Risks

- **Hiera may bake its traced batch** (as GDINO's fusion attention did). Mitigated by using the
  fixed-batch design from the start; the slice-consistency gate detects it either way.
- **FP16 may degrade masks.** The end-to-end IoU gate is the check; TF32 is the fallback and
  costs one rebuild.
- **The decode loop is untouched**, so the ceiling for this work is roughly half the SAM stage.
  If the encoder drops from 45.5 to ~25 ms, expect the `sam` stage ~93 -> ~73 ms and the period
  ~194 -> ~175 ms (~5.7 fps, +11%). Not transformative on its own.
- The TRT 10.9 fidelity issue open on GDINO is **not** expected to apply — that one depresses
  confidence scores through the text branch, which the SAM encoder does not have — but the mask
  IoU gate would catch an analogous problem.

## Out of scope

- The SAM decode loop (the other ~48 ms) — a separate, larger piece of work.
- The SAM 2 decoder as a TRT engine; tier4's ctypes runtime is CPU-in/out and would undo the
  GPU-resident path.
- The GDINO TRT 10.9 score-depression investigation.

## Results (measured 2026-08-05)

Built in-container, TRT 10.9.0.34, `sam2.1_hiera_tiny`, batch 3, FP16, engine 56 MiB.

### Gates — all passed

```
feature high_res_feats_0   err_vs_fp32: bf16=0.0156 trt=0.0037 (0.23x bf16)  [OK]
feature high_res_feats_1   err_vs_fp32: bf16=0.0511 trt=0.0131 (0.26x bf16)  [OK]
feature image_embed        err_vs_fp32: bf16=0.0436 trt=0.0124 (0.28x bf16)  [OK]
slice-consistency gate OK (batch 3)
mask IoU (pytorch vs trt features) = 0.9998   [reference coverage 14.4%]
```

**The FP16 engine is 3-4x MORE faithful to FP32 than the bf16 autocast path already in
production.** This only became visible after the gate was fixed: the first version compared the
engine against the *bf16* reference with fixed thresholds and failed at rel_err 0.0512 on
`high_res_feats_1` -- which is almost exactly the 0.0511 that bf16 itself deviates from FP32.
The original gate was measuring bf16's error and attributing it to the engine, and would have
rejected an engine strictly better than what ships. Fixing the measurement, rather than relaxing
the threshold, was what surfaced this.

### Speed (median of 20, 5 warmups, 1536x2048 frames, batch 3, idle GPU)

| SAM stage (`predict_batch_gpu`, encode + decode) | |
|---|---|
| eager PyTorch | 73.3 ms |
| **TensorRT FP16 encoder** | **59.0 ms (-14.3 ms, -19.5%)** |

Projected onto the app's `sam` stage: GPU 1 **93.3 -> ~75.1 ms**, which beats the **80.0 ms**
that `torch.compile` achieved -- with no 146 s startup stall, no concurrent tracing, and no
Dynamo deadlock class. The plan's adoption criterion is met.

### Two hypotheses that did not hold

- **"The TRT encoder returns fp32 and will cost the decoder its flash-attention kernel."** The
  eager path also yields fp32 features (`vision_feats[-1] + no_mem_embed` promotes back to
  fp32), so the SDPA fallback is pre-existing and not introduced by this change.
- **"Casting the engine's outputs to bf16 will recover that kernel and pay off."** Measured:
  bf16 57.9 ms, fp32 59.0 ms, fp16 59.5 ms. 1.1 ms (1.9%) for a precision reduction -- not worth
  it. The engine's outputs stay fp32 as built.

### Still open

The per-image decode loop -- roughly half the `sam` stage and untouched here -- remains the
largest single target inside SAM.
