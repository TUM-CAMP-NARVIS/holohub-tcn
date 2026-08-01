# Grounding DINO TensorRT integration — design (Phase 1)

Date: 2026-08-02
App: `applications/tcn_artekmed/tcn_shm_vlm_inference`
Status: approved-direction (spike complete), pending implementation plan
Precedes: relies on the multi-camera LangSAM pipeline (`2026-07-28-langsam-multicam-design.md`)

## 1. Goal

Replace the PyTorch Grounding DINO forward — profiled as the pipeline's wall (~230 ms,
**launch-bound and GIL-serialized**) — with a **TensorRT engine**, cutting per-frame
detection latency and, because the engine runs from C++/one enqueue, **escaping the Python
GIL** so the 2-GPU split can finally parallelize. No detection-quality change.

## 2. Spike results (established facts — the basis for this design)

Reproduced on-host (A40, TRT 11.2) using IDEA-Research GroundingDINO-T (Swin-T):
- **Export works:** GroundingDINO-T → ONNX (legacy tracer, `dynamo=False`) → TRT engine.
- **Latency:** TRT TF32 @ **512×672 = 38.4 ms/img (26 img/s)** vs HF-PyTorch **~156 ms** (1
  img) / **~230 ms** (batch worker) → **~4×**. (TF32 is the precision the GD-1.5 paper used
  for its 75 FPS headline; FP16 would be faster.)
- **Quality preserved:** real-image top detection vs PyTorch — **IoU 0.9997, score diff
  0.0007**.
- Env constraints found: transformers **4.x** required (5.x breaks GroundingDINO's
  `BertModelWarper`); `torch.onnx.export(..., dynamo=False)`; the traced ONNX is **fixed to
  its traced resolution** (re-export per resolution); TRT 11 dropped the `FP16`/`EXPLICIT_BATCH`
  builder flags (strongly-typed networks — TF32 via flag, FP16 via fp16 ONNX). Spike artifacts:
  `/home/ecku/develop/vision/GroundingDINO-TensorRT-and-ONNX-Inference/`.

## 3. Decisions (open questions resolved)

| Question | Decision | Rationale |
|---|---|---|
| Batch strategy | **Batch-1 engine, looped per camera** in the worker (v1). Batch-N noted as a follow-up. | Simplest, flexible to per-worker camera counts; 38 ms×N still beats 230 ms and is GIL-free. A batch-N engine bakes the count in (worker0=2, worker1=3 → 2 engines) — defer. |
| Prompt handling | **Fixed prompts baked as constant engine inputs** (precompute `input_ids`/masks once). Prompt change ⇒ offline re-export + rebuild. | Our prompts are static; a fixed-token engine is simplest and matches the spike. |
| Precision | **TF32 for v1** (already quality-matched); FP16 opt-in after a parity re-check. | TF32 gave IoU 0.9997 and ~4×; FP16 needs an fp16 ONNX + validation. |
| Runtime | **Custom Python TRT-runtime operator** loading a prebuilt `.engine`; one `execute_async_v3` per inference. `InferenceOp` noted as an alternative. | Full control over GDINO's 6-input/2-output graph + constant text + our TRT-11 build; a single enqueue collapses the launch flood (≈GIL-free) without fighting `InferenceOp`'s ONNX-build path. |
| Engine build | **Offline** build script; engine stored at `/srv/models/active/groundingdino/`, loaded at runtime (like the DA2/DA3 engines). | Keeps the runtime lean; matches existing model-deployment pattern. |

## 4. Model source & I/O contract

- **Checkpoint:** IDEA-Research `groundingdino_swint_ogc.pth` (Swin-T) — equivalent
  architecture/quality to the HF `grounding-dino-tiny` we use now.
- **Engine inputs:** `img` `(1,3,H,W)` f32 (ImageNet-normalized RGB); `input_ids`,
  `attention_mask`, `position_ids`, `token_type_ids` `(1,L)`; `text_token_mask` `(1,L,L)`.
  The text tensors are **constant** for our prompts (precomputed once).
- **Engine outputs:** `logits` `(1,900,256)`, `boxes` `(1,900,4)` cxcywh-normalized.
- `(H,W)` and `L` are fixed at build time. Default `(H,W)=(512,672)` (our ~512 shortest-edge
  config), `L` = token count of the configured prompts.

## 5. Components

- **Offline: `gdino_trt_export.py`** (adapted from the spike). Given checkpoint + prompts +
  `(H,W)`, produce: the fixed-resolution ONNX, the precomputed text tensors (saved as `.npz`),
  and the TF32 (optionally FP16) TRT engine. Documented in `docs/gdino_trt_export.md`.
- **Runtime: `GDinoTrtDetector`** (in `langsam_common.py`) — loads the `.engine` + text
  `.npz` on `device`; method `detect(rgb_gpu) -> {boxes(GPU xyxy), labels(list[str]),
  scores(GPU)}` for one camera frame. Encapsulates: GPU resize+normalize → set tensor
  addresses → `execute_async_v3` → lean post-process.
- **Lean post-process** (replaces HF `post_process`): `sigmoid(logits)`; per-query score =
  max over each prompt's token positions (token→class map known from the prompt tokenization);
  threshold by `box_threshold`; boxes cxcywh→xyxy scaled to camera resolution. Returns GPU
  tensors (SAM already consumes GPU boxes) + class labels.
- **Integration:** `LangSamBatchOp` (multi-cam) and `LangSAM2Operator` (single-cam) select the
  GDINO backend via config; when `trt`, they call `GDinoTrtDetector.detect` per camera instead
  of `GDINO.predict_gpu_batch`. The SAM path and panoptic output are unchanged.

## 6. Data flow (per worker, unchanged except the GDINO box)

```
color entity ─▶ [per camera: RGB GPU tensor]
                     │
                     ├─▶ GDinoTrtDetector.detect(rgb)  (resize+norm → TRT engine → post-proc)  ──▶ boxes(GPU)+labels
                     └─▶ (existing) SAM predict_batch_gpu(rgb, boxes) ──▶ masks ──▶ panoptic map
```

## 7. Configuration (`langsam_inference`)

```yaml
  gdino_backend: "trt"            # "pytorch" (current) | "trt"
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_tf32.engine"
  gdino_trt_text: "/srv/models/active/groundingdino/gdino_swint_prompts.npz"  # precomputed text tensors
  # box_threshold / text_threshold reused from the existing config
```
`gdino_backend: "pytorch"` keeps the exact current path (safe fallback / A-B).

## 8. Error handling & fallback

- Engine/text file missing or arch-mismatched → **fail fast at operator init** with a clear
  message (don't silently fall back mid-stream).
- Prompt list doesn't match the engine's baked token length → fail fast at init (the `.npz`
  records the prompts; verify they equal `text_prompts.prompts`).
- No detections for a frame → empty boxes → all-background panoptic map (existing behavior).
- `gdino_backend: "pytorch"` remains the always-available fallback.

## 9. Testing / validation

- **Offline parity gate (blocking):** the export script re-runs the spike's real-image parity
  (PyTorch vs engine) and asserts top-box **IoU ≥ 0.99** before accepting an engine. FP16
  engines must pass the same gate.
- **Lean post-process unit tests** (host, numpy): sigmoid+token-max scoring, threshold,
  cxcywh→xyxy scaling, token→class mapping — against hand-computed cases.
- **Integration smoke (container):** run with `gdino_backend: trt`, confirm masks match the
  PyTorch backend visually and the panoptic output is unchanged.
- **Perf (container, nsys):** confirm the `gdino`/`gdino_forward` NVTX range drops from ~230 ms
  toward the ~40 ms×N region, GPU util rises, and — key hypothesis — **re-enabling the 2-GPU
  split now scales** (GDINO no longer GIL-bound).

## 10. Staged delivery

1. **Offline export + parity gate** — produce a TF32 engine for our prompts at 512×672;
   pass IoU ≥ 0.99. (De-risks before touching the app.)
2. **`GDinoTrtDetector` + lean post-process** with unit tests; wire the `gdino_backend`
   toggle into `LangSamBatchOp` (single-worker, single-GPU first).
3. **Validate + profile** single-GPU; confirm latency + quality.
4. **Re-test the 2-GPU split** (should now parallelize) and, if beneficial, a **batch-N
   engine**.
5. Optional: **FP16 engine** (re-run parity gate); optional `InferenceOp` runtime variant.

## 11. Out of scope / risks

- **Out of scope:** keyframe + SAM2-video tracking (that's the separate Lever B); GD-1.5 Edge
  model; multi-process/distributed to escape the GIL (the TRT op already largely does).
- **Risks:** engine is fixed-shape (per resolution & prompt-token-length) → re-export on
  change; FP16 numerical quality (gated by the parity check); TRT-11 strongly-typed build
  quirks; carrying the IDEA-Research GroundingDINO source + Swin-T checkpoint into the runtime
  image; batch-1 looping leaves per-image fixed cost on the table until a batch-N engine.
