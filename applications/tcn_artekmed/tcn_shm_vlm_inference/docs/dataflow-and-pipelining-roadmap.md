# LangSAM compute/data flow, and the pipelining roadmap

Written to be picked up cold, later. Part 1 is the flow as it actually executes (so you do not
have to re-derive it from the code). Part 2 is where the time goes. Part 3 is the four-step plan,
with the measurement that gates it. Part 4 records what was considered and rejected, and why.

**State when written (2026-08-05):** 5 cameras, 2 GPUs, GDINO and the SAM encoder both on
per-worker fixed-batch TensorRT engines. Tick period 177–181 ms, **5.54–5.64 fps** (effective
5.43 fps/worker). Session arc: 4.51 → 5.6 fps. See
[`optimization-playbook.md`](./optimization-playbook.md) for how it got here and the methodology
that produced these numbers.

---

## 1. The flow, as it actually runs

One `LangSamBatchOp.compute()` per worker per tick. **Everything below is sequential inside that
single call** — that is the central fact for Part 3.

Code: `python/langsam_multicam_fragment.py` (`LangSamBatchOp.compute`),
`python/langsam_common.py` (`GDinoTrtDetector.detect_batch`, `SamTrtEncoder.encode`,
`SAM._set_image_batch_gpu`, `SAM.predict_batch_gpu`),
`python/langsam_helpers.py` (`gdino_postprocess_batch`, `build_panoptic_map`).

```
msg = op_input.receive("color_input")     composite entity; tensors live on cuda:0
                                          (the stream_splitter's device)
┌─ INPUT (per camera i)                                    NOT inside any NVTX range
│    cp.asarray(t) → torch.from_dlpack        (H,W,4) BGRA uint8 on cuda:0
│    .to(self.device)                        ← CROSS-DEVICE copy for the cuda:1 worker
│    [..., [2,1,0]].contiguous()             → (H,W,3) RGB, a second full copy
│  12 MiB per camera per copy. Worker 1 (3 cameras) moves ~72 MiB/tick here.
│  `hw` is overwritten per camera, so it ends up being the LAST camera's shape (see §4.4).
│
├─ NVTX "gdino"    GDinoTrtDetector.detect_batch(rgb_gpu)
│    per camera: permute(2,0,1) → float32 → /255 → interpolate(512×672, bilinear, antialias)
│    ((x - IMAGENET_MEAN) / IMAGENET_STD)
│    torch.cat → (n,3,512,672);  zero-pad to engine_batch if n < engine_batch
│    6× set_input_shape + set_tensor_address   (img + 5 baked text tensors, cached per batch)
│    execute_async_v3
│    torch.cuda.current_stream().synchronize()              ◀── SYNC 1
│    outputs: logits (B,900,256), boxes (B,900,4) → cp.from_dlpack(...)[:n]
│    gdino_postprocess_batch(logits, boxes, class_masks, hw0, xp=cp):
│        probs = sigmoid(logits)
│        per class c: where(class_masks[c], probs, 0).max(-1)   ← no bool-index, no sync
│        best_cls = argmax+1 ; best_score = take_along_axis
│        xyxy = cxcywh→pixels using EACH camera's own (H0,W0)
│    cp.asnumpy(cp.stack([best_cls.f32, best_score]))         ◀── SYNC 2 (one, whole batch)
│    per camera: np.nonzero(score_h[i] > box_threshold) → cp.asarray(idx) → integer gather
│  ⇒ list of (xyxy_gpu (K,4), class_ids list[int], scores_gpu (K,)) — one per camera, in order
│
├─ PARTITION  cameras with ≥1 detection → sam_imgs / sam_boxes / sam_labels / sam_idx
│             labels are host strings: self.prompts[c-1]
│
├─ pmaps = {i: build_panoptic_map(None, …)}  for every camera   (early-returns zeros, no sync)
│
├─ NVTX "sam"      SAM.predict_batch_gpu(sam_imgs, xyxy=sam_boxes)
│  under bf16 torch.autocast:
│    _set_image_batch_gpu(images):
│       p.reset_predictor(); p._orig_hw = [(H,W) per image]
│       per image: permute → float32 → /255 → SAM2's scripted Resize(1024)+Normalize
│       torch.stack → (n,3,1024,1024)
│       ├── sam_backend "trt":  SamTrtEncoder.encode(batch)
│       │      pad to engine_batch, execute_async_v3
│       │      stream.synchronize()                          ◀── SYNC 3
│       │      → high_res_feats_0 (n,32,256,256) fp32
│       │        high_res_feats_1 (n,64,128,128) fp32
│       │        image_embed      (n,256,64,64)  fp32
│       └── eager: forward_image → _prepare_backbone_features → += no_mem_embed → reshape
│              (also yields fp32 — `+ no_mem_embed` promotes out of bf16)
│       p._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
│    ── then SEQUENTIALLY, n iterations of a PYTHON loop ──
│       p._prep_prompts(None, None, box, None, True, img_idx=i)
│           boxes are already CUDA tensors, so torch.as_tensor is a no-op (deliberate)
│       p._predict(unnorm_coords, labels, unnorm_box, mask_input,
│                  multimask_output=False, return_logits=False, img_idx=i)
│       masks[:,0] → uint8 → cp.from_dlpack(...).copy()   (copy so cupy owns it)
│       iou → cp.from_dlpack(...).copy()
│  ⇒ masks[k] (M_k,H,W) uint8 cupy, scores[k] (M_k,) cupy — per DETECTING camera
│
├─ NVTX "panoptic"  build_panoptic_map per detecting camera
│    order = xp.argsort(scores); order.tolist()             ◀── SYNC per camera (N more!)
│    host loop, descending score: class_id_for_label → instance numbering
│    host loop, ascending score:  pmap[masks[j] > 0] = v    ← bool-mask scatter per detection
│  ⇒ (H,W) uint16, value = (class_id << 8) | instance_id, 0 = background
│
└─ OUTPUT  per camera: cp.ascontiguousarray(pmap) → hs.as_tensor → out[<cam>_mask]
           op_output.emit(out, "masks")

── once per tick, downstream ──
MaskCollectorOp   merges both workers' dicts, consolidates everything onto GPU 0
LabelMapColorizeOp   panoptic uint16 → RGBA via build_panoptic_lut
HolovizOp         (tiled view, on GPU 0)
```

### 1.1 Synchronisation inventory per tick per worker

| # | where | count |
|---|---|---|
| 1 | `detect_batch` stream sync after the GDINO engine | 1 |
| 2 | `detect_batch` single `cp.asnumpy` of class ids + scores | 1 |
| 3 | `SamTrtEncoder.encode` stream sync after the SAM encoder | 1 (TRT backend only) |
| 4 | **`build_panoptic_map`: `argsort(scores).tolist()`** | **one per detecting camera** |

Item 4 is the one that is easy to miss — the detector and encoder were carefully built to two
syncs each, and then the panoptic stage adds N more. It is only ~7 ms of wall time, so it has
never been worth fixing on its own, but it is the cheapest sync to remove (§3, step 2).

### 1.2 Contracts worth not breaking

- **Boxes never leave the GPU.** `_prep_prompts` accepts CUDA tensors; converting to numpy
  anywhere reintroduces a D2H+H2D per camera.
- **`feats` order is load-bearing**: `feats[-1]` is `image_embed`, `feats[:-1]` the two high-res
  maps. The TRT encoder returns `[high_res_0, high_res_1, image_embed]` to match.
- **Prompt-derived state moves together** — detector `token_class_ids`, `self.prompts` (SAM
  labels), `self._cmap` (panoptic classes). `_apply_prompts` is the single point that does it.
- **Engine batch == worker camera count**, derived from `gpu_workers`, never configured. Engines
  are named `_b<N>_`; a worker resolves its own path from its camera count.

---

## 2. Where the time goes

Measured 2026-08-05, 107.7 s window, 1170 ticks (585 per worker). Earlier runs of 587 s agree to
within ±1.5 ms per stage, so these are stable.

| stage | GPU 0 (2 cam) | GPU 1 (3 cam) |
|---|---|---|
| gdino | 76.3 ms | 69.1 ms |
| sam | 60.4 ms | **86.9 ms** |
| panoptic | 6.9 ms | 6.9 ms |
| **Σ stages** | **143.6 ms** | **162.8 ms** |
| tick period (p50) | 180.7 ms | 177.2 ms |
| unaccounted (period − Σ) | ~37 ms | ~14 ms |

The unaccounted part is the INPUT block above (cross-device copies, channel swap), the emit, and
scheduling. It is larger on GPU 0, which also drives the collector, colorize and Holoviz.

### 2.1 The number that decides the architecture

| | Σ stages/tick | GPU-busy/tick | **not on GPU** |
|---|---|---|---|
| GPU 0 | 143.6 ms | 88.0 ms | **≈56 ms (39%)** |
| GPU 1 | 162.8 ms | 104.1 ms | **≈59 ms (36%)** |

(GPU-busy = summed kernel duration for that device over the window, divided by ticks. Overlapping
kernels are double-counted, so treat it as an upper bound on GPU occupancy.)

Whole-device utilisation is **47.8% / 56.5%**. So roughly a third of each worker's stage time is
Python, launch gaps and synchronisation — and both GPUs are idle about half the time. **That is
why the remaining levers are overlap and CPU-side cost, not faster kernels.**

---

## 3. The roadmap

Steps 1→4 in order. Step 1 is a gate: it decides whether step 3 is worth doing at all.

### Step 1 — Measure the overlap ceiling (no code changes)

**Question:** of the ~56–59 ms/tick that is not GPU work, how much is *GIL-held Python* (which
pipelining cannot hide, because a second Python operator would just queue behind it) versus
*launch gaps and waiting* (which it can)?

**Why it gates everything:** the 2-GPU split was tried early in this project and **cost 11%**,
because GDINO was GIL-bound in PyTorch at the time. Same idea, same trap. Splitting into a DAG is
a substantial rewrite; do not fund it on a guess.

**Recipe A — per-stage wall vs kernel time, from any existing trace.** Attribute kernels to the
NVTX range they were launched inside, by joining through the launching thread:

```python
# after: rm -f run.sqlite; nsys stats --report nvtx_pushpop_sum --format table run.nsys-rep
import sqlite3
c = sqlite3.connect("file:run.sqlite?mode=ro", uri=True)
rows = lambda q: list(c.execute(q))
K, R = "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"

for stage in ("gdino", "sam", "panoptic"):
    for (tid,) in rows(f"SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE text='{stage}'"):
        # per range: wall duration, and kernel time for kernels LAUNCHED inside it by this thread
        q = f"""
        SELECT SUM(n.end - n.start) AS wall,
               (SELECT SUM(k.end - k.start) FROM {K} k JOIN {R} r ON k.correlationId=r.correlationId
                WHERE r.globalTid = {tid}
                  AND r.start >= (SELECT MIN(start) FROM NVTX_EVENTS WHERE text='{stage}' AND globalTid={tid})
                  AND r.start <  (SELECT MAX(end)   FROM NVTX_EVENTS WHERE text='{stage}' AND globalTid={tid})
               ) AS kern
        FROM NVTX_EVENTS n WHERE n.text='{stage}' AND n.globalTid={tid} AND n.end IS NOT NULL"""
        print(stage, tid, rows(q))
```
A stage whose kernel time is far below its wall time is CPU-bound *within itself* — the candidate
for C++ (§3 steps 2–3) rather than for overlap.

**Recipe B — direct GIL measurement with py-spy.** `py-spy dump` labels each thread
`active+gil` / `active` / `idle`. Sample repeatedly while the app runs and count how often each
worker thread holds the GIL:

```bash
# py-spy is already installed in the container at /workspace/holohub/.local/bin/py-spy
docker exec -u root <container> bash -lc '
  for i in $(seq 1 60); do
    /workspace/holohub/.local/bin/py-spy dump --pid $(pgrep -f tcn_shm_vlm_inference | head -1) \
      2>/dev/null | grep -E "^Thread .*(Dummy|gil)"
    sleep 0.5
  done' | sort | uniq -c | sort -rn
```
**Interpretation:** if the two worker threads are frequently `active+gil` *simultaneously
contended*, pipelining will serialise anyway and step 3 should be dropped in favour of steps 2
and 4. If they are mostly `idle` (waiting on CUDA) or only one holds the GIL at a time, the
overlap is real and step 3 is worth it.

**Deliverable:** a go/no-go on step 3, plus a per-stage CPU-vs-GPU split that tells you which
stage to attack with C++.

---

### Step 2 — `build_panoptic_map` as one CUDA kernel

Small, isolated, and it removes the only per-camera sync left in the tick.

**Better motivated after the step 3 measurement (2026-08-09):** the pipelining trace showed
panoptic's inflation under stage-overlap was **mostly stream-queue blocked**, not GIL — the
opposite split from gdino and sam — and per-tick launch counts scale badly: panoptic issues
**97–232 kernel launches/tick** on top of the per-camera `argsort().tolist()` sync already known
about here. A single fused kernel removes both the sync *and* most of those launches, which is
now a stronger case than "removes ~7 ms and N syncs" alone. See
[`optimization-playbook.md`](./optimization-playbook.md) §7.11–§7.12 for how that was measured.

**Today** (`python/langsam_helpers.py`): `argsort(scores).tolist()` (D2H sync), a host loop doing
instance numbering, then one boolean-mask scatter per detection into the `(H,W)` uint16 map.

**Target:** one launch. Compute per-detection packed values `(class_id << 8) | instance_id` on
the host from data you *already* have without a sync (`class_ids` comes back from `detect_batch`
as a host list, and the instance numbering only needs the score *order*, which can be produced on
device), then a single kernel that paints `(M,H,W)` masks into the map in ascending-score order.

**Notes for whoever does it:**
- Painting order matters: ascending score, so the most confident detection wins overlaps. A
  parallel kernel must therefore write `max(value_priority)` per pixel rather than racing — pack
  priority into the value or use an atomic max on a (priority, value) composite.
- Instance ids are per-frame, not temporally stable. Keep that property; nothing depends on
  stability but nothing should start assuming it either.
- **There is already a numpy oracle**: `build_panoptic_map` itself, plus
  `python/tests/test_langsam_multicam.py`. Gate the kernel against it exactly as
  `gdino_postprocess_batch` was gated against `gdino_postprocess` — random masks/scores/labels,
  several M, including the empty and unknown-label cases.
- **Careful: there are TWO `argsort(scores).tolist()` sites in `langsam_helpers.py`.**
  `build_label_map` (~line 71) is the superseded predecessor that produced a flat class-id map;
  `build_panoptic_map` (~line 105) is the one in the hot path, producing the packed
  `(class << 8) | instance` map. `build_label_map` now has no runtime caller — only tests and a
  re-export in `langsam_common`. Optimise the right one, and consider deleting the other.
- Expected: ~7 ms/tick and N syncs per worker. Modest alone; it matters because it removes the
  last sync that would otherwise serialise a pipelined stage.

---

### Step 3 — Split GDINO into Preproc → `InferenceOp` → Postproc (gated on step 1)

**STATUS (2026-08-09): the stage-level pipelining variant of this step was built and measured —
DONE, but NOT ADOPTED.** Splitting `LangSamBatchOp` into three chained operators
(`GdinoOp → SamOp → PanopticOp`, `python/langsam_pipelined.py`, gated behind
`gpu_workers.pipelined`) produced **−12.2% fps** (195.0 → 221.9 ms tick period), against a gate
that had predicted +55% to +72%. The pipelining mechanism worked exactly as designed — 100%
within-worker stage overlap, period tracking `max(stage)` instead of `sum(stages)` — but each
stage's own wall time inflated ~2.5× under overlap because all GPU work in both the monolithic and
pipelined runs was already fully serialized on the shared legacy default stream
(`concurrency = kernel_sum / kernel_union = 1.00` on every device, both runs). Overlapping stages
therefore bought queueing contention on that one stream, not GPU parallelism. Full analysis:
[`specs/2026-08-07-langsam-pipelining-design.md`](./specs/2026-08-07-langsam-pipelining-design.md)
§Results and [`optimization-playbook.md`](./optimization-playbook.md) §7.11–§7.13. **Default
stays `pipelined: false`**; the three-operator code remains as the structural foundation below.

**A retry requires per-stage CUDA streams, not just per-stage operators.** Each of `GdinoOp`,
`SamOp`, `PanopticOp` would need its own CUDA stream, with explicit CUDA events ordering the GPU
work across the operator edges (SamOp's GPU work must be provably complete, via an event wait, not
just enqueued, before PanopticOp's kernels that consume its output are launched — and equivalently
GdinoOp→SamOp). **This deliberately invalidates the default-stream guard `SamOp` carries today** —
see the docstring on `SamOp` in `python/langsam_pipelined.py`, which exists precisely because the
current design's correctness depends on every operator sharing the one legacy default stream (item
1 in that docstring: "A Holoscan `CudaStreamPool` attached to this path... would silently
invalidate this"). Introducing per-stage streams is exactly that invalidating change. The guard
must therefore be **replaced with real event-based ordering across the edges, not simply deleted**
— deleting it without adding the event ordering would trade a loud, checked assumption for the
silent corruption the docstring warns about (intermittently corrupt or empty panoptic maps, no
crash).

The original text below describes the `InferenceOp`-based variant of step 3 (GDINO's engine as a
C++ operator) and was written before the above measurement; it is retained because it is a
different, still-unexplored mechanism (GIL release via a C++ operator, rather than stage overlap on
a shared stream) and may still be worth pursuing independently, but do not expect it alone to
produce the sum→max win the original Motivation section projected — that projection is the one
just measured wrong.

Do this for **one worker first**, measure, then roll out.

**Why `InferenceOp` and not a Python operator:** it is a C++ operator, so it releases the GIL for
the entire inference. That is exactly what pipelining needs, and it is the same TensorRT backend
already used for DA2/DA3 in this app (`da3_inference:` in the yaml).

**Shape:**

```
GdinoPreprocOp   (H,W,3) RGB uint8 ×n  →  (n,3,512,672) float32 normalised
     +           the 5 constant text tensors (input_ids, attention_mask, position_ids,
                 token_type_ids, text_token_mask) from gdino_swint_prompts.npz
InferenceOp      is_engine_path: true, pointing at gdino_swint_512x672_b{batch}_tf32.engine
     ↓           → logits (n,900,256), boxes (n,900,4)
GdinoPostprocOp  gdino_postprocess_batch → threshold → per-camera (xyxy, class_ids, scores)
```

**Things to work out, with the reasons they matter:**
- **The five text inputs are constant.** `InferenceOp` binds inputs from the incoming entity, so
  either an upstream operator emits them every tick (a few KB — negligible) or they are bound
  once if the API allows it. Check `pre_processor_map` semantics before designing around it.
- **Use `is_engine_path: true` with our prebuilt engines.** Letting HoloInfer build from ONNX
  would bypass all three of our gates (image-independence, slice-consistency, PyTorch fidelity)
  and our fixed-batch discipline. That is not a trade worth making — see
  [`gdino_trt_export.md`](./gdino_trt_export.md).
- **Fixed batch:** the engine runs at exactly `engine_batch`. Padding currently lives in
  `detect_batch`; it would move into the preproc operator, and the postproc operator must slice
  back to `n` (this exact mistake — padded rows reaching the post-process — was a Critical review
  finding once already; see `deferred-findings.md` history and `git log` for `409324c33`).
- **Latency:** each added pipeline stage costs one tick (~180 ms) of mask latency. Decide whether
  the overlay tolerates ~2 ticks before committing.
- **VRAM:** frames and features for 2–3 in-flight ticks live simultaneously. `_features` alone is
  ~50 MB at batch 3.
- **Re-gate after the switch.** Different execution path, same engine: the mask output should be
  identical, so compare masks against the current implementation on the same input before
  trusting it.

**Expected win:** period tends toward `max(stage)` rather than `sum(stage)`. GPU 1 today:
`max(69.1, 86.9, 6.9) = 86.9` versus `162.8` — up to ~1.9×, i.e. 9–10 fps, *if* step 1 says the
non-GPU time is not GIL-held. Treat that as a ceiling, not a forecast; the earlier batch-2
estimate came in at half its projection (see `optimization-playbook.md` §3 L3).

---

### Step 4 — The SAM decode loop (~48 ms of GPU 1's 86.9 ms)

**STATUS (2026-08-10): option 1 below (batch the decoder across images) was built and measured as
step 4a — DONE, adopted pending the correctness gate.** `sam_batched_decode: true|false`, default
`false`. Measured on the 5-prompt monolithic config: period 187.6 -> 171.0 ms (**-8.8%**), fps
5.329 -> 5.847 (**+9.7%**), GPU busy/frame flat at ~117 ms (confirms the win is launches and
Python, not pixel work). Launch counts dropped in proportion to camera count (dev1, 3 cameras:
`cudaLaunchKernel`/tick 949.1 -> 406.4, -57%) with `gdino` unchanged as a control (69.44 -> 69.45
ms). The design below predicted +26% from the launch-count ratio; actual was +9.7% because part of
sam's apparent cost was un-synced GPU work that relocated to `panoptic` rather than disappearing
(dev1 panoptic 13.48 -> 28.52 ms) — see
[`optimization-playbook.md`](./optimization-playbook.md) §7.15. **Default stays `false` until the
per-mask IoU >= 0.999 correctness gate is run.** Full analysis:
[`specs/2026-08-10-sam-batched-decode-design.md`](./specs/2026-08-10-sam-batched-decode-design.md)
§Results. The 4a trace also surfaced a new lever, Step 5 below, which re-ranks what is left — see
§Re-ranked priority.

**Was the top remaining priority before 4a landed**, promoted by the step 3 measurement above: sam
is the stage that **sets the tick period** under pipelining (period tracked `max(stage)`, and sam
was the max on both devices), and its inflation under stage-overlap split roughly **2:1
non-API-time to CUDA-API-time** — i.e. about two-thirds of its cost is GIL re-acquisition across
its ~1000 Python-driven launches per tick, not CUDA API/stream-queue time. That is a direct
measurement of what this step already suspected qualitatively ("a burst of small torch ops"); it is
no longer hidden behind step 3's overlap, since step 3 is not adopted. Originally deliberately last
because it is the hardest and because step 3 might have hidden part of it behind overlap — it did
not.

**Today:** `SAM.predict_batch_gpu` runs a **sequential Python loop** over images, calling SAM 2's
`_prep_prompts` and `_predict` once per image. Each call is a burst of small torch ops. The
encoder half is already a TRT engine; this is the other half.

**Options, remaining after 4a (re-ranked in §Re-ranked priority below):**
1. ~~Batch the decoder across images.~~ **DONE — see STATUS above (step 4a).**
2. **Export the SAM 2 mask decoder to TensorRT.** The tier4 exporter at
   `/data/models/active/sam2_trt_inference/sam2_pytorch2onnx/export_sam2_onnx.py` has a
   `SAM2Decoder` wrapper and a documented I/O contract (`image_embed`, `feats0`, `feats1`,
   `point_coords`, `point_labels`, `mask_input`, `has_mask_input` → `masks`, `iou_predictions`).
   **Do not reuse tier4's ctypes runtime** — it is CPU-in/CPU-out and would undo the GPU-resident
   path. Reuse only the ONNX exporter, as was done for the encoder
   ([`sam_trt_export.md`](./sam_trt_export.md)).
3. **Trim its syncs and Python overhead** without restructuring — the cheapest probe, and step 1's
   Recipe A will say whether this stage is CPU-bound enough to be worth it.

---

### Step 5 — CUDA graph capture for the GDINO and SAM-encoder TRT engines (new, surfaced by the 4a trace)

**Finding (2026-08-10):** both fixed-batch TRT engines issue roughly 1000 kernel launches per tick
despite each being a single engine execution. `gdino` alone: 569.9 `cuLaunchKernel` + 402.2
`cuLaunchKernelEx` ≈ 972 launches/tick, costing ~8.8 ms of launch-API time plus ~11.0 ms non-API ≈
**~20 ms/tick of pure overhead** for a stage whose real work is one engine call. The SAM TRT
encoder has the same shape.

**Lever:** CUDA graph capture collapses that whole per-tick launch sequence into a single graph
launch. It requires static input/output shapes and stable IO buffer addresses across ticks — both
of which the existing fixed-batch, per-worker engine work
([`specs/2026-08-05-per-worker-engines-design.md`](./specs/2026-08-05-per-worker-engines-design.md))
already provides, since each worker's engine always runs at exactly its own camera count. Applies
to **both** engines: the GDINO detector and the SAM TRT encoder (`SamTrtEncoder.encode`), not just
one.

**Why it is worth trying before Step 4 option 2:** no export pipeline, no model surgery, no
inlining upstream internals — it wraps the existing `execute_async_v3` call in a captured graph and
replays it. That is categorically cheaper than exporting the SAM 2 mask decoder to TensorRT, which
needs a full two-stage export, new tracing gotchas, and version-locked artifacts (see
[`optimization-playbook.md`](./optimization-playbook.md) §3 L4). See §Re-ranked priority below.

### Remaining budget (dev1, period 171.0 ms, GPU floor 117.5 ms)

Measured 2026-08-10, from the same 4a trace:

| stage | wall | of which sync (GPU wait) | addressable overhead | lever |
|---|---|---|---|---|
| gdino | 69.45 ms | 49.29 ms | ~20 ms | CUDA graphs (Step 5) |
| sam | 51.77 ms | 25.33 ms | ~26 ms | CUDA graphs (encoder, Step 5) + Step 4 option 2 (decoder) |
| panoptic | 28.52 ms | 7.50 ms | ~21 ms | C++/CUDA operator (Step 2) |

~67 ms addressable against 54 ms of headroom to the GPU floor — expect less than linear stacking:
some of what looks addressable in one stage may, as with panoptic in step 4a, turn out to be
relocated GPU wait rather than removable overhead (see
[`optimization-playbook.md`](./optimization-playbook.md) §7.15).

### Re-ranked priority (2026-08-10)

The step-4a measurement changes the order, not just the status, of what is left:

1. **Step 2 — panoptic as one CUDA kernel.** Unchanged position: still the cheapest, most isolated
   item (~21 ms + the last per-camera sync), already fully scoped, no new evidence against it.
2. **Step 5 — CUDA graph capture (gdino + SAM encoder), promoted to co-top priority.** New
   evidence: it is now the cheaper way to reach the ~20 ms of gdino overhead and part of the ~26 ms
   of sam overhead than Step 4 option 2 is, because it costs an engine-side wrapper rather than an
   export pipeline, and it pays out on *two* engines at once. It should be tried before funding
   Step 4 option 2, not after — that option's real prize shrinks by however much of sam's ~26 ms
   the encoder's share of CUDA graphs recovers, so measuring it accurately requires doing Step 5
   first.
3. **Step 4, option 2 — export the SAM 2 mask decoder to TensorRT, demoted.** Still the right move
   for whatever decoder-side overhead survives Step 5, but it is the highest-effort remaining lever
   (export pipeline, new gates, version-locked artifacts) for what is now a smaller and less
   certain remaining prize. Re-measure sam's addressable overhead after Step 5 before committing to
   it.
4. **Step 3 retry (per-stage CUDA streams + event ordering) — still last.** Unchanged: it is the
   largest rewrite on the board, it was already measured negative once (see Step 3 STATUS above),
   and every other lever above is cheaper per addressable millisecond. Nothing in the 4a trace
   changes that ranking; if anything, discovering that GPU work was relocating more than expected
   *inside a single operator* ([`optimization-playbook.md`](./optimization-playbook.md) §7.15) is a
   reason for more caution, not less, about a design whose whole thesis is inter-operator overlap.

---

## 4. Considered and rejected (do not re-litigate without new evidence)

### 4.1 Persistent kernels — no, on current evidence

A persistent kernel amortises *launch overhead* by keeping a resident kernel pulling work from a
queue. Our measurements do not show launch overhead as a meaningful cost: the GPUs are ~50% idle,
and the dominant GPU cost is the TensorRT engine itself (one fused Myelin subgraph was 27% of wall
time in an earlier trace). Persistent kernels would also fight TensorRT, which owns its own launch
schedule, and add substantial hard-to-debug machinery.

**Revisit if** a trace shows many thousands of tiny kernels dominating a stage — check
`cuda_gpu_kern_sum` for a long tail of short kernels before reaching for this.

### 4.2 Finer-grained DAG decomposition — no

Preprocess / infer / postprocess per model is the right granularity. Below that, each hop pays
entity construction and scheduling; a 7 ms `panoptic` stage does not earn its own operator. Make
it a CUDA kernel inside an existing operator instead (step 2).

### 4.3 `torch.compile` — no, measured three times

OOM-killed the app; then hard-deadlocked with >1 worker (Dynamo's tracer and the autocast cache
are process-global); then measured properly at **+4.5% steady state but −23% effective**, because
4 startup compiles cost 284 s of a 586 s run and both workers traced concurrently. A TensorRT
engine gave −19.5% on the same stage with no runtime compilation. See the `sam_compile` comment in
`python/tcn_shm_vlm_inference.yaml` for the full measurement.

### 4.4 Moving Holoviz off GPU 0 — not possible

There is no third GPU. GPU 0 is ~10% slower than GPU 1 on comparable work because it also drives
the display, collector and colorize; that is a fixed cost of the topology.

### 4.5 Known wart, not yet worth fixing

`hw` in `LangSamBatchOp.compute` is assigned inside the per-camera loop, so it ends up being the
**last** camera's shape, and is then used for *every* camera's panoptic map. Harmless while all
five cameras are 2048×1536. `detect_batch` deliberately carries per-frame `(H0,W0)` through to
pixel scaling, so the per-camera contract holds right up to the operator boundary and then
collapses. If cameras of differing resolution are ever mixed, fix this first. Also listed in
[`deferred-findings.md`](./deferred-findings.md) §4.

---

## 5. Related documents

- [`optimization-playbook.md`](./optimization-playbook.md) — methodology, the measurement layers,
  the trap list, and the full optimisation arc with numbers
- [`deferred-findings.md`](./deferred-findings.md) — known issues consciously not fixed
- [`gdino_trt_export.md`](./gdino_trt_export.md) — the two-stage GDINO export and its three gates
- [`sam_trt_export.md`](./sam_trt_export.md) — the single-stage SAM encoder export
- [`specs/2026-08-05-per-worker-engines-design.md`](./specs/2026-08-05-per-worker-engines-design.md)
  — per-worker batches, the `gpu_workers` topology node, and the measured results
