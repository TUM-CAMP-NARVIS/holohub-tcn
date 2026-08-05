# Per-worker engine batches + consolidated GPU topology (design)

Give each LangSAM worker an engine built for **its own** camera count, and make the camera→GPU
split a single global node that the application and the offline builders both read.

## Motivation (measured)

2026-08-05, after the SAM encoder moved to TensorRT:

| stage | GPU 1 (3 cam) | GPU 0 (2 cam) |
|---|---|---|
| gdino | 69.6 | **94.9** |
| sam | 78.3 | **72.8** |
| panoptic | 7.9 | 5.1 |
| **total** | **155.9 ms** | **172.8 ms** |

Period 189.9 ms, 5.27 fps. **GPU 0 is now the bottleneck**, and it is there for a reason we
created: both engines are built at batch 3, so the 2-camera worker **pads to 3 and pays
3-camera cost on both stages**. The SAM encoder's FP16 gain on GPU 0 was almost exactly
cancelled by that padding (72.4 → 72.8 ms).

GPU 0 is additionally ~36% slower than GPU 1 on *identical* batch-3 GDINO work (94.9 vs
69.6 ms) because it also drives Holoviz, the collector and colorize. That part is not
addressable — there is no other GPU to move the display to — so the padding waste is the
remaining lever.

Estimated: GPU 0 172.8 → ~130 ms, period → ~170 ms, **~5.9 fps (+12%)**.

> This reverses an earlier decision. When only GDINO was batched, per-worker engines were
> rejected as "not worth it for ~4 ms on a non-bottleneck worker." That was correct then. Now
> the waste applies to **both** stages **and** lands on the **bottleneck** worker, so the same
> idea is worth roughly four times more.

## Design

### 1. One global topology node

`gpu_workers` becomes the single source of truth for which cameras run on which GPU. The batch
each worker needs is **derived** from its camera count, never written down separately, so the
two can never disagree:

```yaml
# Single source of truth for the camera -> GPU split. Read by the application AND by the
# offline engine builders (docs/*_trt_export.py --from-config), so the engines that exist
# always match the split that runs. A worker's ENGINE BATCH is its camera count.
gpu_workers:
  workers:
    - device: 0
      cameras: ["camera01_colorimage", "camera02_colorimage"]                     # -> batch 2
    - device: 1
      cameras: ["camera03_colorimage", "camera04_colorimage", "camera05_colorimage"]  # -> batch 3
```

This replaces `langsam_multicam.workers`. `langsam_multicam` holds nothing else, so it goes
away; `langsam_multicam_holoviz` is unrelated and stays.

### 2. Engine paths become `{batch}` templates

They stay in `langsam_inference` beside the model settings that produced them, but gain a
placeholder substituted per worker:

```yaml
langsam_inference:
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_b{batch}_tf32.engine"
  sam_trt_engine:   "/srv/models/active/sam2/sam2.1_hiera_tiny_encoder_b{batch}_fp16.engine"
```

Substitution is `str.format(batch=N)`. A path **without** `{batch}` formats to itself, so an
existing single-engine config keeps working unchanged — useful while only some batches exist.

### 3. Runtime

`LangSamBatchOp.__init__` resolves its own engine paths from its camera count before
constructing the detector/encoder. If the resolved file is missing, fail fast at construction
naming the exact build command for that batch — the same ergonomics as the existing
`batch_error`, but caught before an engine load is even attempted.

The existing padding path stays. With per-worker engines `pad` is normally 0, but nothing
depends on that: a worker may still run an engine built for a larger batch (e.g. only the
batch-3 engine exists yet), which is what makes the migration incremental.

### 4. Builders read the same node

Both tools gain `--from-config <yaml>`: read `gpu_workers`, compute the **distinct** camera
counts, and build one engine per distinct batch. For the current split that is `{2, 3}` — two
engines per model, and no need to remember which.

- `sam_trt_export.py --from-config <yaml>` — one in-container command, builds both.
- `gdino_trt_export.py --stage export --from-config <yaml>` (host) then
  `--stage build --from-config <yaml>` (container) — the two-stage split is unchanged, both
  stages iterate the same batch set.

`--batch N` remains for building a single engine by hand; `--from-config` is mutually exclusive
with it.

## Testing

| test | where | gate |
|---|---|---|
| `resolve_workers` reads `gpu_workers`; batch derived from camera count | host, numpy | new tests |
| `worker_engine_path(template, batch)`: substitutes; no-op without `{batch}`; rejects a template that formats to a still-templated string | host, numpy | new tests |
| distinct-batch computation from a worker list | host, numpy | `{2,3}` for the current split, deduplicated |
| existing suites | host, numpy | unchanged |
| per-batch builds | container | each engine passes its own gates |
| end-to-end | container, nsys | GPU 0 total vs **172.8 ms**, period vs **189.9 ms**, fps vs **5.27** |

## Risks

- **A missing engine for some batch.** Caught at construction with the build command, not at
  first tick.
- **GDINO needs a host re-export per batch** (the traced batch is the engine's batch), so the
  batch-2 GDINO engine needs the wingdzero checkout again. SAM is in-container only.
- **Artifact count doubles** (~2.7 GB for GDINO ONNX+engine pairs, ~120 MB for SAM). Acceptable;
  disk has ~290 GB free.
- **Changing the split now requires rebuilding** the affected batch. The fail-fast guard makes
  that loud rather than silent, and `--from-config` makes it one command.

## Out of scope

- The SAM per-image decode loop (~48 ms) — the largest remaining untouched item.
- The GDINO TRT 10.9 score-depression investigation.
- Moving Holoviz off GPU 0 — there is no other GPU.

## Results (measured 2026-08-05)

Rolled out in two steps so each half could be attributed, with the other worker acting as a
control. nsys, 600 s windows, 5 cameras, 2 GPUs.

### Step 1 — SAM b2 only (GPU 1 still on b3 as control)

| stage | GPU 1 (3 cam, control) | GPU 0 (2 cam, b2) |
|---|---|---|
| sam | 78.3 -> 80.8 (+2.5) | **72.8 -> 64.2 (-8.6)** |
| total | 155.9 -> 159.6 | 172.8 -> 167.6 |

Period 189.9 -> 185.3/186.2 ms, 5.27 -> 5.37/5.40 fps.

### Step 2 — GDINO b2 as well (both models per-worker)

| stage | GPU 0 (2 cam, b2/b2) | GPU 1 (3 cam, b3/b3) |
|---|---|---|
| **gdino** | **96.5 -> 75.2 (-21.3)** | 69.1 -> 69.6 (+0.5) |
| sam | 64.2 -> 60.6 (-3.6) | 80.8 -> 87.5 (+6.7) |
| panoptic | 6.9 -> 6.8 | 9.6 -> 7.1 (-2.5) |
| **total** | **167.6 -> 142.6 (-25.0)** | 159.6 -> 164.2 (+4.6) |

Period 185.3/186.2 -> **177.6/181.4 ms**; median-period fps 5.37/5.40 -> **5.51/5.63**; effective
(stall-free) throughput 5.24 -> **5.38 fps/worker**. GPU busy 60.0/54.0% -> 47.6/56.3%.

**The bottleneck flipped as intended: GPU 0 is now 142.6 ms against GPU 1's 164.2 ms.**

### Where the estimate was wrong

GPU 0's `gdino` was projected at ~60 ms and measured 75.2. **A batch-2 engine is not 2/3 the
cost of a batch-3 one** -- per-camera cost RISES as the batch shrinks, because there is less to
amortise. That is exactly the effect that made batching a win, running in reverse, and it caps
what removing padding can return. Net gain was ~+2.7% effective rather than the projected +6%,
the remainder absorbed by GPU 1 through contention as the pipeline sped up.

### What this leaves

GPU 1 is now the bottleneck at 164.2 ms, of which `sam` is 87.5 ms -- and roughly half of that
is the per-image decode loop, still untouched. GPU utilisation FELL to 47.6/56.3%, so the
pipeline is increasingly latency- and CPU-bound rather than compute-bound: the remaining wins
are in removing serialisation (pipelining the stages across frames) rather than making kernels
faster.
