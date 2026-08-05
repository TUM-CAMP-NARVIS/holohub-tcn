# Optimizing a PyTorch model inside a Holoscan pipeline — methodology and case study

A record of how the LangSAM (Grounding DINO + SAM 2) subgraph in this app went from
**~2 fps on one camera** to **~5.6 fps on five cameras across two GPUs**, written to be reused as
a template for the next network.

The case study is secondary. The reusable parts are **the measurement methodology**, **the order
in which levers paid off**, and **the traps** — several of which cost days here and will cost
them again elsewhere.

---

## 1. How to use this document

Work top to bottom:

1. **§2 Methodology** — build the measurement apparatus *first*. Every wrong turn in §6 came from
   optimizing before attributing.
2. **§3 Lever order** — the levers ranked by what they actually returned, with the reasoning for
   the ranking.
3. **§4 Results** — the full numeric arc, so you can calibrate expectations.
4. **§5 Correctness gates** — how to avoid making something fast and wrong.
5. **§6 What failed** — read this before trying the same things.
6. **§7 Traps** — the measurement and tooling traps specifically.
7. **§8 Checklist** — the condensed template.

---

## 2. Methodology

### 2.1 The rule that matters most

> **Attribute before you act. Then re-attribute after every change.**

Every significant mistake in this project was a violation: optimizing a stage that was not the
bottleneck, or — worse — optimizing against a number that did not mean what we thought (§7.1).

### 2.2 Four measurement layers, cheapest first

Use the cheapest layer that can answer the question, and escalate only when it cannot.

| Layer | Tool | Answers | Cost |
|---|---|---|---|
| **L1 Pipeline** | Holoscan dataflow tracker (`--tracking`) | Which *path* is slow? | ~free |
| **L2 Stage** | In-operator timers + NVTX ranges | Which *stage inside the operator* is slow? | needs `cuda.synchronize()`, so opt-in |
| **L3 System** | `nsys profile` + NVTX + CUDA | Where does the time really go; do workers overlap; is the GPU idle? | heavy trace, ~1 GB |
| **L4 Component** | Standalone micro-benchmark of one component | What does this operation cost in isolation, with warmup and medians? | minutes |

**L1** found the langsam path at ~515 ms/frame while every other path was ~11 ms — that framed
the whole project.

**L2** split it: GDINO ≈ 284 ms (65%), SAM ≈ 151 ms (35%). That single split determined the
order of everything that followed.

**L4 is not optional.** It is what finally corrected the misattribution in §7.1 — the L3 trace had
been read wrongly for weeks, and only an isolated timing loop settled it.

### 2.3 Instrumentation to build into the operator

Both of these are in `langsam2operator.py` / `langsam_common.py` and are worth copying:

**Opt-in per-stage timing.** A `timing: bool` and `timing_log_every: int` config pair, logging a
rolling average. It calls `torch.cuda.synchronize()`, so it *changes* what it measures — keep it
`false` in production and say so in the config comment.

```python
if timing: self._sync(); t0 = time.perf_counter()
...stage...
if timing: self._sync(); self.last_encode_ms = (time.perf_counter() - t0) * 1000.0
```

**NVTX ranges around every stage**, always on (they are nearly free without a profiler attached):

```python
torch.cuda.nvtx.range_push("gdino");  ...;  torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("sam");    ...;  torch.cuda.nvtx.range_pop()
```

Name them for *stages you would act on*, and nest sub-ranges (`gdino_forward`,
`gdino_postprocess`) where you suspect a split. NVTX is what makes an nsys trace answerable
instead of a wall of kernels.

### 2.4 Reading an nsys trace

`nsys stats --report nvtx_pushpop_sum` is a starting point, but with `--tracking` on, the table is
swamped by per-frame Holoscan ranges. Query the SQLite export directly instead:

```bash
# WARNING: nsys reuses an existing .sqlite. Delete it or you will analyse the PREVIOUS run.
rm -f run.sqlite
nsys stats --report nvtx_pushpop_sum --format table run.nsys-rep > /dev/null   # builds run.sqlite
```

```python
import sqlite3, statistics as st
c = sqlite3.connect("file:run.sqlite?mode=ro", uri=True)
rows = lambda q: list(c.execute(q))

# 1. Window. Range STARTS, not MAX(end): a stray unclosed range blows up the span.
ss = sorted(s for (s,) in rows("SELECT start FROM NVTX_EVENTS WHERE text='gdino'"))
CUT = ss[0] + 600*10**9        # clamp to the capture duration; drop post-run strays

# 2. Per-stage, per-thread distribution (mean AND p50/p95 — tails matter)
for stage in ("gdino", "sam", "panoptic"):
    for (tid,) in rows(f"SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE text='{stage}'"):
        d = sorted((e-s)/1e6 for s, e in rows(
            f"SELECT start,end FROM NVTX_EVENTS WHERE text='{stage}' AND globalTid={tid} "
            f"AND end IS NOT NULL AND start<{CUT}"))
        print(stage, tid, len(d), st.mean(d), d[len(d)//2], d[int(len(d)*.95)])

# 3. Tick period -> fps (gaps between consecutive range starts on one thread)
# 4. GPU utilisation per device
rows("""SELECT deviceId, SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY deviceId""")
# 5. Which GPU does a thread drive? Join kernels to their launching thread:
rows("""SELECT k.deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
        WHERE r.globalTid = <tid> GROUP BY k.deviceId""")
```

Three questions worth asking of every trace:

- **Is the GPU actually busy?** Ours sat at 51–62%. A half-idle GPU means the bottleneck is CPU,
  synchronisation, or scheduling — not compute, and no kernel optimization will help.
- **Do parallel workers overlap?** Compute the union of each worker's busy intervals and their
  intersection. We measured **99.8% overlap**, which is what proved the multi-GPU split was no
  longer GIL-bound. Without that number, "we added a second GPU and it got faster" is a guess.
- **Do the stage totals explain the tick period?** If `sum(stages) << period`, time is being lost
  to scheduling or upstream starvation, not to your model.

### 2.5 Micro-benchmark discipline (L4)

```python
for _ in range(5): run()          # warmup: allocator, autotune, lazy init
torch.cuda.synchronize()
ts = []
for _ in range(20):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    run(); torch.cuda.synchronize()
    ts.append((time.perf_counter()-t0)*1000)
ts.sort(); median = ts[len(ts)//2]     # median, never mean — first iterations are outliers
```

Report medians of ≥20 iterations after ≥5 warmups, and always sanity-check the isolated number
against the in-pipeline NVTX number. When they disagree, one of them is measuring the wrong
thing — find out which before acting.

---

## 3. The levers, in the order they paid off

Ranked by return in *this* project. The ranking itself is the transferable part: it is roughly
"reduce work" → "stop moving data" → "amortise fixed cost" → "change the runtime".

### L1. Reduce the input resolution (biggest single win, near-zero effort)

GDINO cost is **resolution-bound**: the deformable-attention encoder and the text-image fusion
scale with token count, i.e. with H×W. The HF processor defaulted to `shortest_edge=800`, which
was **upscaling** a 1208×680 camera frame to 1333×750 — paying for tokens that carry no extra
information.

Making it configurable (`gdino_input_size: 512`) was the single largest GDINO win of the project.

> **Transferable:** check what your preprocessing actually does to the input size before
> anything else. An upscale hidden in a library default is free money. Verify with the *actual*
> tensor shape at the model boundary, not the config you think is in effect.

### L2. Stop the data leaving the GPU

Two changes, both large:

- **GPU-native GDINO preprocessing** (`predict_gpu`): frame stays on the GPU via
  `torch.from_dlpack`; resize/normalize on GPU replicating the HF processor exactly (read
  `image_mean`/`image_std`/`rescale`/`size` *from* the processor rather than hardcoding); text
  tokenization cached because prompts are static. Removes PIL, the HF image processor, and a
  host round-trip per frame.
- **GPU-resident SAM masks** (`predict_batch_gpu`): SAM 2's `predict_batch` ends with
  `.float().cpu().numpy()` on full-resolution masks. Returning cupy `uint8` via dlpack instead cut
  SAM decode+post from **~75 ms to ~13 ms**.

The second one is the instructive case: we assumed the SAM decoder was slow. It wasn't — the
cost was a redundant full-resolution float mask round-trip GPU→host→GPU. **Measure the transfer,
not just the compute.**

### L3. Batch across independent inputs

Two distinct batching wins:

- **SAM batching across cameras**: per-camera SAM ~70 ms (batch 1) → ~30 ms/camera at batch 3,
  purely from amortising the encoder.
- **GDINO batching across cameras** (this session): 38.64 ms/camera at batch 1 → **27.0
  ms/camera at batch 3**, a 30% cut in engine time, giving −49.0 ms on the bottleneck worker.

> **Transferable:** batching helps even when each item is independent and the GPU looks busy,
> because it amortises launch overhead and improves occupancy. Try it before harder things.

**Size the batch per worker, not globally.** We first shipped one batch-3 engine shared by a
2-camera and a 3-camera worker, so the smaller worker padded and paid 3-camera cost **on every
stage** — which silently cancelled a later optimization on that worker entirely. Giving each
worker an engine at its own camera count recovered 21.3 ms of GDINO and 8.6 ms of SAM.

But the recovery is **bounded, and less than proportional**: a batch-2 engine is not ⅔ the cost
of a batch-3 one (measured 75.2 ms, not the ~60 predicted), because per-camera cost *rises* as
the batch shrinks — the amortisation that made batching a win, running in reverse. Budget for
partial recovery, not linear scaling.

### L4. Export to TensorRT

GDINO PyTorch ≈ 112 ms/camera → TRT ≈ 38.6 ms/camera (~2.9×), and it **releases the GIL**, which
is what made multi-GPU scaling work at all (§L5).

This is high-value but by far the highest-effort lever: a two-stage export pipeline, four
separate tracing gotchas, version-locked artifacts, and an accuracy regression still open. Budget
accordingly. See `gdino_trt_export.md` for the full procedure.

### L5. Parallelise across GPUs — but only after the GIL is gone

The 2-GPU split was tried early and **cost 11%**. The same split, after GDINO moved to TensorRT,
gives **99.8% worker overlap** and near-linear scaling.

> **Transferable:** in a Python pipeline, multi-GPU parallelism is worthless until the dominant
> stage releases the GIL. Measure overlap (§2.4) rather than assuming; the failure mode is
> silent, just no speedup.

### L6. Cheap hygiene (do early, expect little)

bf16 `torch.autocast`; deleting per-frame `gc.collect()` / `torch.cuda.empty_cache()`;
eliminating a redundant RGB copy. Individually small, collectively worthwhile, zero risk. Do them
first because they cost minutes — but do not expect them to change the picture.

---

## 4. Results

### 4.1 The arc

| Stage of work | Config | GDINO | SAM | Result |
|---|---|---|---|---|
| Baseline | 1 camera, PyTorch | 284 ms | 151 ms | ~515 ms/frame, ~2 fps |
| + hygiene, resolution, GPU-native paths | 1 camera, PyTorch | 112 ms | 70 ms | ~182 ms op work, ~4–5 fps |
| + multi-camera, 2 GPUs, GDINO→TRT (batch 1) | 5 cameras | 96.5 / 118.0 ms | 67.3 / 90.6 ms | 221.8 ms tick, **4.51 fps** |
| + GDINO batched (batch 3) | 5 cameras | 95.5 / 69.0 ms | 72.4 / 93.3 ms | 194.5 ms tick, **5.14 fps** |
| + SAM encoder → TRT FP16 | 5 cameras | 94.9 / 69.1 ms | 72.8 / 78.3 ms | 189.9 ms tick, **5.27 fps** |
| + SAM engine per worker | 5 cameras | 96.5 / 69.1 ms | **64.2** / 80.8 ms | 185.3 ms tick, **5.40 fps** |
| + GDINO engine per worker | 5 cameras | **75.2** / 69.6 ms | 60.6 / 87.5 ms | 177.6 ms tick, **5.63 fps** |

Paired values are GPU 0 (2 cameras, also drives Holoviz) / GPU 1 (3 cameras).

Overall: **~2 fps on 1 camera → ~5.6 fps on 5 cameras** ≈ **14× more camera-throughput**.

Note how the bottleneck migrates: GDINO on one camera, then the 3-camera worker, then the
2-camera worker (which was padding), then back to the 3-camera worker. **Re-attribute after every
change** (§2.1) — three of these steps optimised a stage that had only just become the leader.

By the last step GPU utilisation had *fallen* to 47.6% / 56.3%, i.e. the pipeline stopped being
compute-bound and became latency-bound. That is a signal to change lever class, not to keep
tuning kernels: what remains is serialisation (the three stages run sequentially inside one
operator's `compute`), not slow math.

### 4.2 The final batching step in detail

| worker | gdino before | gdino after | change |
|---|---|---|---|
| GPU 1, 3 cameras | 118.0 ms | **69.0 ms** | **−49.0 ms (−41.5%)** |
| GPU 0, 2 cameras (pads 2→3) | 96.5 ms | 95.5 ms | −1.0 ms |

The 2-camera worker performs 50% *more* GDINO work (3 slices for 2 cameras, padded) in the same
wall time — the padding cost is fully absorbed. Workers ended up balanced (175.4 vs 172.1 ms,
previously 215.2 vs 170.6), so further gains now need both to improve.

GPU utilisation moved from 51.2% / 62.4% to 61.4% / 56.8%.

### 4.3 Component micro-benchmarks (container TRT 10.9, medians of 20)

| Component | Measurement |
|---|---|
| GDINO engine, batch 1 | 38.64 ms/execution |
| GDINO engine, batch 3 | 81.03 ms/execution = **27.0 ms/camera** |
| SAM encode eager, batch 1 / 2 / 3 | 16.4 / 32.0 / 45.5 ms |
| SAM encode `torch.compile`, batch 1 / 2 / 3 | 12.7 / 21.0 / 27.4 ms |
| TRT Myelin graph load+unload | 1.5 s over 588 s = **0.25%** (investigated, non-issue) |
| SAM stage (encode+decode) eager vs TRT-FP16 encoder, batch 3 | 73.3 → **59.0 ms (−19.5%)** |
| SAM TRT encoder output dtype: fp32 / bf16 / fp16 | 59.0 / 57.9 / 59.5 ms (cast not worth it) |
| GDINO engine, batch 2 (per-worker) vs batch 3 padded | 75.2 vs 96.5 ms in-pipeline |

---

## 5. Correctness gates

Speed without a fidelity check is not an optimization, it is a regression you have not noticed
yet. Every model-transforming step here is gated.

**Gate the export against the original model.** The GDINO export compares the TRT engine's top
box against a PyTorch reference captured *before* export, requiring IoU ≥ 0.99. This caught a
genuinely broken engine (see §7.4).

**Gate the transformation separately from the model.** When batching, two different questions
need two different thresholds:

- *Is batching correct?* Compare slices of one batched run against each other — identical inputs,
  so demand near-exactness (IoU ≥ 0.999; we measured **1.000000**). **Blocking.**
- *Is the engine faithful to PyTorch?* A different question with a different tolerance, and it
  can fail for reasons unrelated to your change. Ours does (§7.5): **reported loudly, not
  blocking**, so a pre-existing issue cannot block every build.

**Gate against the highest-precision reference available — not the regime you happen to ship.**
This nearly cost us a good engine. The SAM encoder gate first compared an **FP16** engine against
a **bf16** reference (bf16 is what production runs) using fixed thresholds, and failed at
relative error 0.0512. But bf16 has *fewer* mantissa bits (8) than FP16 (10), so the measurement
conflated the engine's error with bf16's own — and bf16's own deviation from FP32 on that tensor
turned out to be 0.0511, essentially the entire "failure". Re-gated against an FP32 reference,
requiring the engine to be no further from it than the bf16 path already is, the engine measured
**0.23–0.28× bf16's error** — three to four times *more* faithful than what was already shipping.

> **Transferable:** when a fidelity gate fails marginally, check what it is measuring *against*
> before you touch the threshold. Comparing two lossy formats to each other tells you they
> differ, not which is wrong. And express the bar as "no worse than what we already accept",
> which is a criterion with meaning, rather than as a constant someone picked.

**Prove the artifact still depends on its inputs.** A gate comparing one image against a
reference cannot catch an export that ignores its input entirely — every image gives the same
(plausible) answer. Add an explicit check: run two different inputs (we mirror one) and require
the outputs to differ. This mattered when a tool started tracing the same model instance once per
batch: batches 2..N were traced from an already-traced model, exactly the cached-feature hazard
in §7.4, and the image-independence gate was the only check able to prove it had not fired.

**Never let a failed gate leave the artifact installed.** Build to a temp path and move into
place only after every gate passes — and **keep the temp artifacts on failure**, printing the
path. An earlier version deleted them, which destroyed the ONNX, i.e. exactly what was needed to
diagnose the failure. Not installing an unvalidated engine is the requirement; discarding the
evidence never was.

**Run every gate before aborting.** Ours are ordered cheap-to-decisive, and the decisive one
(end-to-end mask IoU) was being hidden by a proxy gate failing first. Collect failures and report
them all, then exit.

**Make the pure logic host-testable.** The numeric post-processing lives in a numpy-only module
(`langsam_helpers.py`) with no torch/cupy/TensorRT imports, so it unit-tests on the host without
a GPU. The batched post-process is verified against the original per-image implementation as an
oracle over randomised inputs — that equivalence test is what made replacing it safe.

**A failed gate must not leave a broken artifact.** Our builder writes the engine *before*
running the gates, so a failed build silently left an unvalidated engine on disk and the failure
went unnoticed for a day. Build to a temp path; move into place only after the gates pass.

---

## 6. What failed, and why

Negative results, with the generalisable lesson.

| Attempt | Outcome | Lesson |
|---|---|---|
| **Smaller backbone** (GDINO base → tiny) | Only −15% | The bottleneck was resolution, not backbone. The BERT text encoder is identical in both; the backbone is a minor fraction. Attribute before substituting. |
| **Temporal decoupling** (GDINO every N frames, reuse boxes) | Reverted | SAM re-segmenting with stale boxes made large masks jitter and bleed. For temporal reuse use SAM 2's *video predictor* mask propagation, which tracks motion — not stale boxes. |
| **`torch.compile`** | Abandoned three times | (1) OOM-killed the app during first-forward compilation inside the streaming loop. (2) **Hard deadlock** with >1 worker: Dynamo's tracer and the autocast cache are *process-global*, so one worker tracing while another exits `torch.autocast` blocks both forever. (3) Measured properly at last: **+4.5% steady state but −23% effective**, because 4 startup compiles cost 284 s of a 586 s run, and both workers traced *concurrently* — the deadlock condition, survived by luck. A TensorRT engine gave −19.5% on the same stage with no runtime compilation at all. |
| **Warming up compiled batch sizes at init** | Measured not to work | A warmed, already-cached batch size still re-traced 20 s later. Size-1 dims are specialised, so a dynamic-batch graph guards `batch != 1`. Verify a mitigation works before building on it. |
| **Batch-dynamic TRT profile** (min 1 / opt 3 / max 5) | Not achievable | `dynamic_axes` declares the batch symbolic and the parsed network shows `-1`, but a `Where` in the fusion attention bakes a broadcast conformable only at the *traced* batch. **The trace's batch is the engine's batch.** TensorRT either fails the build or silently specialises to a static shape. |
| **2-GPU split, tried early** | −11% (worse) | GDINO in PyTorch was GIL-bound. The same split later scaled almost perfectly. Right idea, wrong order. |

---

## 7. Traps

### 7.1 A profiler range is not the whole operation

We read the nsys `myelinGraphExecute` range (11.9 ms/camera) as the TensorRT engine's execution
time and concluded that the remaining ~27 ms of the 39.3 ms stage was removable overhead. **It
was not** — that range covers only the Myelin-fused subgraphs. An L4 micro-benchmark measured
**38.64 ms** for the full execution, i.e. essentially the entire stage.

A whole spec was written on the wrong premise. **Cross-check any profiler-derived attribution
against an isolated end-to-end measurement of the same thing before designing around it.**

### 7.2 nsys reuses a stale `.sqlite`

`nsys stats` will happily analyse a `.sqlite` left over from a *previous* run and report the old
numbers with no warning. Delete it first. We nearly reported a stale run's figures.

### 7.3 Worker/thread order is not stable between runs

Thread A in one trace is not thread A in the next. Ours swapped, which inverted every
per-worker conclusion until caught. **Resolve identity from data** — join CUPTI kernel
`deviceId` to the launching thread, or count per-worker engine executions — never from thread
ordering.

### 7.4 A traced graph can silently ignore its input

`GroundingDINO.forward` caches backbone features and only recomputes them when absent, so any
prior forward pass makes the ONNX trace bake *stale* features and ignore the image input
entirely — producing an engine that returns the same output for every image. Similarly, without
`dynamic_axes` the legacy tracer constant-folds the image branch and drops `img` as a graph
input.

**Test that the exported artifact is actually input-dependent:**
`assert not np.allclose(engine(imgA), engine(imgB))`.

### 7.5 The runtime version can change the numbers

Engines built with the host's TRT 11.2 matched PyTorch at IoU 0.9994. Engines built inside the
container with **TRT 10.9** put the box in nearly the right place (IoU ~0.964) but **depress
confidence scores** (0.52 vs 0.889). With a `box_threshold` of 0.3, marginal detections silently
disappear. *(Still open; suspects are TF32 handling and INT64→INT32 input truncation, which
TensorRT warns about.)*

TensorRT engines are also **version-locked** — an engine built outside the runtime container will
not deserialize inside it. Build where you run.

### 7.6 Deadlocks need a stack, not a guess

A hung pipeline was diagnosed with `py-spy dump --pid <pid>` from inside the container: two
samples 25 s apart showing **identical frames** proved a deadlock rather than slow progress, and
`py-spy dump --locals` showed the two threads were in *different operator instances* — which
overturned the first diagnosis. CPU% and memory alone would have been ambiguous.

### 7.7 Hidden synchronisation in array libraries

In CuPy, boolean-mask indexing (`arr[mask]`) must size its output on the host and therefore
**synchronises**; integer-array indexing and basic slicing do not. Replacing
`probs[..., mask].max()` with `xp.where(mask, probs, 0.0).max()` is equivalent here — sigmoids
are strictly positive — and sync-free. `.get()`, `cp.asnumpy`, and `float()`/`int()`/`.item()` on
a device scalar all synchronise too.

Budget syncs explicitly: our batched detector documents "exactly two synchronisations per call"
and reviews enforce it.

---

### 7.8 An exporter default can change under you

`torch.onnx.export` now defaults to `dynamo=True`. For one of our models that: required an extra
package the container lacks, silently raised the opset then failed converting back down
(`No Adapter To Version $17 for Resize`), and handed the graph to a constant folder that
evaluated `Resize` nodes through ONNX's **pure-Python** reference implementation. On 1024²
feature maps that never finishes — presenting as a hang at ~112% CPU with no output.

Pass `dynamo=False` explicitly if the legacy tracer is what you validated against, and pin the
choice in a comment. More generally: an export that suddenly hangs or needs new dependencies is
more likely a changed default than a changed model.

### 7.9 Watch for the pipeline changing character

GPU utilisation *fell* from 62% to 47–56% across this project as stages got faster. That is not a
regression — it means the work left is no longer compute. Past that point, kernel-level
optimization has poor returns and the leverage moves to removing serialisation (overlapping
stages, cutting syncs, shortening critical paths).

Track utilisation alongside stage times for exactly this reason: it tells you *which class* of
lever to reach for next, which is more useful than another 5% on a stage that is already idle
half the time.

## 8. Checklist for the next network

**Measure**
- [ ] L1: which pipeline path dominates?
- [ ] L2: add NVTX ranges + opt-in timers; which stage inside it dominates?
- [ ] L3: nsys — GPU utilisation, worker overlap, stage totals vs tick period
- [ ] L4: micro-benchmark the dominant stage in isolation; **reconcile with L2/L3 before acting**

**Cheap wins**
- [ ] What resolution does preprocessing actually feed the model? Is it upscaling?
- [ ] Mixed precision (bf16 autocast)
- [ ] Remove per-frame `gc.collect()` / `empty_cache()`
- [ ] Count host↔device transfers per frame; each is a candidate

**Structural**
- [ ] Keep tensors GPU-resident end to end (dlpack across libraries)
- [ ] Batch across independent inputs (cameras/streams) — measure per-item cost, not total
- [ ] Size the batch **per worker**; a shared batch taxes smaller workers on every stage
- [ ] Track GPU utilisation, not just stage times — falling utilisation means change lever class
- [ ] Only then consider export to TensorRT
- [ ] Only after the dominant stage releases the GIL, parallelise across GPUs — and *measure overlap*

**Gates**
- [ ] Numeric gate of the exported model vs the original, on a representative input
- [ ] Gate against the **highest-precision** reference, and express the bar as "no worse than what
      we already accept" rather than as a hand-picked constant
- [ ] Prove the artifact is input-dependent (two different inputs must give different outputs)
- [ ] Run every gate before aborting, so the decisive one is never hidden by a proxy
- [ ] On failure, keep the intermediate artifacts and print the path
- [ ] Separate gate for the *transformation* (e.g. batch-slice consistency), with its own threshold
- [ ] Artifact only moves into place after gates pass
- [ ] Pure numeric logic in a dependency-free module, host-testable, with the old implementation as oracle
- [ ] Assert the exported artifact is input-dependent

**Record**
- [ ] Every negative result, with the reason — they are as valuable as the wins
- [ ] Every measured number with its conditions (batch, resolution, precision, runtime version)
- [ ] Config comments that explain *why*, not just *what* (e.g. why `sam_compile` must stay false)

---

## 9. Related documents

- [`gdino_trt_export.md`](./gdino_trt_export.md) — the two-stage export procedure and its gotchas
- [`dataflow-and-pipelining-roadmap.md`](./dataflow-and-pipelining-roadmap.md) — the extracted
  compute/data flow and the gated four-step plan for what remains
- [`deferred-findings.md`](./deferred-findings.md) — review findings consciously not fixed, re-verified
- [`sam_trt_export.md`](./sam_trt_export.md) — the single-stage SAM 2 encoder export and its gates
- [`da3_onnx_export.md`](./da3_onnx_export.md) — the Depth-Anything-3 NHWC export
- [`specs/2026-08-04-gdino-batched-inference-design.md`](./specs/2026-08-04-gdino-batched-inference-design.md) — batching design, corrections and measured results
- [`specs/2026-08-05-sam-encoder-trt-design.md`](./specs/2026-08-05-sam-encoder-trt-design.md) — SAM encoder → TRT, with the gate-design correction
- [`specs/2026-08-05-per-worker-engines-design.md`](./specs/2026-08-05-per-worker-engines-design.md) — per-worker batches and the `gpu_workers` topology node
- [`specs/2026-07-28-langsam-multicam-design.md`](./specs/2026-07-28-langsam-multicam-design.md) — the multi-camera architecture
