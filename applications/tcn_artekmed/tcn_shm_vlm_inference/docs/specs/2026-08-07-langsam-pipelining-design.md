# Pipelined LangSAM operators (design)

Split `LangSamBatchOp` into three chained operators so a worker's stages overlap across frames,
instead of running sequentially inside one `compute()`.

Implements step 3 of [`../dataflow-and-pipelining-roadmap.md`](../dataflow-and-pipelining-roadmap.md).
Read Part 1 of that document for the flow this preserves.

## Motivation — the gate result

Step 1 of the roadmap was a gate: measure how much of each stage is GPU work before funding the
rewrite, because the same idea (2-GPU split) **cost 11%** early in this project when the dominant
stage was GIL-bound. Measured 2026-08-07 from the 2026-08-05 trace, attributing each kernel to the
NVTX range it was launched inside:

| stage | GPU | wall | kernel | CPU | kernel% |
|---|---|---|---|---|---|
| gdino | 1 | 69.1 | 62.0 | 7.1 | **90%** |
| gdino | 0 | 76.3 | 54.9 | 21.4 | 72% |
| **sam** | **1** | **86.9** | **40.4** | **46.5** | **46%** |
| sam | 0 | 60.4 | 31.4 | 29.0 | 52% |
| panoptic | 1 | 6.9 | 1.4 | 5.4 | 21% |
| panoptic | 0 | 6.9 | 1.1 | 5.9 | 15% |

GDINO is already GPU-bound; essentially all the non-GPU cost is the `sam` stage's per-image Python
decode loop (46.5 ms/tick on GPU 1) plus panoptic's ~5.5 ms.

Two independent floors on the pipelined tick period:

- **GPU floor** — a worker's own kernel time cannot overlap with itself: GPU 1 = **103.8 ms**.
- **GIL floor** — pessimistically assuming *every* microsecond of CPU time holds the GIL, and both
  workers share one interpreter: 56.2 + 59.0 = **115.2 ms**.

Against today's **177–181 ms**, that is **+55% to +72%** (8.7–9.6 fps). The gate clears in both
directions, which is why this is worth building.

## Design

### 1. Three operators, in a new module

`python/langsam_pipelined.py` holds `GdinoOp`, `SamOp`, `PanopticOp`.
`langsam_multicam_fragment.py` already carries `LangSamBatchOp`, `MaskCollectorOp` and
`LabelMapColorizeOp`; adding three more would make it the wrong size for one file.

```
color_input ─→ GdinoOp ─→ SamOp ─→ PanopticOp ─→ MaskCollectorOp ─→ LabelMapColorizeOp
                (per worker, all three pinned to that worker's device)
```

Each stage keeps its existing NVTX range name (`gdino`, `sam`, `panoptic`) so every analysis
recipe in the playbook and the roadmap keeps working unchanged.

### 2. What crosses the boundaries

Holoscan Python operators can emit arbitrary Python objects — `LangSamBatchOp` already emits a
`dict`. Payloads are plain dicts; GPU arrays cross as-is (no copy, no host round-trip).

**GdinoOp → SamOp**

| key | type | note |
|---|---|---|
| `names` | `list[str]` | camera port names, in order |
| `hw` | `(H, W)` | see §5 |
| `sam_idx` | `list[int]` | indices into `names` that had ≥1 detection |
| `sam_imgs` | `list[torch (H,W,3) uint8 cuda]` | **only the detecting cameras** |
| `sam_boxes` | `list[cupy (K,4)]` | pixel xyxy, on device |
| `sam_labels` | `list[list[str]]` | prompt strings |

Non-detecting cameras' frames are dropped here rather than carried, which also frees that memory
one stage earlier.

**SamOp → PanopticOp**: `names`, `hw`, `sam_idx`, `sam_labels`, plus
`masks: list[cupy (M,H,W) uint8]` and `scores: list[cupy (M,)]`.

**PanopticOp → collector**: unchanged — `{"<cam>_mask": holoscan tensor}`.

### 3. Model ownership and the prompt invariant

`GdinoOp` owns the detector (TRT or PyTorch) and `self.prompts` (for labels); `SamOp` owns the SAM
model; `PanopticOp` owns `_cmap`.

This **splits an invariant across operators**. Today `_apply_prompts` is the single place keeping
the detector's `token_class_ids`, the label list and `_cmap` consistent — get that wrong and class
ids disagree with mask colours silently. After the split, `GdinoOp` and `PanopticOp` each derive
their half from the same `prompts` value in `compose`, which holds because prompts are static
config. **If a `text_prompts` input port is ever added to this path, it must fan out to both
operators**, and that is the moment this design needs revisiting.

> Deliberately *not* fixed here: the class-id → label → class-id round-trip (GdinoOp turns ids into
> strings via `self.prompts[c-1]`, `build_panoptic_map` turns them back via `_cmap`). Passing ids
> directly would delete the invariant above entirely, but it changes a tested helper's signature,
> and mixing a behavioural change into a structural refactor is how regressions hide. Recorded as
> a follow-up.

### 4. Opt-in, with the monolith retained

`gpu_workers.pipelined: true|false`, **default `false`**. `LangSamBatchOp` stays exactly as it is.
`compose` builds either one operator per worker or three. This follows the `gdino_backend` /
`sam_backend` precedent: the new path is opt-in, A/B-able against the old one in a single config
edit, and instantly revertible if the measurement disappoints.

### 5. Behaviour that must not change

- Stage boundaries and NVTX names identical, so before/after traces are comparable.
- `hw` keeps its current (wrong but harmless) semantics — the **last** camera's shape, used for
  every panoptic map. Preserving the quirk keeps this refactor behaviour-neutral; it is recorded in
  [`../deferred-findings.md`](../deferred-findings.md) §4 and should be fixed separately.
- Boxes and masks stay on the GPU end to end.
- Empty cases: no cameras in the message, and no detections on any camera, must still emit a
  well-formed all-background result for every camera.

## Testing

| test | where | gate |
|---|---|---|
| existing host suites | host, numpy | 8/8, 11/11, 7/7, 9/9 unchanged |
| module parses; no stale references | host, `py_compile` + grep | the only host check possible |
| masks identical, pipelined vs monolithic, same input | container | the correctness gate |
| `gdino` / `sam` / `panoptic` NVTX + tick period | container, nsys | vs 69.1/86.9/6.9 and 177.2 ms |
| end-to-end frame→mask latency | container, `--tracking` | expected to roughly double; quantify it |

> **The operators cannot be host-tested.** They import torch, cupy and holoscan, so unlike
> `langsam_helpers` there is no numpy-only surface to exercise off-GPU. An earlier draft of this
> table claimed otherwise. The real correctness gate is therefore the container A/B: **masks
> identical between `pipelined: false` and `pipelined: true` on the same input**. Everything pure
> already lives in `langsam_helpers` and is unchanged by this work, so its suites still apply.

## Risks

- **The GIL may bind tighter than the pessimistic floor predicts.** Mitigation: the toggle. If the
  measurement disappoints, flip it back in one line and pursue roadmap steps 2 and 4 instead, which
  *remove* CPU time rather than hiding it.
- **Queue depth.** Overlap requires the upstream to run while the downstream computes. Holoscan
  pops a message into the operator at compute time, so a capacity-1 queue should already allow
  2-deep pipelining; if the trace shows no overlap, raise the receiver capacity before concluding
  the GIL is at fault.
- **Memory.** Frames and masks for 2–3 in-flight ticks are alive at once (~36 MiB of frames per
  worker per tick, plus masks). Small next to the engines, but it is new.
- **Latency grows by ~2 ticks**, accepted explicitly for this design.

## Out of scope

- `InferenceOp` for the GDINO engine (roadmap step 3's second half) — a separate change, now less
  urgent since GDINO measured 90% GPU-bound.
- The panoptic CUDA kernel (step 2) and the SAM decode loop (step 4).
- Passing class ids instead of label strings (§3).

## Results

Measured 2026-08-08/09, container, `nsys` traces of both configs (5 prompts, 85 s steady-state
window; see `.superpowers/sdd/2026-08-07-langsam-pipelining/measured-results.md` for the full
analysis). **The gate's prediction of +55% to +72% was wrong; the measured result is −12.2%.**

| | monolithic | pipelined | delta |
|---|---|---|---|
| period | 195.0 ms | 221.9 ms | +13.8% |
| fps | 5.13 | 4.51 | -12.2% |
| GPU util dev0 / dev1 | 46.8 / 60.2% | 41.3 / 54.1% | down |

Per-stage wall time (ms/tick), dev0 / dev1:

| stage | mono | pipe |
|---|---|---|
| gdino | 74.81 / 70.10 | 125.58 / 102.65 |
| sam | 63.89 / 89.87 | 205.93 / 220.75 |
| panoptic | 9.23 / 14.65 | 95.72 / 124.16 |

**The Motivation section's gate reasoned from a GIL floor and a GPU floor over stage wall times it
treated as fixed, and named the GIL (Risks, above) as the thing to watch.** Both were incomplete:
the wall times were not fixed, and the GIL was not the dominant cause.

**The pipelining itself worked exactly as designed.** Within-worker gdino↔sam overlap was 100% of
the smaller stage's busy time (total stage overlap 51.8% w0 / 50.7% w1 of the summed stages).
Monolithic period (195.0 ms) tracked the *sum* of its stages (174.6 ms dev1); pipelined period
(221.9 ms) tracked the *max* stage (sam, 220.75 ms). The sum→max conversion this design set out to
produce happened. It lost anyway because each stage's own wall time inflated ~2.5×, so
`max(inflated stages)` came out larger than `sum(original stages)`.

**Why the inflation, and why it is not primarily the Risks section's GIL:** per-tick CUDA call
counts are identical between the two runs (e.g. sam: 923.6 vs 944.5 `cudaLaunchKernel` calls/tick)
— same work, more expensive per call. The median `cuLaunchKernel` duration only rose 1.6× (8.62 →
13.88 us, consistent with GIL handoff cost), but the blocking *tail* rose 27×: calls over 50 us
went from 0.5% to 10.7% of launches, accounting for ~25.8 s of the ~28.4 s total launch inflation
across the trace. `concurrency = kernel_sum / kernel_union` was **1.00 on every device in both
runs** — the GPU work was already fully serialized on the shared legacy default stream in the
monolithic version too (the same assumption `SamOp`'s stream guard in `langsam_pipelined.py`
enforces), so overlapping the *stages* bought zero GPU overlap. What pipelining did instead was
route roughly three stages' worth of kernel launches — ~2400 kernels/frame — through that one
shared stream's queue, and the queue backs up. The per-stage split of the inflation (dev1) shows
this is not uniform: gdino's inflation is essentially all GIL (its own CUDA-API time *fell*, since
the GPU had already finished by the time it synchronized); panoptic's is mostly stream-queue
blocking; sam — the stage that sets the period — splits roughly 2:1 non-API-to-API, i.e. mostly GIL
re-acquisition across its ~1000 Python-driven launches per tick, but with a real stream-queue
component too. In short: the risk that materialized was real GIL cost *plus* a queue-serialization
effect the Risks section did not name, and the second effect is comparable in size to the first.

**Conclusion: pipelining is a prerequisite, not a win on its own.** It only pays once (a) each
stage runs on its own CUDA stream with explicit event ordering across the operator edges instead
of relying on the shared default stream, and (b) the per-stage Python work — sam's decode loop
above all — is substantially reduced. Estimated effect of streams alone, from the same analysis:
panoptic and part of sam recover, but sam still lands ~190 ms, putting the period at roughly
break-even (~190 ms) against the monolithic 195 ms — not a win by itself, for substantial added
complexity. **The default therefore stays `pipelined: false`.** The three-operator code is
retained as the structural foundation for a future retry (see
[`../dataflow-and-pipelining-roadmap.md`](../dataflow-and-pipelining-roadmap.md) step 3), which
would need per-stage streams and explicit cross-edge event ordering, not another rearrangement of
Python calls.
