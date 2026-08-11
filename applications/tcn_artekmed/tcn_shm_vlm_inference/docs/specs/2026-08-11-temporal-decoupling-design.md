# Temporal decoupling via SAM 2 mask propagation (design)

Run Grounding DINO every N-th frame and propagate masks on the frames between, using SAM 2's
**video** predictor rather than re-prompting with cached boxes.

**This spec's deliverable is a measurement, not a feature.** Step 1 produces the propagation-cost
and quality-vs-staleness curves. Only those numbers justify the operator work in step 2, and they
are exactly what the previous attempt lacked.

## Why this is the only remaining order-of-magnitude lever

At the 2026-08-11 checkpoint (see
[`2026-08-10-tensorrt-upgrade-assessment.md`](./2026-08-10-tensorrt-upgrade-assessment.md)): period
197.2 ms, dev1 GPU utilisation **80.3%**, dev1 GPU work 158.2 ms/frame of which **gdino is
92.19 ms**. The pipeline is GPU-bound; overhead levers are spent (panoptic is down to 0.54 ms of
GPU). The only way to remove GDINO's cost is to stop paying it every frame.

## Why the previous rejection does not apply

The 2026-08 attempt cached GDINO's **boxes** and re-ran `SAM.predict_batch_gpu` against them. Masks
jittered and the floor bled onto other objects, and it was reverted. That is the expected outcome: a
stale box is a stale *prompt*, so SAM re-segments from scratch against wrong guidance, and the error
grows with staleness.

SAM 2 is a video model. `SAM2VideoPredictor` propagates masklets through **memory attention** — it
carries the previous frames' mask features and tracks the object. That is a different operation from
re-prompting, and it is what the model was trained for. The prior note in project memory already
says as much: *"If temporal is revisited, use SAM2 video predictor mask propagation, not stale
boxes."*

So the rejection was of the cheap approximation, not of temporal decoupling.

## Expected ceiling, and why it saturates

Amortised GPU cost per frame is roughly `gdino/N + propagation`. With gdino at 92 ms:

| N | amortised gdino | period (if propagation ≈ today's SAM) | fps |
|---|---|---|---|
| 1 (today) | 92 ms | 197 ms | 5.1 |
| 4 | 23 ms | ~110 ms | ~9 |
| 8 | 12 ms | ~95 ms | ~10.5 |

**It saturates around 2×**: once GDINO is amortised away, SAM's per-frame Hiera encoder becomes the
floor. Propagation still runs that encoder every frame. So this lever buys a doubling, not an order
of magnitude — worth stating plainly up front, because "realtime" may need more than this alone.

Memory attention may also cost more than the prompt encoding it replaces. That is measured in
step 1, not assumed.

## Step 1 — measure first (the deliverable)

Single camera, offline, on the replay harness so frames are deterministic. Script, not an operator:
`docs/measure_propagation.py`.

For each N in {2, 4, 8, 16}:
1. run GDINO + SAM on frame 0 to seed masklets;
2. propagate with `SAM2VideoPredictor` for N-1 frames;
3. at each offset k, compare the propagated mask against **per-frame GDINO+SAM output for that same
   frame** as ground truth.

Report:

| metric | why |
|---|---|
| propagation GPU ms/frame | decides whether the table above is real; if it approaches 92 ms there is no win |
| per-class IoU vs ground truth, **as a function of k** | the staleness curve; where quality falls off sets the usable N |
| instance-count agreement vs ground truth | catches masklets merging or being lost — invisible to IoU (see the playbook trap) |
| objects entering late | how many frames a new object is missed; this is a product decision, so quantify it rather than argue it |

The dataset is 6 frames per capture, so **this needs a longer export than `k4a_capture`** to measure
staleness beyond k=5. Getting a ~100-frame export is a prerequisite for step 1 and the cheapest
thing to do first.

Gate to proceed to step 2: propagation materially cheaper than GDINO **and** per-class IoU holding
above ~0.9 out to a useful N. If IoU decays fast, the honest answer is that this scene does not
tolerate decoupling and the lever is dead — which is a result worth having for the cost of a script.

## Step 2 — the operator change, only if step 1 passes

Three problems, in descending order of difficulty. None is a blocker; all are design decisions that
must be made deliberately.

### 2.1 Instance identity across a detection boundary

Panoptic values pack `(class << 8) | instance` and instance ids are **per-frame** today (documented
in `build_panoptic_map`). Propagation gives *tracked* masklets, which is strictly better — but on a
re-detection frame you must decide whether to adopt the new detection's ids or match them to
existing tracks. Adopting naively makes colours flicker every N frames, which will look worse than
the jitter this is meant to fix.

Match new detections to live masklets by IoU and keep the existing instance id where they
correspond. This is new logic and belongs in `langsam_helpers` (numpy, host-testable) rather than
inside an operator.

### 2.2 The video predictor is stateful

`SAM2ImagePredictor` is stateless per call, which is why the current operator is clean and why the
deterministic harness works at all. `SAM2VideoPredictor` holds a memory bank **per stream**: five
cameras means five states, persisting across ticks, reset coherently on prompt change.

Consequences: the operator gains real state, so the harness's determinism guarantee must be
re-verified (two identical replays must still match — it is the harness's own gate); and a prompt
change must invalidate every camera's memory, not just re-run detection.

### 2.3 Objects entering the scene

Up to N-1 frames of latency before a new object is segmented at all. Whether that is acceptable is a
clinical/application judgement, not a technical one. Quantified in step 1 so the decision is informed.

## Interaction with what already exists

- **The harness is what makes this tractable now** and was not available in August: deterministic
  frames, scriptable mask comparison, per-class instance counts. Step 1 is a harness measurement.
- Keep it behind a toggle, as with every other lever here (`gdino_backend`, `sam_backend`,
  `sam_batched_decode`, `panoptic_backend`, `pipelined`): `langsam_inference.detect_every_n: 1`,
  default 1 = today's behaviour.
- Orthogonal to the remaining GDINO levers (resolution, the 30.94 ms fused myelin node). Those
  reduce GDINO's cost per detection; this reduces how often it is paid. They compose.

## Out of scope

- CUDA graphs (re-gated down to ~6% on 2026-08-11).
- GDINO FP16 (blocked; see the assessment addendum).
- Any claim that this reaches 30 fps for 5 cameras on 2 A40s. It does not, on this arithmetic.
