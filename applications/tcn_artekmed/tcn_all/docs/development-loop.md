# The edit → test-in-container → verify loop

A development technique for Holoscan pipelines: keep editing on the host, run every check inside the
**already-running** container, and gate each change on a machine-readable verdict. Repeat until the
verdict is green.

This document exists because the loop measurably changed the quality of work on this application. It
is not a style preference — several defects in this codebase were found *only* because a gate asserted
something a human reading the code or watching the viewer would not have checked. Examples are at the
end.

## Why it works for Holoscan specifically

Holoscan development has three properties that defeat ordinary "run it and look" development:

1. **The toolchain only exists in the container.** CUDA, TensorRT, the Holoscan SDK, torch, cupy and
   the dataset reader are installed there, not on the host. Host Python cannot even `import holoscan`.
2. **Failures are asynchronous and partial.** A GXF operator throws inside `compute()` on a scheduler
   thread; a CUDA fault surfaces at the next synchronisation, in a different operator; a use-after-free
   corrupts one consumer on some frames. The error you see is rarely where the bug is.
3. **A live camera stream is not reproducible.** Two runs differ, so "it looks better now" is not
   evidence, and a regression cannot be attributed to a change.

The loop addresses each: the container supplies the toolchain, the verdict makes asynchronous failure
visible as a line of text, and a fixed dataset makes runs comparable.

## The setup

### 1. Source is bind-mounted, so edits are live

Find the mappings rather than assuming them:

```bash
docker inspect <container> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
```

For this project:

| host | container | use |
|---|---|---|
| `<repo>` | `/workspace/holohub` | source; **Python edits take effect with no rebuild** |
| `/tmp/tcn` | `/srv/tmp` | gate configs, traces, mask dumps — the shared scratch space |
| `/data/models_trt11` | `/srv/models` | engines and weights |

The shared scratch directory is what makes profiling work in both directions: the app writes a trace
inside the container, and it appears on the host at the mapped path (and vice versa for gate configs).
Knowing this mapping saved copying a 200 MB trace back and forth.

C++ changes need a build, but an incremental one:

```bash
docker exec <container> bash -lc 'cd /workspace/holohub/build/<app> && ninja <target>'
```

### 2. Run inside the user's existing container, not a fresh one

```bash
docker exec <container> bash -lc '
  B=/workspace/holohub/build/<app>
  SRC=/workspace/holohub/applications/.../python
  cd $B && PYTHONPATH=$B/python/lib:/workspace/holohub:$SRC:$PYTHONPATH \
    timeout 700 python3 $SRC/<app>.py -c /srv/tmp/gates/<gate>.yaml'
```

Using the container that is already up matters for three reasons:

- **It is the environment that will run in production.** Engines already built, models already
  downloaded, drivers already matched. A fresh container tests a different thing.
- **It respects the human's environment.** Nothing is started, stopped or rebuilt without asking; the
  loop is read-mostly with respect to the container's state.
- **Start-up cost disappears.** Model loading and engine deserialisation dominate a cold start;
  reusing a warm container turns a 3-minute check into a 40-second one, which is what makes iterating
  tolerable.

Always bound the run with `timeout`. A Holoscan graph that starves waits forever, and an unbounded
hang is indistinguishable from slow work.

### 3. Long or expensive operations go behind one approved script

Container image builds, engine exports and anything taking tens of minutes should be a single script
the human approves **once**, rather than a sequence of individually-approved commands:

```bash
./docs/trt11_build_test.sh --stage sdk       # one approval, many steps
./docs/trt11_build_test.sh --stage verify
```

Add a post-condition assertion inside such a script. A build that silently ignored a `--build-arg`
produced a working-looking image with the wrong TensorRT version; the assertion is what turns that
into an error instead of a mystery three steps later.

## The controlled input

A gate needs an input that does not change between runs. The pattern used here is a **replay operator
that is a drop-in replacement for the live source**:

- `TcnDatasetReplayerOp` emits on the same two ports as `TcnShmSubscriberOp`
  (`color_outputs`, `depth_outputs`), with the same tensor naming and the same channel order, so
  swapping it in is a one-line change at the `compose()` call site and nothing downstream can tell.
- It preloads a fixed on-disk export into pinned host memory at `start()`, so no decoding happens on
  the hot path and successive runs are identical.
- `frame_count` bounds the run via a `CountCondition`, so a gate run **terminates on its own** instead
  of needing to be killed — essential when the loop is automated.
- It forwards acquisition timestamps from the export, so anything downstream that depends on frame
  identity behaves as it does live.

Two traps this specific design had to solve, both worth copying:

- **Faithfulness beats convenience.** The reader decodes RGB, but the live subscriber delivers BGR and
  every consumer swaps channels itself. The replayer therefore reverses channels once at preload and
  emits BGR. A replayer that emitted "correct" RGB would make every downstream consumer wrong — and
  the images would still look plausible.
- **Looping needs advancing timestamps.** Replaying real capture times verbatim means pass 2 repeats
  timestamps already seen, and any consumer keyed on frame identity correctly rejects them. Each pass
  is shifted forward by one dataset span.

## The verifiable outcome

**This is the part that does the work.** A gate is a single line of output that a script can grep and
a human can read, which distinguishes three states: passed, failed, and *did not run*.

Good verdicts from this codebase:

```
GroupCheckOp: PASS -- 8 groups, 0 timestamp-inconsistent
camera01: PASS -- every selected pixel kept its depth
5/5 cases passed
CloudFusionCheckOp: frame 1 fused points per class: class_1=210877 class_2=4 ...
```

Properties that make them useful:

- **They are printed by the code, not inferred from logs.** A summary at `stop()` rather than a
  per-frame line, because per-frame logging gets rate-limited and then "no errors" and "never ran"
  look identical. `GroupCheckOp` explicitly emits `NO GROUPS` for the third state.
- **They assert an invariant, not an appearance.** "Every selected pixel kept its depth" is a property
  derivable from the geometry: a pixel can only be selected if its depth was valid, so masking must
  keep it. That single assertion found a use-after-free that looked, on screen, like a correct point
  cloud.
- **Some are exact numbers, used as byte-comparison.** `class_1=210877` is not checked against a
  threshold; it is compared to the previous run. After a pure refactor it must be *identical*, and
  that is how a 3,500-line code move was verified in one line.
- **Failures name the cause, not the symptom.** `input entity missing tensor 'camera01_depthimage'`
  points at the splitter; the fix was to make the code refuse at compose time with the camera name and
  the list of channels the source actually provides.

Read only the verdict. Running the full app and grepping is what keeps the loop fast:

```bash
... | grep -E "PASS|FAIL|Traceback|error\]|NO GROUPS"
```

## Tier the gates by cost

Push every check to the cheapest tier that can catch the bug it is guarding.

| tier | what | cost | run it |
|---|---|---|---|
| **0** | pure logic, no container: `python3 tests/test_planning.py`, `g++ -std=c++17 tests/test_sync_logic.cpp && ./t` | milliseconds | every edit |
| **1** | one operator in a tiny Holoscan app in the container: `PYTHONPATH=<build>/python/lib python3 tests/test_label_sampler.py` | seconds | every edit to that operator |
| **2** | whole app on the replay dataset, verdict-grepped | ~1 minute | before every commit |
| **3** | live stream + `nsys` trace, run by the human | minutes + attention | performance claims, and final acceptance |

Tier 0 is worth engineering for. The ring buffer, the frame matcher, the frame-plan arithmetic and the
calibration field mapping are all pure functions in files that import neither holoscan nor cupy,
precisely so their tests run on the host in milliseconds. 62 checks on the synchroniser's logic ran
before the operator existed.

Tier 1 needs the operator's Python module built, and catches everything about the operator's own
contract: dtypes, shapes, empty inputs, parameter validation.

Only tier 3 can answer "is it faster in production", and it is the one tier the human runs. Which
means: leave the pipeline in a state where they can run it, and tell them exactly what to look for.

## Discipline for performance work

Correctness gates and performance gates fail differently, and performance needs its own rules:

- **Measure in isolation *and* end to end**, because they disagree. One optimisation cut an operator's
  median by 57% in a micro-benchmark and moved end-to-end throughput by nothing, because the operator
  was never on the critical path.
- **A duration measured inside a blocking call is queue depth, not the callee's cost.** "32% of its
  time is in `cudaStreamSynchronize`" says the GPU was busy with someone else's work.
- **Gate optimisation work on whether the code is on the critical path**, never on how much time it
  appears to spend.
- **Benchmark on an idle device.** A background process at 12% utilisation swamped a 15% difference;
  the A/B only became conclusive after moving to the unused GPU.
- **A/B by alternating builds**, comparing medians of medians across several runs, and only claiming a
  difference when the run-to-run ranges do not overlap.

## When a step fails, verify the state before continuing

A multi-step edit script that dies halfway leaves an inconsistent tree. This happened here: a
refactoring script failed on a text match *after* a second script had already rewritten the imports
that depended on it, leaving a module importing names it still defined. Nothing detected it except
looking.

After any partial failure, inspect rather than assume:

```bash
ls <expected new file>            # did the write happen at all?
git diff --stat <touched file>    # what actually changed?
python3 -c "import ast; ast.parse(open('<file>').read())"
```

## The loop, in short

1. Write the gate first, and make it fail for the right reason. A gate that cannot distinguish "passed"
   from "did not run" is not a gate.
2. Edit on the host.
3. Build only if C++ changed (`ninja <target>`).
4. Run in the existing container, bounded by `timeout`, at the cheapest tier that can catch the bug.
5. Grep for the verdict. Read only that.
6. If red: change one thing. If the same failure recurs three times, the model of the problem is
   wrong — go and measure instead of guessing.
7. If green: run the next tier up. Commit when tier 2 is green.
8. Hand tier 3 to the human with a specific thing to look at.

## What this loop actually caught

Each of these was invisible to code reading and to watching the output:

| defect | what surfaced it |
|---|---|
| `wrapMemory` without a keep-alive in `tcn_stream_splitter` — a use-after-free corrupting one camera on ~1 frame in 8 | the mask-implies-depth invariant; the point cloud looked correct |
| `emit_depth` hardcoded false, so the sync gate was grouping *empty* payloads | the splitter refusing loudly instead of forwarding nothing |
| A `Timestamp` component appearing as a non-tensor key in every consumer's tensor map | instrumenting the failing operator to print the key and type, then re-running |
| Allocations landing on the wrong CUDA device when `cuda_device_ordinal != 0` | running a benchmark on a non-zero ordinal, which nothing else did |
| A "-60%" optimisation that improved throughput by zero | re-tracing end to end instead of trusting the isolated measurement |
| A backend key silently restricting a dynamic-vocabulary path to its baked vocabulary | testing with a term that had *never* been baked, rather than one that happened to be a subset |

The pattern in every row is the same: the loop turned a silent, plausible-looking wrong result into a
line of text that said so.
