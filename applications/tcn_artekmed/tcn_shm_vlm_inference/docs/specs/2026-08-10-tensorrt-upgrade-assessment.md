# TensorRT upgrade: the GDINO score depression is a TRT 10.9 defect

## The finding

The Grounding DINO confidence-score depression we have carried as an open issue since the batched
inference work is **a TensorRT 10.9 defect that TensorRT 11.2 fixes exactly.**

Same ONNX (`gdino_swint_512x672_b2_tf32.onnx`), same TF32 build settings, same
`gdino_swint_512x672_parity_ref.npz` PyTorch reference (top score 0.8711732). Only the TRT version
differs:

| TensorRT | top score vs PyTorch 0.871 | top-box IoU | gate verdict |
|---|---|---|---|
| **10.9.0.34** (container) | 0.487 — **\|d\| = 0.384** | 0.9721 | `[DEVIATION]` |
| **11.2.1.2** (host venv) | **0.871 — \|d\| = 0.000** | **1.0000** | `[OK]` |

Not "improved" — exact agreement with PyTorch on the reference image and a perfect box match.

Reproduce (host, no container, nothing installed):

```bash
S=/tmp/trt11 && mkdir -p $S
G=/data/models/active/groundingdino
ln -s $G/gdino_swint_512x672_b2_tf32.onnx $G/gdino_swint_512x672_parity_ref.npz \
      $G/gdino_swint_prompts.npz $S/
cd /home/ecku/develop/vision/GroundingDINO-TensorRT-and-ONNX-Inference   # for images/in/person.jpg
/home/ecku/develop/vision/GroundingDINO/.venv-gdino-export/bin/python \
  <docs>/gdino_trt_export.py --stage build --batch 2 --hw 512 672 --out $S
```

`--out` MUST be a scratch directory: an engine built by TRT 11 cannot be deserialized by the
container's TRT 10.9, so installing it into `/data/models/active/groundingdino` would break the app.

## Why this matters more than a fidelity number

1. **It is the root cause of A1's (GDINO FP16) failure.** The 2026-08-10 harness A/B rejected FP16
   because 36 instances existed in only one engine's output — whole people appearing and vanishing,
   16 of them larger than 0.5% of the frame. That was never FP16 inventing fragility: scores
   depressed by 0.384 sit right on `box_threshold`, so *any* numerical perturbation flips detections.
   On a correct baseline, FP16 deserves re-testing rather than rejection.
2. **It retires the long-standing open issue** recorded as "the likeliest cause of corner-case
   detection misses" (see [`../deferred-findings.md`](../deferred-findings.md)) — with a fix, not a
   workaround.
3. **TF32 alone suffices for the correctness win.** At TRT 11 the TF32 engine already matches
   PyTorch exactly, so fixing detection quality does not depend on FP16 at all.

## Limit of the evidence

One image, one top box. `|d| 0.384 → 0.000` is far stronger than a marginal shift, but end-to-end
mask equivalence over the dataset cannot be measured until an engine of that version can run inside
the container — which is what the upgrade is for. Treat the direction as established and the
magnitude across a scene as unmeasured.

## Feasibility of upgrading the image

### GXF is NOT the blocker

Scanning every library in `/opt/nvidia/holoscan/lib`, **exactly one** links TensorRT:

```
libholoscan_infer.so.4.4.0 -> libnvinfer_plugin.so.10
```

No GXF extension links `libnvinfer` at all. The closed-source GXF concern does not apply.

### holoinfer is the blocker, and it is on our critical path

`libholoscan_infer` is soname-bound to `libnvinfer_plugin.so.10`, and both `da2_fragment.py` and
`da3_fragment.py` construct `InferenceOp`, which is holoinfer. So bumping a version string alone is
**not sufficient**: the prebuilt `/opt/nvidia/holoscan` binaries would still demand `.so.10` while
the image ships `.so.11`, and the depth path would fail at library load.

Our GDINO and SAM paths need nothing — they call the Python `tensorrt` module directly and pick up
whatever is installed.

### What actually has to change

`holohub`'s own `Dockerfile` has no TRT knob; TRT arrives via
`BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}`. The knob is in
the **Holoscan SDK** `Dockerfile` (`/home/ecku/develop/holoscan/holoscan-sdk/Dockerfile`):

```
ARG TENSORRT_CU12_VERSION=10.3   # "last version that supports CUDA 12 on sbsa 22.04"
ARG TENSORRT_CU13_VERSION=10.16
```

Three things make this more than an ARG bump:

1. **The package names hardcode the major version.** The `tensorrt-dev` stage installs
   `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10`, and resolves the version with
   `apt-cache madison libnvinfer10 | grep ${TRT_VERSION}`. At TRT 11 these all become `...11`, so
   the major must be parameterised, not just the version string.
2. **The SDK is actively pinning TRT 11 away.** The stage's own comment says
   *"the pin blocks apt from selecting the newer TRT 11 candidate as the resolved dep"* — so TRT 11
   **is** available in the apt channel and is being deliberately held back.
3. **CUDA major likely has to move to 13.** The cu12 pin is capped at 10.3, and the host's TRT
   11.2.1.2 venv carries `torch 2.13.0+cu130`. Our runs currently use `./holohub run --cuda 12`, so
   this probably means `--cuda 13`. GXF already ships a cu13 build (`GXF_CU13_VERSION`), so that
   part is supported.

### Code risk in holoinfer: low

`modules/holoinfer/src/infer/trt/` uses only the modern explicit-tensor API — `enqueueV3` (x4) and
`setTensorAddress` (x6), plus `IBuilder` / `IBuilderConfig` / `IRuntime` / `ICudaEngine` /
`IExecutionContext` / `IOptimizationProfile` / `OptProfileSelector` / `TensorIOMode` /
`MemoryPoolType` / `BuilderFlag`.

Crucially **absent**: `enqueueV2`, `setBindingDimensions`, `getBindingIndex`, `getNbBindings` — the
deprecated binding API whose removal is the main 10→11 source breakage. No red flags, though the
plugin registry, `libnvinfer_plugin` initialisation and ONNX-parser behaviour are unverified.

## Consequences to plan for

- **Every existing engine must be rebuilt.** Engines are TRT-version-locked (this is what produced
  the original `stdVersionRead == kSERIALIZATION_VERSION` failure, serialized 243 vs current 239).
  That means GDINO b2/b3, the SAM encoders, and the DA2/DA3 engines.
- **DA2/DA3 need re-gating**, since holoinfer and their engines both change underneath them.
- **The harness comparison across the upgrade is not an equivalence check.** TRT 10.9 → 11.2 is the
  *fix* landing, so masks are expected to differ. The meaningful reading is per-class instance counts
  against what the scene actually contains, not IoU against the old (wrong) output. Do not gate this
  step with `--iou-gate` against pre-upgrade dumps and conclude a regression.

## Suggested order

1. Rebuild the SDK image with a parameterised TRT major at cu13 / TRT 11.x.
2. Rebuild GDINO + SAM + DA engines against the new TRT; confirm `fidelity [OK]` at build time.
3. Harness run, reading instance counts rather than IoU-vs-old (previous point).
4. Re-test A1 (FP16) on the corrected baseline — it may now pass.
5. Re-measure perf: the whole A1 rationale was −20 to −24 ms of GPU work from FP16, still unclaimed.

## Out of scope here

- The B2 panoptic CUDA kernel, which is independent of TRT and already verified byte-identical.
- CUDA graph capture (roadmap step 5), also TRT-version-independent in principle but worth
  re-checking after an upgrade, since capture support is a TRT behaviour.

---

## Addendum 2026-08-11: results, and why GDINO stayed TF32

### Measured outcome of the upgrade

| | period | fps | dev1 gdino GPU | dev1 sam GPU |
|---|---|---|---|---|
| TRT 10.9 | 171.0 ms | 5.85 | 62.00 ms | 52.02 ms |
| TRT 11, first run | 253.0 ms | 3.95 | 92.13 | 97.91 |
| TRT 11 + FP16 SAM + CUDA panoptic | **199.5 ms** | **5.01** | 93.84 | 67.09 |

Quality improved as intended — the score depression is gone (|d| 0.384 → 0.001) and masks are
visibly more stable. The residual ~17% throughput gap versus TRT 10.9 is **entirely GDINO**, whose
GPU time rose 62 → 94 ms/tick. It was TF32 before and after, so this is TensorRT 11 kernel
selection, with one fused node (`__myl_DivMulReshTranReshConcReshMoveMulSum`) at 30.94 ms/tick.

Two recoveries along the way, both committed:
- **SAM strongly-typed FP16** — 2.18× on the encoder (75.41 → 34.63 ms at batch 3). TRT 11 removed
  `BuilderFlag.FP16`, and the export tool's `hasattr` guard had been silently skipping it and
  building TF32 while still naming the file `_fp16`.
- **panoptic CUDA kernel** — −19 ms, 240 kernels → 3.

### GDINO FP16: four approaches, all blocked

Attempted 2026-08-11 and **reverted**. Recorded so it is not retried blindly. The obstacle is
structural: GDINO is **cross-modal**, so text and image tensors meet inside the fusion encoder.
There is no clean seam to cut along, and a strongly-typed TensorRT network refuses to auto-promote,
so every mixed edge is a hard parse error.

| approach | failure |
|---|---|
| `model.half()`, traced on CUDA | the vendored Swin builds its shift mask inline with `torch.zeros(...)` at fp32; adding it promotes `attn` while `v` stays half → `swin_transformer.py:171 ... expected scalar type Float but found Half`. Needs patching the fork |
| `torch.autocast(cuda, float16)` | `NameError: name '_C' is not defined` — GroundingDINO's deformable-attention CUDA extension is not built, and the traceable pure-PyTorch fallback is CPU-only. This is precisely why `--stage export` traces on CPU |
| graph FP16 (`onnxconverter-common`), `op_block_list` widened | `/bert/encoder/layer.0/attention/self/Div` — types Half and Float |
| graph FP16, whole BERT subgraph block-listed by node name (1122 nodes) | boundary moves into the fusion: `/transformer/encoder/Mul_9` — types Float and Half |
| graph FP16, whole graph, shape inference enabled | back to `/bert/Sub` — types Half and Float |

`onnxconverter-common` does not place casts correctly for this graph in any configuration tried.
Routes worth considering if it is revisited, cheapest first:

1. patch the vendored Swin to build its mask in the activation dtype, making `.half()` viable —
   but the CUDA-op problem still blocks tracing on GPU;
2. build GroundingDINO's `_C` extension so `autocast` on CUDA becomes tractable;
3. insert the casts by hand rather than relying on the converter.

### Better next lever

**CUDA graphs** (see [`2026-08-10-trt-cuda-graphs-design.md`](./2026-08-10-trt-cuda-graphs-design.md))
is precision-independent and targets a measured cost: GDINO issues ~1843 TensorRT kernel launches
per tick, ~94% of that stage's launches. Nothing about the TRT 11 move invalidates that design.

