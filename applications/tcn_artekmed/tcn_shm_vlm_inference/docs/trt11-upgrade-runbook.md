# Rebuilding everything for TensorRT 11 — runbook

Why: TensorRT 10.9 depresses Grounding DINO confidence scores (0.487 against PyTorch's 0.871,
|d| = 0.384); TensorRT 11.2 reproduces PyTorch almost exactly (|d| = 0.001, top-box IoU 0.9999).
See [`specs/2026-08-10-tensorrt-upgrade-assessment.md`](./specs/2026-08-10-tensorrt-upgrade-assessment.md)
for the measurement and [`specs/2026-08-10-sam-batched-decode-design.md`](./specs/2026-08-10-sam-batched-decode-design.md)
for why it also explains the GDINO FP16 rejection.

**Everything TensorRT touches must be rebuilt, in this order.** A serialized engine only loads in
the TensorRT that built it; mixing produces

```
Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed.
Version tag does not match. Note: Current Version: 243, Serialized Engine Version: 239
```

`243` is TRT 11.2, `239` is TRT 10.9. Seeing 243-vs-239 means a **TRT 11 runtime met a TRT 10
engine** — nearly always a model-tree mount pointing at the old tree (step 4).

---

## What has to be rebuilt, and what does not

| artifact | action | why |
|---|---|---|
| Holoscan SDK image | **rebuild** | `libholoscan_infer` links TensorRT; it is the only library in the image that does |
| HoloHub app container | **rebuild** | must sit on the new SDK base |
| GDINO engines (`_b{N}_*.engine`) | **rebuild** | version-locked |
| SAM encoder engines | **rebuild** | version-locked |
| GDINO ONNX, prompts npz, parity ref | **reuse** | precision- and version-neutral |
| SAM configs + `.pt` checkpoints | **reuse** | inputs to the export, not artifacts |
| **DA2 / DA3 engines** | **nothing** | `da3_inference.is_engine_path: false`, so holoinfer builds from ONNX and caches per version (`...trt.10.3.0.26.engine.fp32` → a new `...trt.11.2.1.2...` appears). First run is slow; old caches are inert |

---

## Step 1 — Holoscan SDK image with TensorRT 11

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/docs
./trt11_build_test.sh --stage preflight   # cheap; verifies the patches apply
./trt11_build_test.sh --stage sdk         # long
```

This applies two patches to the SDK checkout, builds, and **reverts them on exit** — including on
error or Ctrl-C. The checkout is never left modified. The patches live in `patches/`:

- `holoscan-sdk-4.4.0-trt-major-param.patch` — the SDK's `tensorrt-dev` stage hardcodes
  `libnvinfer10` / `libnvinfer-plugin10` / `libnvonnxparsers10`. Derives the major from the existing
  `TENSORRT_CU*_VERSION` arg instead. Backward compatible: the stock `10.3` pin still resolves
  `libnvinfer10`.
- `holoscan-sdk-4.4.0-holoinfer-trt11.patch` — `BuilderFlag::kFP16` and
  `kPREFER_PRECISION_CONSTRAINTS` were deprecated in TRT 10.12 and **removed** in TRT 11 (networks
  are strongly typed; precision comes from the model). Guarded with `#if NV_TENSORRT_MAJOR < 11`,
  matching the guard that same function already uses for a 10.7 deprecation. **This is the entire
  source-level incompatibility** — two enum constants in one file.

The version is injected by rewriting the `ARG TENSORRT_CU12_VERSION` default, **not** via
`--build-arg`: the SDK's `run` script forwards trailing arguments to an internal `build` that hands
them to `docker run`, where `--build-arg` is rejected — and on `build_image` it was accepted and
silently ignored, producing a confident-looking TRT 10.3 image. Watch for this line:

```
libnvinfer11 11.2.1.2-1+cuda12.9
```

Packaging is done by `patches/Dockerfile.holoscan-trt11`, not by the SDK, because
`./run build_run_image` is **broken upstream at v4.4.0**: it passes
`-f ${TOP}/runtime_docker/Dockerfile`, but `runtime_docker/` was deleted in the v4.0.0 release
commit. Unrelated to TensorRT; compile and install are fine.

Result: **`holoscan-trt11:4.4.0-cu12`**.

CUDA 12 is deliberate — TRT 11.2.1.2 ships both `+cuda12.9` and `+cuda13.3`, so the CUDA 13 move
the SDK's cu12 comment implies is not required. Override with `CUDA_MAJOR=13` if ever needed.

## Step 2 — rebuild the app containers on that base

```bash
./holohub run --base-img holoscan-trt11:4.4.0-cu12 ... tcn_shm_vlm_inference
```

Add `--base-img holoscan-trt11:4.4.0-cu12` to each `run_*.sh` you use. Do this rather than
hand-rolling `docker run`: the launcher already supplies X11 (`DISPLAY`, the X socket, xauth) and
`--runtime=nvidia`, and without them the app fails with `Failed to initialize glfw` and then
`Failed to create the Vulkan instance` — only the nvidia runtime injects
`/etc/vulkan/icd.d/nvidia_icd.json`, so `--gpus all` is not sufficient.

> The TRT 11 base is derived from a HoloHub image, so HoloHub's Dockerfile runs a second time over
> its own output. Two steps there were not idempotent and are now fixed (a misspelled
> `depencencies` path, and slang's `unzip`/`ln -s`). If more stale-state failures appear, the durable
> fix is to base the overlay on the SDK **builder** image (`holoscan-sdk-build-cu12-x86_64`, which
> already carries TRT 11 and has never seen HoloHub) instead.

## Step 3 — rebuild the GDINO and SAM engines

```bash
ENGINE_BATCHES="2 3" ./trt11_build_test.sh --stage engines
```

**Always name the batches explicitly.** Without `ENGINE_BATCHES`, the stage uses `--from-config` and
builds only what the *currently active* `gpu_workers` profile needs. That is a trap whenever more
than one camera topology is in use: building with the 4-camera profile active yields batch 2 only,
and switching to live then fails at startup with

```
FileNotFoundError: GDINO engine for batch 3 not found: .../gdino_swint_512x672_b3_tf32.engine
This worker owns 3 cameras, so it needs a batch-3 engine.
```

Engine batch **is** the per-worker camera count, because the batch is baked at ONNX trace time:

| topology | worker split | batches needed |
|---|---|---|
| 4-camera | 2 + 2 | 2 |
| 5-camera | 2 + 3 | 2 **and** 3 |

`ENGINE_BATCHES="2 3"` covers both, so either topology runs without a rebuild. Both rigs are
supported deployments, not test-vs-production.

Builds into `/data/models_trt11` (host) — **never over `/data/models`**, so the TRT 10.9 setup stays
loadable while the new one is unproven.

Expect, per engine:

```
image-independence gate OK
slice-consistency gate OK (batch 3: worst IoU 1.000000, worst |score delta| 0.000000)
pytorch fidelity [OK]: pytorch score=0.871 vs trt 0.872 (|d|=0.001) | top-box IoU=0.9999
```

That `[OK]` with `|d|` near zero is the whole point of the upgrade. Under TRT 10.9 the same line read
`[DEVIATION] ... trt 0.487 (|d|=0.384) | top-box IoU=0.9721`.

Worth noting: batch 3 is slice-exact under TRT 11 (`worst IoU 1.000000`). Under TRT 10.9 an FP16
batch-3 build **failed** that gate at 0.998016, which was the reason b3 stayed TF32 then.

The GDINO ONNX is precision- and version-neutral and is reused, so no host-side re-export is needed
for either batch. Confirm the inventory before switching topology:

```bash
ls /data/models_trt11/active/groundingdino/*.engine /data/models_trt11/active/sam2/*.engine
```

Expect four files: GDINO `b2` + `b3`, SAM encoder `b2` + `b3`.

## Step 4 — point the model mount at the new tree

**This is the step whose omission produces the 243-vs-239 error.** In your `run_*.sh`, one line:

```diff
- --mount type=bind,src=/data/models,dst=/srv/models
+ --mount type=bind,src=/data/models_trt11,dst=/srv/models
```

That is the *only* mount change needed. `/data/models_trt11` is **self-contained**: the rebuilt
engines are real files, and everything else (SAM configs, the `.pt` checkpoints, the DA and dinov3
trees) is **hardlinked** from `/data/models`. Both live on the same filesystem, so the hardlinks cost
no additional disk — `du` reports ~13 GB for the tree but the blocks are shared, and `stat` shows
`links=2`.

Hardlinks rather than symlinks on purpose: absolute symlinks pointing into `/data/models` dangle
inside the container unless that path is *also* mounted, which cost two failed startups
(`No such file or directory: .../sam2.1_hiera_t.yaml`) before being made self-contained. If you
rebuild this tree from scratch, use `cp -al` for the shared trees, not `ln -s`.

Tree layout:

```
/data/models_trt11/active/
├── groundingdino/         TRT 11 engines (real) + reused ONNX/npz
├── sam2/                  TRT 11 engine (real), configs/ copied, *.pt hardlinked
├── depth_anything_v2/     hardlinked from /data/models
├── depth_anything_v3/     hardlinked
├── dinov3/                hardlinked
└── sam2_trt_inference/    hardlinked
```

Verify before running the app — this catches the whole class of problem in one command:

```bash
docker run --rm -v /data/models_trt11:/srv/models --entrypoint bash holoscan-trt11:4.4.0-cu12 -c '
for p in active/groundingdino/gdino_swint_512x672_b2_tf32.engine \
         active/sam2/sam2.1_hiera_tiny_encoder_b2_fp16.engine \
         active/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
         active/sam2/sam2.1_hiera_tiny.pt ; do
  [ -e "/srv/models/$p" ] && echo "OK   $p" || echo "MISS $p"; done'
```

## Step 5 — verify

```bash
./trt11_build_test.sh --stage verify
```

Expect all four:

```
python tensorrt : 11.2.1.2
libnvinfer      : ... libnvinfer.so.10 ... libnvinfer.so.11 ...     (coexisting, expected)
holoinfer links : libnvinfer_plugin.so.11
InferenceOp import OK
```

`holoinfer links` showing `.so.10` means the SDK was not actually rebuilt and the image is stale.
Both sonames being present is correct — TRT 10 is left installed because its soname differs and
removing it risks transitive breakage for no gain.

## Step 6 — re-gate correctness

```bash
./trt11_build_test.sh --stage gate     # or a normal dataset run with mask_dump_dir set
```

**Read this on per-class instance counts, not on IoU against the old dumps.** TRT 10.9 → 11.2 is the
fix landing, so masks are *expected* to differ; a low `--iou-gate` score against pre-upgrade output
means it worked. Concluding "regression" from that number inverts the result.

The specific question: the pre-upgrade FP16 A/B found 36 instances present in only one engine's
output (median 15,290 px, max 78,474 px — whole people flickering). If those stabilise, the FP16
rejection was a symptom of the score depression rather than a property of FP16, and
[`specs/2026-08-10-trt-cuda-graphs-design.md`](./specs/2026-08-10-trt-cuda-graphs-design.md)-era
perf work can revisit it.

## Rollback

Nothing destructive was done: `/data/models` is untouched and the old images still exist.

1. Drop `--base-img` from the run scripts.
2. Restore the `src=/data/models` mount.
3. `cd <holoscan-sdk> && git status` — should be clean; the script reverts its own patches. If a run
   was killed mid-build: `git checkout -- Dockerfile modules/holoinfer`.

## Gotchas hit while establishing this

Recorded so they are not rediscovered:

| symptom | cause |
|---|---|
| `unknown flag: --build-arg`, or a build that silently produces TRT 10.3 | the SDK `run` CLI does not forward `--build-arg`; rewrite the ARG default instead |
| SDK builder image tagged with **holohub's** git sha | `./run` derives tags from `git rev-parse` in the *current* directory; it must be invoked with cwd inside the SDK checkout |
| `lstat .../runtime_docker: no such file` | `build_run_image` is broken upstream at v4.4.0 |
| `chmod: Operation not permitted` in HoloHub's build | the base image must end as `USER root`; HoloHub drops privileges itself at the end |
| `No module named 'tensorrt'` after a pip install that reported success | `pip install tensorrt==11.2.1.2` resolves to `tensorrt_cu13` in a CUDA 12 image, uninstalling the working one. Pin `tensorrt-cu12` |
| apt: `held broken packages` on `libnvinfer-dev` | `libnvinfer-safe-headers-dev` resolves to `+cuda13.3`; pin it explicitly |
| `fatal: destination path 'pybind11' already exists` | non-idempotent step in HoloHub's Dockerfile (fixed) |
| slang `unzip` prompting, then `ln -s` failing | same class (fixed) |
| 243-vs-239 serialization error | step 4 — the model mount still points at `/data/models` |
| `GDINO engine for batch 3 not found` after switching to live | step 3 built only the active profile's batches; rebuild with `ENGINE_BATCHES="2 3"` |
| `No such file: .../sam2.1_hiera_t.yaml` under the new tree | absolute symlinks into `/data/models` dangling in the container; the tree is now hardlinked and self-contained (step 4) |
