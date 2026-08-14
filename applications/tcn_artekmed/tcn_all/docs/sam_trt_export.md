# SAM 2 image encoder → ONNX → TensorRT export

Builds the fixed-batch FP16 TensorRT engine for the SAM 2 Hiera image encoder, for a future
`sam_backend: "trt"` path alongside the existing PyTorch `SAM._set_image_batch_gpu` (see
`../../../../operators/tcn_artekmed/tcn_langsam/models.py`). Companion tool: [`sam_trt_export.py`](sam_trt_export.py).

## Why this exists

SAM 2's image encoder is a per-frame cost that scales with the number of cameras in a LangSAM
worker's batch. A TensorRT engine collapses the per-layer PyTorch launches into a single
enqueue, the same win the GDINO TRT tool already banks for the detector.

## One stage, one environment

Unlike the GDINO tool, this one does **not** split into a host export stage and a container
build stage. It runs entirely **inside the holoscan runtime container**:

- the `sam2` package and the checkpoints under `/srv/models/active/sam2/` are already there
  (no fork, no separate venv, no checkpoint download step), and
- a serialized TensorRT engine is version-locked to the TensorRT that built it, so building
  where the engine will actually load sidesteps the whole host/container version-mismatch
  problem that forces the GDINO tool into two stages.

There is nothing to run on the host for this tool.

## Usage

```bash
# Inside the already-running container:
./holohub run --local --cuda 13 tcn_all
python3 /workspace/holohub/applications/tcn_artekmed/tcn_all/docs/sam_trt_export.py \
  --sam-type sam2.1_hiera_tiny \
  --batch 3 \
  --out /srv/models/active/sam2 \
  --image /path/to/a/frame/with/a/real/object.jpg   # optional, see "the mask gate" below
```

Single command, no separate export/build split — the ONNX is exported and the engine built
and gated in one process, and only the validated engine artifact is left behind.

### `--batch`

`--batch N` is the engine's batch, baked at ONNX trace time and pinned as `min = opt = max`
in the TensorRT optimization profile (same lesson as the GDINO tool: a `--batch` that tries to
stay dynamic across a profile range does not reliably work for this kind of fused network, so
the design does not rely on it). Set `N` to the **largest LangSAM worker's camera count** —
workers with fewer cameras pad their batch up to `N` rather than needing their own engine.
Changing camera count means re-running this tool at the new `N`; the artifact name carries it
(`<sam_type>_encoder_b<N>_<fp16|tf32>.engine`) so a mismatched engine cannot silently get used
in its place.

### `--from-config`

Instead of `--batch N`, pass `--from-config /path/to/tcn_all.yaml` to build one
engine per **distinct worker camera count** in the app's `gpu_workers` node (mutually exclusive
with `--batch`):

```bash
python3 /workspace/holohub/applications/tcn_artekmed/tcn_all/docs/sam_trt_export.py \
  --sam-type sam2.1_hiera_tiny \
  --from-config /workspace/holohub/applications/tcn_artekmed/tcn_all/python/tcn_all.yaml \
  --out /srv/models/active/sam2
```

With a `gpu_workers.workers` list of `[{cameras: [a, b]}, {cameras: [c, d, e]}]` this builds
`..._encoder_b2_fp16.engine` and `..._encoder_b3_fp16.engine` in one run — reading the *same*
`gpu_workers` node the application reads at startup is the point: the engines that exist cannot
drift from the GPU split that actually runs. Model construction and the encoder wrapper are done
once and reused across batches; only the test image, the preprocessed input and both gate
references are recomputed per batch, since those are batch-shaped. `--batch N` still works
unchanged for building a single engine by hand.

### Precision

The engine is built **FP16** by default (`--tf32` switches to TF32 instead). FP16 is a
deliberate match to the existing PyTorch path: `SAM._autocast()` already runs the encoder and
decoder under bf16 autocast in production, so the TRT engine is not introducing a new
lower-precision regime — it is replacing one narrow (bf16) float format with another (FP16) of
comparable width, which is exactly what the feature fidelity gate below checks for (cosine
similarity / relative error, not bit-exactness, since FP16 and bf16 legitimately differ in
their last bits).

### Gotcha: `dynamo=False` is required

`torch.onnx.export` is called with **`dynamo=False`**, the same as `gdino_trt_export.py`. Recent
PyTorch defaults to the dynamo exporter, which for this model does three unhelpful things:

1. requires an extra `onnxscript` dependency that the container does not ship;
2. silently raises the opset to 18, then fails converting back down —
   `RuntimeError: No Adapter To Version $17 for Resize`;
3. hands the graph to onnxscript's constant folder, which evaluates the Hiera `Resize` nodes
   through ONNX's **pure-Python** reference implementation (`onnx/reference/ops/op_resize.py`).
   On 1024² feature maps that never completes — it presents as a hang at ~112% CPU with no
   further output. (Diagnosed with `py-spy dump`; see `../optimization-playbook.md` §7.6.)

The legacy tracer does none of this. If you ever see the export sitting at full CPU with no
`ONNX written:` line, check that `dynamo=False` is still there before assuming the model is at
fault. `onnxscript` is **not** needed.

## The three gates

All three gates run against the batch produced by this same invocation, and **all three always
run and report** before anything aborts — the mask IoU is the decisive one and must not be
hidden by a proxy gate failing first. If any gate failed, the engine is not moved into place
and the run exits with the collected reasons.

1. **Feature fidelity** (`feature_gate`). Compares the TRT engine's three output tensors
   (`high_res_feats_0`, `high_res_feats_1`, `image_embed`) against an **FP32** reference, and
   requires the engine to be no further from it than the **bf16** autocast path already running
   in production, times a `margin` of 1.5.

   > The first version of this gate compared the engine against the bf16 reference with fixed
   > thresholds (cosine `>= 0.999`, relative L2 `<= 0.05`). That was wrong twice over. bf16 has
   > **fewer** mantissa bits (8) than the engine's FP16 (10), so the measurement conflated the
   > engine's error with bf16's own — a few percent disagreement between two different narrow
   > float formats is expected, not evidence of a bad engine. And the thresholds were arbitrary
   > constants rather than anything derived from the system. Judging the engine against the
   > precision regime you already ship is a criterion with meaning: *not a precision regression
   > relative to what you already accept*. It also answers FP16-vs-TF32 directly — if the TRT
   > error lands far outside bf16's, FP16 is genuinely too coarse and `--tf32` is the answer.

2. **Slice consistency** (`slice_gate`). At `--batch N >= 2`, every image in the batch is the
   same test image, so every slice of every output tensor must be identical to slice 0
   (max abs diff `<= 1e-4`). Proves the engine's batching is correct — a broken batch dimension
   would otherwise silently corrupt per-camera output for slices `1..N-1`, indistinguishable
   from ordinary per-camera detection noise. Skipped (prints and returns) at `--batch 1`.

3. **Mask IoU** (`mask_gate`), the one that matters end-to-end. Runs the *unchanged* PyTorch
   `SAM2ImagePredictor` decoder twice with the same box prompt, once fed the PyTorch reference
   features and once fed the TRT engine's features, and requires the resulting binary masks to
   agree (IoU `>= 0.99` by default, `--min-iou` to change it). This is the gate that proves the
   encoder swap does not change what a user actually sees. It also refuses to run on a
   reference mask that covers `<1%` or `>90%` of the frame (`MASK GATE UNUSABLE`) — an
   uninformative mask makes the IoU meaningless, which is why `--image` should point at a real
   frame with an object inside the fixed evaluation box; the default synthetic image is a
   structured two-rectangle test pattern (not noise, which yields degenerate empty/full masks)
   used when no `--image` is given.

## Artifacts move into place only after every gate passes

The ONNX export and the TensorRT build both happen inside a `build-<tag>-XXXX/` directory under
`--out`. Only after the feature, slice, and mask gates all pass does the tool `os.replace()` the
temp engine onto the final path (`<out>/<sam_type>_encoder_b<N>_<fp16|tf32>.engine`) and delete
the directory. A failed run therefore can never leave a half-validated or corrupt engine sitting
at the path the app would load — the failure mode the GDINO builder had before this design was
adopted for it too.

**On failure the directory is kept and its path printed.** An earlier version deleted it, which
destroyed the ONNX — precisely the artifact needed to diagnose why the run failed. Not installing
an unvalidated engine is the requirement; discarding the evidence was never part of it. Delete
those `build-*` directories yourself once you are done with them.
