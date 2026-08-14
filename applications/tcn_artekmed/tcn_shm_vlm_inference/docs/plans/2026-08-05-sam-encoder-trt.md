# SAM 2 encoder → TensorRT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the PyTorch SAM 2 Hiera image encoder with a prebuilt FP16 TensorRT engine, leaving the GPU resize/normalize and the PyTorch decode loop untouched.

**Architecture:** One in-container tool exports the encoder to ONNX, builds a fixed-batch FP16 engine, and gates it (feature fidelity, slice consistency, end-to-end mask IoU) before moving artifacts into place. At run time `SamTrtEncoder` replaces the `forward_image`/`_prepare_backbone_features` block inside `SAM._set_image_batch_gpu`.

**Tech Stack:** Python 3.12, TensorRT 10.9 (container), PyTorch, sam2, CuPy, numpy.

**Spec:** [`../specs/2026-08-05-sam-encoder-trt-design.md`](../specs/2026-08-05-sam-encoder-trt-design.md)

## Global Constraints

- Everything runs **in the container** (`/workspace/holohub`, models at `/srv/models`). Unlike the GDINO tool there is no host stage.
- Features and masks must **stay on the GPU**; the encoder returns torch CUDA tensors assigned straight into `p._features`.
- Host tests: no pytest, numpy-only, plain `__main__` PASS/FAIL runner, run from `applications/tcn_artekmed/tcn_shm_vlm_inference/python`.
- Commit style `feat(tcn_artekmed): ...` ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Never `git add -A`; stage only the files a task names.
- **Baselines to beat:** `sam` stage 93.3 ms (GPU 1, 3 cameras) / 72.4 ms (GPU 0). `torch.compile` reached 80.0 ms on GPU 1 — the engine must beat that to be worth keeping. Eager encode at batch 3 = 45.5 ms.
- `plan_batch_padding(n_frames, engine_batch)` already exists in `langsam_helpers.py` — reuse it.

---

### Task S1: The export + build tool

**Files:**
- Create: `../../../tcn_all/docs/sam_trt_export.py`
- Create: `../../../tcn_all/docs/sam_trt_export.md`

**Interfaces produced:** a CLI producing `<out>/<sam_type>_encoder_b<N>_<fp16|tf32>.engine`.

- [ ] **Step 1: Write the tool**

```python
#!/usr/bin/env python3
"""Export the SAM 2 image encoder -> ONNX -> fixed-batch TensorRT engine, gated.

Runs entirely INSIDE the holoscan runtime container: it needs the `sam2` package, the
checkpoint, and the same TensorRT that will load the engine (engines are version-locked, so
building where we run removes that whole problem -- unlike the GDINO tool, which needs a
separate host stage for its fork).

The encoder wrapper is derived from tier4's sam2_pytorch2onnx/export_sam2_onnx.py
(Apache-2.0) and verified against SAM._set_image_batch_gpu in ../python/langsam_common.py.
"""
from __future__ import annotations
import argparse, os, sys, tempfile

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from langsam_common import SAM, SAM_MODELS          # noqa: E402

IMAGE_SIZE = 1024
OUT_NAMES = ["high_res_feats_0", "high_res_feats_1", "image_embed"]


class SAM2EncoderWrapper(nn.Module):
    """SAM 2 image encoder exposing exactly the three tensors the predictor caches.

    Mirrors `SAM._set_image_batch_gpu`: forward_image (which applies conv_s0/conv_s1 when
    use_high_res_features_in_sam is set) -> _prepare_backbone_features -> += no_mem_embed ->
    reshape per level. Both model flags are asserted rather than assumed: a model configured
    otherwise would make the engine silently wrong.
    """

    def __init__(self, sam_model):
        super().__init__()
        if not getattr(sam_model, "use_high_res_features_in_sam", False):
            raise SystemExit("model has use_high_res_features_in_sam=False; this wrapper "
                             "applies conv_s0/conv_s1 unconditionally and would be wrong")
        if not getattr(sam_model, "directly_add_no_mem_embed", False):
            raise SystemExit("model has directly_add_no_mem_embed=False; this wrapper adds "
                             "no_mem_embed unconditionally and would be wrong")
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, input_image):
        backbone_out = self.image_encoder(input_image)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1])
        n = self.model.num_feature_levels
        feature_maps = backbone_out["backbone_fpn"][-n:]
        vision_pos = backbone_out["vision_pos_enc"][-n:]
        sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos]
        feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        feats[-1] = feats[-1] + self.no_mem_embed
        out = [f.permute(1, 2, 0).reshape(input_image.shape[0], -1, *s)
               for f, s in zip(feats[::-1], sizes[::-1])][::-1]
        return out[0], out[1], out[2]


def build_sam(sam_type, device):
    sam = SAM(sam_type, None, device=device, compile_model=False)
    sam.build_model()
    return sam


def export_onnx(wrapper, onnx_path, batch, device):
    dummy = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dyn = {"image": {0: "batch_size"}, **{n: {0: "batch_size"} for n in OUT_NAMES}}
    torch.onnx.export(wrapper, dummy, onnx_path, export_params=True, opset_version=17,
                      do_constant_folding=True, input_names=["image"],
                      output_names=OUT_NAMES, dynamic_axes=dyn)
    print(f"ONNX written: {onnx_path}")


def build_engine(onnx_path, engine_path, batch, fp16=True):
    """min = opt = max = batch. GDINO taught us the traced batch is the engine's batch for
    that model; SAM 2's Hiera may be genuinely dynamic, but the design does not rely on it."""
    lg = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(lg)
    net = b.create_network(0)
    p = trt.OnnxParser(net, lg)
    with open(onnx_path, "rb") as f:
        if not p.parse(f.read()):
            for i in range(p.num_errors):
                print("PARSE ERROR:", p.get_error(i))
            raise SystemExit("ONNX parse failed")
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif hasattr(trt.BuilderFlag, "TF32"):
        cfg.set_flag(trt.BuilderFlag.TF32)
    prof = b.create_optimization_profile()
    s = (int(batch), 3, IMAGE_SIZE, IMAGE_SIZE)
    prof.set_shape("image", s, s, s)
    cfg.add_optimization_profile(prof)
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise SystemExit("engine build returned None")
    data = bytes(memoryview(ser))
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"engine written: {engine_path} ({len(data)/2**20:.0f} MiB, batch {batch}, "
          f"{'fp16' if fp16 else 'tf32'})")
    return data


def run_engine(engine_bytes, img):
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    img = img.contiguous()
    ctx.set_input_shape("image", tuple(img.shape))
    ctx.set_tensor_address("image", img.data_ptr())
    outs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            outs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), device=img.device,
                                  dtype=torch.float32)
            ctx.set_tensor_address(n, outs[n].data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    return [outs[n] for n in OUT_NAMES]


def make_test_image(sam, batch, device, image_path=None):
    """A deterministic, STRUCTURED image (not noise) so the decoder produces non-degenerate
    masks; noise yields empty or full masks and makes the IoU gate meaningless."""
    if image_path:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise SystemExit(f"test image not found: {image_path}")
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).to(device)
    else:
        H = W = 720
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        rgb = ((yy * 255 // H).unsqueeze(-1).repeat(1, 1, 3)).to(torch.uint8).to(device)
        rgb[150:400, 150:450] = torch.tensor([220, 40, 40], dtype=torch.uint8, device=device)
        rgb[450:650, 300:650] = torch.tensor([40, 220, 90], dtype=torch.uint8, device=device)
    return [rgb] * int(batch)


def preprocess(sam, images_gpu):
    p = sam.predictor
    resize_norm = p._transforms.transforms
    return torch.stack([resize_norm(im.permute(2, 0, 1).to(torch.float32).div_(255.0))
                        for im in images_gpu], dim=0).to(sam.device)


def feature_gate(ref, trt_out, min_cos=0.999, max_rel=0.05):
    """BLOCKING. FP16 vs bf16 differ in the last bits; require direction and magnitude to
    agree, not bit-exactness."""
    ok = True
    for name, a, b in zip(OUT_NAMES, ref, trt_out):
        a32, b32 = a.float().flatten(), b.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()
        rel = ((a32 - b32).norm() / a32.norm().clamp_min(1e-12)).item()
        flag = "OK" if (cos >= min_cos and rel <= max_rel) else "FAIL"
        print(f"  feature {name:18} cos={cos:.6f} rel_err={rel:.4f}  [{flag}]")
        ok = ok and flag == "OK"
    if not ok:
        raise SystemExit("FEATURE FIDELITY GATE FAILED")
    print("feature fidelity gate OK")


def slice_gate(trt_out, batch):
    """BLOCKING. Identical inputs across the batch must give identical outputs."""
    if batch < 2:
        print("slice-consistency gate: batch 1, nothing to compare")
        return
    for name, t in zip(OUT_NAMES, trt_out):
        d = max((t[i] - t[0]).abs().max().item() for i in range(1, t.shape[0]))
        if d > 1e-4:
            raise SystemExit(f"SLICE CONSISTENCY GATE FAILED: {name} differs by {d:.3e} "
                             f"across slices; the engine baked its traced batch incorrectly.")
    print(f"slice-consistency gate OK (batch {batch})")


def mask_gate(sam, images_gpu, ref, trt_out, min_iou=0.99):
    """BLOCKING, and the one that matters: same decoder, same boxes, features swapped."""
    p = sam.predictor
    H, W = int(images_gpu[0].shape[0]), int(images_gpu[0].shape[1])
    box = np.array([[W * 0.2, H * 0.2, W * 0.65, H * 0.58]], dtype=np.float32)

    def masks_from(feats):
        p.reset_predictor()
        p._orig_hw = [(H, W) for _ in images_gpu]
        p._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        p._is_image_set = True
        p._is_batch = True
        mi, uc, lb, ub = p._prep_prompts(None, None, box, None, True, img_idx=0)
        m, _, _ = p._predict(uc, lb, ub, mi, multimask_output=False, return_logits=False,
                             img_idx=0)
        return (m[0, 0] > 0)

    a, b = masks_from(ref), masks_from(trt_out)
    cov = a.float().mean().item()
    if not 0.01 <= cov <= 0.90:
        raise SystemExit(f"MASK GATE UNUSABLE: reference mask covers {cov:.1%} of the image; "
                         f"pass --image with a frame where the box contains a real object.")
    iou = ((a & b).sum().float() / (a | b).sum().float().clamp_min(1)).item()
    print(f"mask IoU (pytorch vs trt features) = {iou:.4f}  [reference coverage {cov:.1%}]")
    if iou < min_iou:
        raise SystemExit(f"MASK GATE FAILED: IoU {iou:.4f} < {min_iou}")
    print("mask gate OK")


def main():
    ap = argparse.ArgumentParser(description="SAM 2 image encoder -> ONNX -> TRT (in-container)")
    ap.add_argument("--sam-type", default="sam2.1_hiera_tiny", choices=sorted(SAM_MODELS))
    ap.add_argument("--batch", type=int, default=3,
                    help="engine batch = the largest LangSAM worker's camera count; the trace "
                         "and the profile both use it, and the artifacts are named _b<N>_")
    ap.add_argument("--out", default="/srv/models/active/sam2")
    ap.add_argument("--image", default=None,
                    help="optional real frame for the mask gate; a synthetic structured image "
                         "is used otherwise")
    ap.add_argument("--tf32", action="store_true", help="build TF32 instead of FP16")
    ap.add_argument("--min-iou", type=float, default=0.99)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    tag = f"{args.sam_type}_encoder_b{int(args.batch)}_{'tf32' if args.tf32 else 'fp16'}"
    final_engine = os.path.join(args.out, tag + ".engine")
    os.makedirs(args.out, exist_ok=True)

    print(f"TensorRT {trt.__version__} | building {tag}")
    sam = build_sam(args.sam_type, dev)
    wrapper = SAM2EncoderWrapper(sam.model).eval().to(dev)

    images = make_test_image(sam, args.batch, dev, args.image)
    batch_in = preprocess(sam, images)
    with torch.no_grad(), sam._autocast():
        ref = list(wrapper(batch_in))

    # Build into a temp dir; move into place only after every gate passes, so a failed build
    # can never leave an unvalidated engine on disk (the GDINO builder's defect).
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        onnx_tmp = os.path.join(tmp, tag + ".onnx")
        eng_tmp = os.path.join(tmp, tag + ".engine")
        with torch.no_grad():
            export_onnx(wrapper, onnx_tmp, args.batch, dev)
        data = build_engine(onnx_tmp, eng_tmp, args.batch, fp16=not args.tf32)
        out = run_engine(data, batch_in)
        feature_gate(ref, out)
        slice_gate(out, int(args.batch))
        mask_gate(sam, images, ref, out, min_iou=args.min_iou)
        os.replace(eng_tmp, final_engine)
    print(f"\nDONE. Engine: {final_engine}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/docs && python3 -m py_compile sam_trt_export.py && echo "compile OK"
```
Expected: `compile OK`. (It cannot run outside the container — that is Task S3.)

- [ ] **Step 3: Write `sam_trt_export.md`**

Document: that it runs entirely in the container (and why — sam2 + checkpoints are present, and building where we run avoids the TRT version lock); the single command; the batch rule (engine batch = largest worker's camera count, workers pad); FP16 and why it matches the existing bf16 autocast; the three gates and what each proves; and that artifacts move into place only after gates pass.

- [ ] **Step 4: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.md
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): SAM 2 encoder -> ONNX -> fixed-batch FP16 TRT tool

Runs entirely in the container: sam2 and the checkpoints are already there, and
building where we run removes the TRT version-lock problem that forced the GDINO
tool into two stages.

Three blocking gates -- feature fidelity (cosine, not bit-exactness, since FP16 vs
bf16 differ in the last bits), slice consistency at batch N, and end-to-end mask IoU
through the unchanged PyTorch decoder. Artifacts are built in a temp dir and moved
into place only after all three pass.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task S2: Runtime `SamTrtEncoder` + wiring

**Files:**
- Modify: `python/langsam_common.py` (add `SamTrtEncoder`; branch in `SAM.__init__` and `_set_image_batch_gpu`)
- Modify: `python/langsam_multicam_fragment.py` (pass the config through)
- Modify: `python/tcn_shm_vlm_inference.yaml` (add `sam_backend`, `sam_trt_engine`)

**Interfaces consumed:** `plan_batch_padding` (already in `langsam_helpers.py`).
**Interfaces produced:** `SamTrtEncoder(engine_path, device)` with `.engine_batch`, `.encode(batch_gpu) -> [high_res_0, high_res_1, image_embed]`.

- [ ] **Step 1: Add `SamTrtEncoder` to `langsam_common.py`**

Place it immediately before `class GDinoTrtDetector`:

```python
class SamTrtEncoder:
    """SAM 2 Hiera image encoder as a prebuilt fixed-batch TensorRT engine.

    Replaces the forward_image / _prepare_backbone_features / no_mem_embed / reshape block of
    SAM._set_image_batch_gpu with one execute_async_v3. Inputs and outputs stay on the GPU, so
    the decode path is unchanged. The engine runs at exactly `engine_batch` images (baked at
    ONNX trace time); a worker with fewer cameras pads and the extra slices are discarded.
    """

    def __init__(self, engine_path, device):
        import tensorrt as trt
        self.trt = trt
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{int(device)}")
        self._engine_path = engine_path
        with torch.cuda.device(self.device):
            logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(
                    f"Failed to load SAM TRT engine: {engine_path}\n"
                    f"Runtime TensorRT is {trt.__version__}. Engines only load in the TRT that "
                    f"built them -- rebuild INSIDE this container:\n"
                    f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                    f"sam_trt_export.py --batch <N> --out {os.path.dirname(engine_path)}\n"
                    f"Or set langsam_inference.sam_backend: \"pytorch\" to fall back.")
            self.ctx = self.engine.create_execution_context()
            try:
                self.engine_batch = int(self.engine.get_tensor_profile_shape("image", 0)[0][0])
            except Exception as e:
                print(f"Failed to read SAM TRT engine profile (assuming batch=1): {e}")
                self.engine_batch = 1
            self._out_names = [self.engine.get_tensor_name(i)
                               for i in range(self.engine.num_io_tensors)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.OUTPUT]

    def batch_error(self, n):
        return (f"{n} images requested but the SAM encoder engine is built for batch "
                f"{self.engine_batch}. The batch is baked at ONNX trace time, so rebuild at "
                f"--batch {n}:\n"
                f"  python3 <holohub>/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/"
                f"sam_trt_export.py --batch {n} --out {os.path.dirname(self._engine_path)}")

    def encode(self, batch_gpu):
        """batch_gpu: (n,3,1024,1024) float32 CUDA, already resized+normalised.

        Returns [high_res_feats_0, high_res_feats_1, image_embed], each sliced back to n.
        One execution and one stream sync regardless of n.
        """
        n = int(batch_gpu.shape[0])
        try:
            pad = plan_batch_padding(n, self.engine_batch)
        except ValueError as e:
            raise ValueError(f"{e}\n{self.batch_error(n)}") from None
        with torch.cuda.device(self.device):
            img = batch_gpu
            if pad:
                img = torch.cat([img, torch.zeros((pad,) + tuple(img.shape[1:]),
                                                  device=img.device, dtype=img.dtype)], dim=0)
            img = img.contiguous()
            self.ctx.set_input_shape("image", tuple(img.shape))
            self.ctx.set_tensor_address("image", img.data_ptr())
            outs = {}
            for nm in self._out_names:
                outs[nm] = torch.empty(tuple(self.ctx.get_tensor_shape(nm)),
                                       device=self.device, dtype=torch.float32)
                self.ctx.set_tensor_address(nm, outs[nm].data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            return [outs["high_res_feats_0"][:n], outs["high_res_feats_1"][:n],
                    outs["image_embed"][:n]]
```

Add `plan_batch_padding` to the `from langsam_helpers import (` block if not already imported.

- [ ] **Step 2: Accept the config in `SAM.__init__`**

Add `sam_backend="pytorch"` and `sam_trt_engine=None` keyword arguments; store them, and set `self.trt_encoder = None`. At the end of `build_model()`, after the `compile_model` block:

```python
        if self.sam_backend == "trt":
            if not self.sam_trt_engine:
                raise ValueError('sam_backend: "trt" requires sam_trt_engine')
            self.trt_encoder = SamTrtEncoder(self.sam_trt_engine, self.device)
            print(f"SAM2 image encoder: TensorRT engine (batch {self.trt_encoder.engine_batch})")
```

- [ ] **Step 3: Branch in `_set_image_batch_gpu`**

Replace the block from `model = p.model` through the `feats = [...]` comprehension with:

```python
        model = p.model
        if self.trt_encoder is not None:
            feats = self.trt_encoder.encode(batch)
        else:
            backbone_out = model.forward_image(batch)
            _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
            if model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
            bsz = batch.shape[0]
            feats = [
                feat.permute(1, 2, 0).view(bsz, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], p._bb_feat_sizes[::-1])
            ][::-1]
```

The `p._features = {...}` assignment and everything after it are unchanged.

- [ ] **Step 4: Pass the config through**

In `langsam_multicam_fragment.py`, where `SAM(...)` is constructed, add:

```python
                sam_backend=langsam_cfg.get("sam_backend", "pytorch"),
                sam_trt_engine=langsam_cfg.get("sam_trt_engine"),
```

In `tcn_all.yaml`, under `langsam_inference`, after `sam_gpu_output`:

```yaml
  # SAM 2 image encoder backend: "pytorch" (eager) | "trt" (prebuilt engine from
  # docs/sam_trt_export.py). The _b<N>_ in the engine name is its FIXED batch, baked at ONNX
  # trace time; N must be >= every worker's camera count and smaller workers pad up to it.
  sam_backend: "pytorch"
  sam_trt_engine: "/srv/models/active/sam2/sam2.1_hiera_tiny_encoder_b3_fp16.engine"
```

- [ ] **Step 5: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_common.py langsam_multicam_fragment.py && echo "compile OK"
python3 -c "import yaml; c=yaml.safe_load(open('tcn_shm_vlm_inference.yaml'))['langsam_inference']; print(c['sam_backend'], c['sam_trt_engine'])"
python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py && python3 tests/test_langsam_multicam.py
```
Expected: `compile OK`; `pytorch /srv/...b3_fp16.engine`; `11/11`, `7/7`, `9/9`.

- [ ] **Step 6: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_common.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_all.yaml
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): optional TensorRT backend for the SAM 2 image encoder

SamTrtEncoder replaces the forward_image/_prepare_backbone_features block inside
_set_image_batch_gpu with one execute_async_v3, padding to the engine's fixed batch
and slicing the results back. Features never leave the GPU, so the decode path is
untouched. Defaults to the eager path (sam_backend: "pytorch").

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task S3: Build, validate, measure

Verification only; needs the running container.

- [ ] **Step 1: Build the engine**

```bash
python3 /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.py \
  --sam-type sam2.1_hiera_tiny --batch 3 --out /srv/models/active/sam2
```
Expected: ONNX written, engine written (batch 3, fp16), then `feature fidelity gate OK`,
`slice-consistency gate OK (batch 3)`, `mask IoU ... mask gate OK`, then `DONE`.

**If the mask gate fails**, rebuild with `--tf32` before concluding FP16 is unusable. **If the
slice gate fails**, Hiera baked its traced batch wrongly — stop and report.

- [ ] **Step 2: Micro-benchmark before adopting**

Time `encode()` at batch 3 against the 45.5 ms eager baseline (5 warmups, median of 20, per
§2.5 of `../optimization-playbook.md`). It must beat `torch.compile`'s equivalent or the engine
is not worth keeping.

- [ ] **Step 3: Run and compare**

Set `sam_backend: "trt"`, run the app, confirm masks are unchanged, then profile and compare the
`sam` NVTX stage against **93.3 ms (GPU 1)** and **72.4 ms (GPU 0)**, and the period against
194.5 ms.

- [ ] **Step 4: Record**

Append a `## Results` section to the spec with the measured stage times, period, fps and gate
outputs. Update `../optimization-playbook.md` §4 with the new arc entry. Commit.

---

## Self-Review notes

- **Spec coverage:** §1 wrapper → S1 Step 1 (`SAM2EncoderWrapper`, with both flags asserted); §2 fixed batch → S1 `build_engine` + S2 `encode`; §3 FP16 → S1 `--tf32` default-off; §4 gates → S1 `feature_gate`/`slice_gate`/`mask_gate` + temp-then-`os.replace`; §5 runtime → S2 Steps 1-3; §6 config → S2 Step 4. Testing table → S2 Step 5, S3 Steps 1-3. Risks → S3 Step 1 stop conditions, Step 2 the "must beat compile" bar.
- **Naming consistency:** `engine_batch`, `batch_error`, `plan_batch_padding` match the GDINO implementation. Output names `high_res_feats_0/1`, `image_embed` are identical in the exporter, `run_engine`, and `SamTrtEncoder.encode`.
- **Ordering:** S1 and S2 are independent (disjoint files); S3 needs both.
