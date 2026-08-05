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
import argparse, os, shutil, sys, tempfile

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from langsam_common import SAM, SAM_MODELS          # noqa: E402
from langsam_helpers import resolve_workers, distinct_batches      # noqa: E402

IMAGE_SIZE = 1024
OUT_NAMES = ["high_res_feats_0", "high_res_feats_1", "image_embed"]


def batches_from_config(config_path):
    """Distinct engine batches the configured GPU split needs.

    Reading the same `gpu_workers` node the application reads is the point: the engines that
    get built cannot drift from the split that runs.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    workers = resolve_workers(cfg.get("gpu_workers"), [])
    batches = distinct_batches(workers)
    if not batches:
        raise SystemExit(f"no workers with cameras in {config_path}: nothing to build")
    print(f"gpu_workers -> {[(w['device'], len(w['cameras'])) for w in workers]} "
          f"-> building batches {batches}")
    return batches


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
    # dynamo=False is REQUIRED (same as gdino_trt_export.py). Recent torch defaults to the
    # dynamo exporter, which here: pulls in an onnxscript dependency, silently bumps to opset
    # 18 and then fails converting back down ("No Adapter To Version 17 for Resize"), and --
    # worst -- hands the graph to onnxscript's constant folder, which evaluates the Hiera
    # Resize nodes through ONNX's pure-PYTHON reference implementation
    # (onnx/reference/ops/op_resize.py). On 1024^2 feature maps that never finishes: observed
    # as a hang at 112% CPU with no output. The legacy tracer does none of that.
    torch.onnx.export(wrapper, dummy, onnx_path, export_params=True, opset_version=17,
                      do_constant_folding=True, input_names=["image"],
                      output_names=OUT_NAMES, dynamic_axes=dyn, dynamo=False)
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


def _rel_err(ref32, other):
    a = ref32.float().flatten()
    b = other.float().flatten()
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def feature_gate(ref_fp32, ref_bf16, trt_out, margin=1.5):
    """Is the engine any further from full precision than the path we already ship?

    An earlier version of this gate compared the engine against the bf16 reference with fixed
    thresholds. That was wrong twice over: bf16 has FEWER mantissa bits (8) than the engine's
    FP16 (10), so the measurement conflated the engine's error with bf16's own, and the
    thresholds were arbitrary constants rather than anything derived from the system.

    So: measure both against an FP32 reference, and require the engine to be no worse than the
    bf16 autocast path already running in production, times `margin`. That is a criterion with
    meaning -- "not a precision regression relative to what you already accept" -- and it also
    answers FP16-vs-TF32 directly: if trt error lands far outside bf16's, FP16 is too coarse.

    Returns a list of failure strings (empty when the gate passes) rather than raising, so every
    gate can report before anything aborts; the mask IoU below is the decisive one and must not
    be hidden by a proxy failing first.
    """
    fails = []
    for name, f32, bf16, t in zip(OUT_NAMES, ref_fp32, ref_bf16, trt_out):
        e_bf16 = _rel_err(f32, bf16)
        e_trt = _rel_err(f32, t)
        budget = max(e_bf16 * margin, 1e-3)
        ratio = e_trt / max(e_bf16, 1e-12)
        ok = e_trt <= budget
        print(f"  feature {name:18} err_vs_fp32: bf16={e_bf16:.4f} trt={e_trt:.4f} "
              f"({ratio:.2f}x bf16, budget {budget:.4f})  [{'OK' if ok else 'FAIL'}]")
        if not ok:
            fails.append(f"{name}: trt err {e_trt:.4f} > {margin}x the bf16 path's {e_bf16:.4f}")
    print("feature fidelity gate " + ("OK" if not fails else "FAILED"))
    return fails


def slice_gate(trt_out, batch):
    """Identical inputs across the batch must give identical outputs. Returns failures."""
    if batch < 2:
        print("slice-consistency gate: batch 1, nothing to compare")
        return []
    fails = []
    for name, t in zip(OUT_NAMES, trt_out):
        d = max((t[i] - t[0]).abs().max().item() for i in range(1, t.shape[0]))
        if d > 1e-4:
            fails.append(f"{name} differs by {d:.3e} across slices; the engine baked its "
                         f"traced batch incorrectly")
    print("slice-consistency gate " + (f"OK (batch {batch})" if not fails else "FAILED"))
    return fails


def mask_gate(sam, images_gpu, ref, trt_out, min_iou=0.99):
    """The decisive gate: same decoder, same boxes, features swapped. Returns failures."""
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
    print(f"  mask IoU (pytorch vs trt features) = {iou:.4f}  "
          f"[reference coverage {cov:.1%}, need >= {min_iou}]")
    fails = [] if iou >= min_iou else [f"mask IoU {iou:.4f} < {min_iou}"]
    print("mask gate " + ("OK" if not fails else "FAILED"))
    return fails


def main():
    ap = argparse.ArgumentParser(description="SAM 2 image encoder -> ONNX -> TRT (in-container)")
    ap.add_argument("--sam-type", default="sam2.1_hiera_tiny", choices=sorted(SAM_MODELS))
    ap.add_argument("--batch", type=int, default=None,
                    help="engine batch = the largest LangSAM worker's camera count; the trace "
                         "and the profile both use it, and the artifacts are named _b<N>_ "
                         "(default: 3; mutually exclusive with --from-config)")
    ap.add_argument("--from-config", default=None,
                    help="path to tcn_shm_vlm_inference.yaml; builds one engine per distinct "
                         "worker camera count in its gpu_workers node (mutually exclusive "
                         "with --batch)")
    ap.add_argument("--out", default="/srv/models/active/sam2")
    ap.add_argument("--image", default=None,
                    help="optional real frame for the mask gate; a synthetic structured image "
                         "is used otherwise")
    ap.add_argument("--tf32", action="store_true", help="build TF32 instead of FP16")
    ap.add_argument("--min-iou", type=float, default=0.99)
    args = ap.parse_args()

    # A plain argparse mutually-exclusive group is NOT safe here: it flags a conflict only when
    # the parsed value is not identical (by `is`) to the argument's default, and CPython caches
    # small ints, so `--batch 3 --from-config ...` (3 happens to be this tool's default) would
    # silently pass through uncaught. Checking `args.batch is not None` (its default is None,
    # never a user-supplied value) sidesteps that entirely.
    if args.batch is not None and args.from_config:
        ap.error("argument --from-config: not allowed with argument --batch")

    if args.from_config:
        batches = batches_from_config(args.from_config)
    else:
        batches = [int(args.batch) if args.batch is not None else 3]

    dev = torch.device("cuda:0")
    os.makedirs(args.out, exist_ok=True)

    print(f"TensorRT {trt.__version__} | sam-type {args.sam_type}")
    sam = build_sam(args.sam_type, dev)
    wrapper = SAM2EncoderWrapper(sam.model).eval().to(dev)

    for batch in batches:
        tag = f"{args.sam_type}_encoder_b{int(batch)}_{'tf32' if args.tf32 else 'fp16'}"
        final_engine = os.path.join(args.out, tag + ".engine")

        print(f"\nbuilding {tag}")
        images = make_test_image(sam, batch, dev, args.image)
        batch_in = preprocess(sam, images)
        # TWO references. fp32 is the ground truth the engine is judged against; bf16 is what
        # the production path (SAM._autocast) already produces, and therefore the yardstick for
        # how much deviation is acceptable. Comparing the engine only against bf16 -- which has
        # FEWER mantissa bits than FP16 -- measured the wrong thing.
        with torch.no_grad():
            ref_fp32 = list(wrapper(batch_in))
        with torch.no_grad(), sam._autocast():
            ref_bf16 = list(wrapper(batch_in))

        # Build into a temp dir and move into place only after every gate passes, so a failed
        # run can never leave an unvalidated engine at the final path (the GDINO builder's
        # defect). On failure the directory is KEPT and its path printed: deleting the ONNX
        # along with it destroyed exactly the artifact needed to diagnose the failure.
        tmp = tempfile.mkdtemp(dir=args.out, prefix=f"build-{tag}-")
        ok = False
        try:
            onnx_tmp = os.path.join(tmp, tag + ".onnx")
            eng_tmp = os.path.join(tmp, tag + ".engine")
            with torch.no_grad():
                export_onnx(wrapper, onnx_tmp, batch, dev)
            data = build_engine(onnx_tmp, eng_tmp, batch, fp16=not args.tf32)
            out = run_engine(data, batch_in)
            # Run every gate before deciding: the mask IoU is the decisive one and must not be
            # hidden by a proxy gate aborting first.
            fails = (feature_gate(ref_fp32, ref_bf16, out)
                     + slice_gate(out, int(batch))
                     + mask_gate(sam, images, ref_bf16, out, min_iou=args.min_iou))
            if fails:
                raise SystemExit("GATES FAILED:\n  - " + "\n  - ".join(fails))
            os.replace(eng_tmp, final_engine)
            ok = True
        finally:
            if ok:
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                print(f"\nArtifacts kept for diagnosis: {tmp}")
        print(f"\nDONE. Engine: {final_engine}")


if __name__ == "__main__":
    main()
