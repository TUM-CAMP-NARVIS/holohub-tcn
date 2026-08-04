#!/usr/bin/env python3
"""Export Grounding DINO-T -> fixed-resolution ONNX -> TF32 TensorRT engine for the TCN
LangSAM pipeline, with baked text tensors for fixed prompts and a PyTorch-vs-engine parity
gate. See gdino_trt_export.md for the environment + usage.

TWO STAGES, because a serialized TensorRT engine is version-locked and can only be loaded by
the TRT that built it, while the two halves of the toolchain live in different environments:

  --stage export   host, inside the wingdzero GroundingDINO fork checkout (torch + opencv +
                   groundingdino). Writes the text tensors, the ONNX, and a PyTorch parity
                   reference. Builds NO engine -- the host TRT is not the runtime TRT.
                     <out>/gdino_swint_prompts.npz          (input_ids, attention_mask,
                         position_ids, token_type_ids, text_token_mask, token_class_ids, prompts)
                     <out>/gdino_swint_<H>x<W>_tf32.onnx
                     <out>/gdino_swint_<H>x<W>_parity_ref.npz  (preprocessed image, top score,
                         top box) -- lets the build stage run the same faithfulness gate without
                         torch weights or the GroundingDINO checkout.
                   Requires: --prompts.

  --stage build    INSIDE the holohub runtime container (tensorrt + torch), so the engine
                   matches the TRT that will deserialize it at run time. Reads the three
                   artifacts above, writes the engine, and re-runs the parity gate against the
                   saved reference.
                     <out>/gdino_swint_<H>x<W>_tf32.engine

Both stages compare against the SAME PyTorch reference, so the full model -> ONNX -> engine
chain stays gated even though no single environment can run all of it.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import tensorrt as trt
import torch

MAX_TEXT_LEN = 256
INPUT_NAMES = ["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]


def load_model(config_file: str, checkpoint_path: str):
    # Imported lazily: the build stage runs in the runtime container, which has no
    # GroundingDINO checkout and no checkpoint.
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(config_file)
    args.device = "cpu"
    # Both must be off: gradient checkpointing wraps encoder layers in
    # torch.utils.checkpoint.checkpoint, which the TorchScript ONNX tracer cannot trace
    # (RuntimeError: unordered_map::at). The config defaults both to True.
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    model = build_model(args)
    ck = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ck["model"]), strict=False)
    model.eval()
    return model


def build_text(model, prompts):
    """Return (caption, text_tensors dict, token_class_ids[256], L) for the fixed prompts."""
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    caption = ". ".join(p.strip().strip(".").strip() for p in prompts) + " ."
    tok = model.tokenizer([caption], padding="longest", return_tensors="pt")
    specials = model.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tsam, position_ids, cate = generate_masks_with_special_tokens_and_transfer_map(
        tok, specials, model.tokenizer)
    L = tok["input_ids"].shape[1]

    # token_class_ids over the 256 logit text slots: category i (left-to-right) -> class i+1.
    tcid = np.zeros(MAX_TEXT_LEN, dtype=np.int64)
    cats = cate[0]  # list of per-category token masks over L tokens
    if len(cats) != len(prompts):
        print(f"WARNING: {len(cats)} caption categories vs {len(prompts)} prompts — "
              f"check the caption '{caption}'")
    for i, m in enumerate(cats):
        m = m.cpu().numpy().astype(bool) if hasattr(m, "cpu") else np.asarray(m, bool)
        tcid[:L][m] = i + 1

    text = {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"].bool(),
        "position_ids": position_ids,
        "token_type_ids": tok["token_type_ids"],
        "text_token_mask": tsam,
    }
    return caption, text, tcid, L


def export_onnx(model, text, H, W, onnx_path):
    # CRITICAL: GroundingDINO.forward caches backbone features in self.features/self.poss and only
    # recomputes them from the image if those attrs are absent. Any prior forward (e.g. the parity
    # reference) leaves them set, so the trace would bake those cached features as constants and
    # never connect the `img` input -> an image-independent engine. Clear the cache first so the
    # traced forward recomputes the backbone from the dummy image and wires img -> features -> out.
    if hasattr(model, "unset_image_tensor"):
        model.unset_image_tensor()
    dummy = (torch.randn(1, 3, H, W), text["input_ids"], text["attention_mask"],
             text["position_ids"], text["token_type_ids"], text["text_token_mask"])
    # dynamic_axes is REQUIRED: without it the legacy tracer constant-folds the image branch and
    # drops `img` as a graph input entirely, so the engine output becomes image-independent (it
    # bakes the dummy image). Mirror the wingdzero export config; the engine profiles below still
    # pin fixed shapes for the deployment resolution/prompt length.
    dynamic_axes = {
        "img": {0: "batch_size", 2: "height", 3: "width"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
        "logits": {0: "batch_size"},
        "boxes": {0: "batch_size"},
    }
    torch.onnx.export(
        model, f=onnx_path, args=dummy, input_names=INPUT_NAMES,
        output_names=["logits", "boxes"], opset_version=17, dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX written: {onnx_path}")


def build_engine(onnx_path, engine_path, H, W, L, fp16=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)   # TRT 10/11: explicit batch is implicit
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("PARSE ERROR:", parser.get_error(i))
            raise SystemExit("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)   # TRT<11; TRT11 uses strongly-typed fp16 ONNX
    prof = builder.create_optimization_profile()
    prof.set_shape("img", (1, 3, H, W), (1, 3, H, W), (1, 3, H, W))
    for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        prof.set_shape(n, (1, L), (1, L), (1, L))
    prof.set_shape("text_token_mask", (1, L, L), (1, L, L), (1, L, L))
    config.add_optimization_profile(prof)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"engine written: {engine_path}")
    return ser


def _top_box(logits, boxes):
    p = 1.0 / (1.0 + np.exp(-logits[0]))
    scores = np.nan_to_num(p, neginf=0.0).max(axis=1)
    j = int(scores.argmax())
    return scores[j], boxes[0, j]


def _iou(a, b):  # cxcywh
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2)-max(ax1, bx1)); iy = max(0.0, min(ay2, by2)-max(ay1, by1))
    inter = ix*iy; ua = a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua > 0 else 0.0


def _run_engine(ser, feed_cpu):
    """Deserialize `ser`, run one forward with the given CPU tensors, return {name: np.ndarray}."""
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(ser)
    ctx = eng.create_execution_context()
    feed = dict(feed_cpu)
    for n, t in list(feed.items()):
        want = trt.nptype(eng.get_tensor_dtype(n))
        t = t.to({np.int32: torch.int32, np.int64: torch.int64, np.float32: torch.float32,
                  np.bool_: torch.bool}[want]).cuda().contiguous()
        ctx.set_input_shape(n, tuple(t.shape)); ctx.set_tensor_address(n, t.data_ptr()); feed[n] = t
    outs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            outs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), device="cuda", dtype=torch.float32)
            ctx.set_tensor_address(n, outs[n].data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
    return {n: v.cpu().numpy() for n, v in outs.items()}


def _load_parity_img(image_path, H, W):
    import cv2  # export stage only; the build stage reads the preprocessed tensor from the ref npz

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"parity image not found: {image_path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None]).float()


def pytorch_reference(model, text, H, W, image_path):
    """Clean PyTorch (CPU) top-box on the parity image; returns (img, score, box_cxcywh).

    MUST be called BEFORE export_onnx. torch.onnx.export traces the model with a dummy random
    image and leaves GroundingDINO's cached image state such that the FIRST subsequent real
    forward is numerically wrong (e.g. top score 0.82 instead of 0.89, flipping the argmax box).
    Capturing the reference before any export sidesteps that; the engine is compared against it.
    """
    img = _load_parity_img(image_path, H, W)
    with torch.no_grad():
        out = model(img, text["input_ids"], text["attention_mask"], text["position_ids"],
                    text["token_type_ids"], text["text_token_mask"])
    lg, bx = (out["pred_logits"], out["pred_boxes"]) if isinstance(out, dict) else out
    s_pt, b_pt = _top_box(lg.numpy(), bx.numpy())
    return img, s_pt, b_pt


def parity_gate(ser, text, img, s_pt, b_pt, image_path, min_iou=0.99, min_detect=0.30):
    """Compare engine (GPU) top-box against the pre-captured PyTorch reference; raise on mismatch.

    The top-1 IoU is only meaningful when the image actually contains a prompted class: with no
    true detection every query is low-confidence noise and TF32 rounding flips which mediocre
    query wins argmax, so the boxes disagree even for a faithful engine. We therefore require the
    PyTorch top score to clear `min_detect` (a real detection) before trusting the IoU -- otherwise
    we abort telling the user to supply an image containing the prompted classes.
    """
    if s_pt < min_detect:
        raise SystemExit(
            f"PARITY IMAGE UNUSABLE: PyTorch top score {s_pt:.3f} < {min_detect} -- the parity "
            f"image '{image_path}' does not contain the prompted classes, so top-box IoU is not a "
            f"valid faithfulness check. Supply --parity-image with an image that clearly shows the "
            f"prompted classes (e.g. a person on a floor for --prompts floor person).")

    o = _run_engine(ser, {"img": img, **text})
    s_trt, b_trt = _top_box(o["logits"], o["boxes"])
    iou = _iou(b_pt, b_trt)
    print(f"parity: pytorch score={s_pt:.3f} vs trt score={s_trt:.3f} "
          f"(|d|={abs(s_pt - s_trt):.3f}) | top-box IoU={iou:.4f}")
    if iou < min_iou:
        raise SystemExit(f"PARITY GATE FAILED: IoU {iou:.4f} < {min_iou}")
    print("parity gate OK")


def save_parity_ref(path, img, s_pt, b_pt, image_path, H, W):
    """Persist the PyTorch reference so the build stage can gate without torch weights."""
    np.savez(path, image=img.numpy(), score=np.float32(s_pt), box_cxcywh=np.asarray(b_pt, np.float32),
             image_path=np.array(image_path), hw=np.array([H, W], np.int64))
    print(f"parity reference written: {path}")


def load_parity_ref(path, H, W):
    if not os.path.exists(path):
        raise SystemExit(
            f"parity reference not found: {path}\nRun the export stage on the host first:\n"
            f"  python3 gdino_trt_export.py --stage export --prompts <...> --hw {H} {W}")
    d = np.load(path, allow_pickle=False)
    rh, rw = (int(x) for x in d["hw"])
    if (rh, rw) != (H, W):
        raise SystemExit(f"parity reference is {rh}x{rw} but --hw is {H}x{W}; re-run the export stage")
    return (torch.from_numpy(d["image"]).float(), float(d["score"]), d["box_cxcywh"],
            str(d["image_path"]))


def load_text_npz(path):
    """Return (text_tensors dict, L) from the exported prompt tensors."""
    if not os.path.exists(path):
        raise SystemExit(
            f"text tensors not found: {path}\nRun the export stage on the host first.")
    d = np.load(path, allow_pickle=True)
    text = {n: torch.from_numpy(np.ascontiguousarray(d[n])) for n in INPUT_NAMES[1:]}
    return text, int(text["input_ids"].shape[1])


def stage_export(args, H, W, onnx_path, npz_path, ref_path):
    print("Loading model (CPU) ...")
    model = load_model(args.config, args.checkpoint)
    caption, text, tcid, L = build_text(model, args.prompts)
    print(f"caption='{caption}'  L={L}  token_class_ids nonzero={int((tcid > 0).sum())}")

    # Capture the PyTorch reference BEFORE export_onnx corrupts the model's first forward.
    ref_img, s_pt, b_pt = pytorch_reference(model, text, H, W, args.parity_image)
    print(f"pytorch reference: top score={s_pt:.3f} on {args.parity_image}")
    if s_pt < args.min_detect:
        raise SystemExit(
            f"PARITY IMAGE UNUSABLE: PyTorch top score {s_pt:.3f} < {args.min_detect} -- the parity "
            f"image '{args.parity_image}' does not contain the prompted classes, so top-box IoU is "
            f"not a valid faithfulness check. Supply --parity-image with an image that clearly "
            f"shows the prompted classes (e.g. a person on a floor for --prompts floor person).")

    np.savez(npz_path,
             input_ids=text["input_ids"].numpy(), attention_mask=text["attention_mask"].numpy(),
             position_ids=text["position_ids"].numpy(), token_type_ids=text["token_type_ids"].numpy(),
             text_token_mask=text["text_token_mask"].numpy(), token_class_ids=tcid,
             prompts=np.array(args.prompts, dtype=object))
    print(f"text tensors written: {npz_path}")

    save_parity_ref(ref_path, ref_img, s_pt, b_pt, args.parity_image, H, W)
    export_onnx(model, text, H, W, onnx_path)
    print(f"\nDONE (export). Now build the engine INSIDE the runtime container:\n"
          f"  python3 gdino_trt_export.py --stage build --hw {H} {W} --out <container path>")


def stage_build(args, H, W, onnx_path, engine_path, npz_path, ref_path):
    if not os.path.exists(onnx_path):
        raise SystemExit(
            f"ONNX not found: {onnx_path}\nRun the export stage on the host first.")
    text, L = load_text_npz(npz_path)
    ref_img, s_pt, b_pt, ref_image_path = load_parity_ref(ref_path, H, W)
    print(f"TensorRT {trt.__version__}  |  L={L}  |  pytorch reference: top score={s_pt:.3f} "
          f"on {ref_image_path}")

    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16)
    parity_gate(ser, text, ref_img, s_pt, b_pt, ref_image_path,
                min_iou=args.min_iou, min_detect=args.min_detect)
    print(f"\nDONE. Engine: {engine_path}\n      Text:   {npz_path}")


def main():
    ap = argparse.ArgumentParser(
        description="GDINO -> ONNX (host) -> TRT engine (container) + parity gate",
        epilog="A TRT engine only loads in the TRT that built it: run --stage build inside the "
               "runtime container, never on the host.")
    ap.add_argument("--stage", required=True, choices=("export", "build"),
                    help="'export' (host, GroundingDINO checkout): text tensors + ONNX + parity "
                         "reference, no engine. 'build' (runtime container): ONNX -> engine + "
                         "parity gate against the saved reference.")
    ap.add_argument("--checkpoint", default="weights/groundingdino_swint_ogc.pth")
    ap.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--prompts", nargs="+", help='export stage only; e.g. --prompts floor person')
    ap.add_argument("--hw", nargs=2, type=int, default=[512, 672], metavar=("H", "W"))
    ap.add_argument("--out", default="/data/models/active/groundingdino",
                    help="artifact directory; inside the container this is the mounted path, "
                         "e.g. /srv/models/active/groundingdino")
    ap.add_argument("--parity-image", default="images/in/person.jpg",
                    help="export stage only; must contain the prompted classes (see "
                         "--min-detect); default suits --prompts floor person")
    ap.add_argument("--min-iou", type=float, default=0.99)
    ap.add_argument("--min-detect", type=float, default=0.30,
                    help="PyTorch top score the parity image must reach for the IoU check to be "
                         "valid (i.e. the image must actually contain a prompted class)")
    ap.add_argument("--fp16", action="store_true", help="experimental; TF32 is the validated default")
    args = ap.parse_args()

    H, W = args.hw
    os.makedirs(args.out, exist_ok=True)
    tag = f"gdino_swint_{H}x{W}_{'fp16' if args.fp16 else 'tf32'}"
    onnx_path = os.path.join(args.out, tag + ".onnx")
    engine_path = os.path.join(args.out, tag + ".engine")
    npz_path = os.path.join(args.out, "gdino_swint_prompts.npz")
    ref_path = os.path.join(args.out, f"gdino_swint_{H}x{W}_parity_ref.npz")

    if args.stage == "export":
        if not args.prompts:
            ap.error("--stage export requires --prompts")
        stage_export(args, H, W, onnx_path, npz_path, ref_path)
    else:
        stage_build(args, H, W, onnx_path, engine_path, npz_path, ref_path)


if __name__ == "__main__":
    main()
