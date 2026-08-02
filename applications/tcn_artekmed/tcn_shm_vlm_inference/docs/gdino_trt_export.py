#!/usr/bin/env python3
"""Export Grounding DINO-T -> fixed-resolution ONNX -> TF32 TensorRT engine for the TCN
LangSAM pipeline, with baked text tensors for fixed prompts and a PyTorch-vs-engine parity
gate. See gdino_trt_export.md for the environment + usage.

Run this INSIDE an IDEA-Research GroundingDINO checkout, in a dedicated venv with
transformers 4.x, tensorrt, onnx and opencv (NOT the Depth-Anything-3 venv). Produces:
  <out>/gdino_swint_<H>x<W>_tf32.engine
  <out>/gdino_swint_prompts.npz   (input_ids, attention_mask, position_ids, token_type_ids,
                                   text_token_mask, token_class_ids, prompts)
"""
from __future__ import annotations
import argparse
import os

import cv2
import numpy as np
import tensorrt as trt
import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)

MAX_TEXT_LEN = 256
INPUT_NAMES = ["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]


def load_model(config_file: str, checkpoint_path: str):
    args = SLConfig.fromfile(config_file)
    args.device = "cpu"
    args.use_checkpoint = False
    model = build_model(args)
    ck = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ck["model"]), strict=False)
    model.eval()
    return model


def build_text(model, prompts):
    """Return (caption, text_tensors dict, token_class_ids[256], L) for the fixed prompts."""
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
    dummy = (torch.randn(1, 3, H, W), text["input_ids"], text["attention_mask"],
             text["position_ids"], text["token_type_ids"], text["text_token_mask"])
    torch.onnx.export(
        model, f=onnx_path, args=dummy, input_names=INPUT_NAMES,
        output_names=["logits", "boxes"], opset_version=17, dynamo=False,
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


def parity_gate(model, ser, text, H, W, L, image_path, min_iou=0.99):
    """PyTorch (CPU) vs engine (GPU) top-box IoU on a real image; raise if < min_iou."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"parity image not found: {image_path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None]).float()

    with torch.no_grad():
        out = model(img, text["input_ids"], text["attention_mask"], text["position_ids"],
                    text["token_type_ids"], text["text_token_mask"])
    lg, bx = (out["pred_logits"], out["pred_boxes"]) if isinstance(out, dict) else out
    s_pt, b_pt = _top_box(lg.numpy(), bx.numpy())

    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(ser)
    ctx = eng.create_execution_context()
    feed = {"img": img, **text}
    for n, t in feed.items():
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
    s_trt, b_trt = _top_box(outs["logits"].cpu().numpy(), outs["boxes"].cpu().numpy())

    iou = _iou(b_pt, b_trt)
    print(f"parity: pytorch score={s_pt:.3f} vs trt score={s_trt:.3f} | top-box IoU={iou:.4f}")
    if iou < min_iou:
        raise SystemExit(f"PARITY GATE FAILED: IoU {iou:.4f} < {min_iou}")
    print("parity gate OK")


def main():
    ap = argparse.ArgumentParser(description="GDINO -> ONNX -> TRT engine + parity gate")
    ap.add_argument("--checkpoint", default="weights/groundingdino_swint_ogc.pth")
    ap.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--prompts", nargs="+", required=True, help='e.g. --prompts floor person')
    ap.add_argument("--hw", nargs=2, type=int, default=[512, 672], metavar=("H", "W"))
    ap.add_argument("--out", default="/data/models/active/groundingdino")
    ap.add_argument("--parity-image", default="images/in/car_1.jpg")
    ap.add_argument("--min-iou", type=float, default=0.99)
    ap.add_argument("--fp16", action="store_true", help="experimental; TF32 is the validated default")
    args = ap.parse_args()

    H, W = args.hw
    os.makedirs(args.out, exist_ok=True)
    tag = f"gdino_swint_{H}x{W}_{'fp16' if args.fp16 else 'tf32'}"
    onnx_path = os.path.join(args.out, tag + ".onnx")
    engine_path = os.path.join(args.out, tag + ".engine")
    npz_path = os.path.join(args.out, "gdino_swint_prompts.npz")

    print("Loading model (CPU) ...")
    model = load_model(args.config, args.checkpoint)
    caption, text, tcid, L = build_text(model, args.prompts)
    print(f"caption='{caption}'  L={L}  token_class_ids nonzero={int((tcid > 0).sum())}")

    np.savez(npz_path,
             input_ids=text["input_ids"].numpy(), attention_mask=text["attention_mask"].numpy(),
             position_ids=text["position_ids"].numpy(), token_type_ids=text["token_type_ids"].numpy(),
             text_token_mask=text["text_token_mask"].numpy(), token_class_ids=tcid,
             prompts=np.array(args.prompts, dtype=object))
    print(f"text tensors written: {npz_path}")

    export_onnx(model, text, H, W, onnx_path)
    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16)
    parity_gate(model, ser, text, H, W, L, args.parity_image, min_iou=args.min_iou)
    print(f"\nDONE. Engine: {engine_path}\n      Text:   {npz_path}")


if __name__ == "__main__":
    main()
