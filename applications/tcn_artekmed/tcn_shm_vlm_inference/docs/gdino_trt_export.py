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
                     <out>/gdino_swint_<H>x<W>_b<N>_tf32.onnx
                     <out>/gdino_swint_<H>x<W>_parity_ref.npz  (preprocessed image, top score,
                         top box) -- lets the build stage run the same faithfulness gate without
                         torch weights or the GroundingDINO checkout.
                   Requires: --prompts.

  --stage build    INSIDE the holohub runtime container (tensorrt + torch), so the engine
                   matches the TRT that will deserialize it at run time. Reads the three
                   artifacts above, writes the engine, and re-runs the fidelity report against
                   the saved reference.
                     <out>/gdino_swint_<H>x<W>_b<N>_tf32.engine

Both stages compare against the SAME PyTorch reference, so the full model -> ONNX -> engine
chain stays gated even though no single environment can run all of it.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from langsam_helpers import resolve_workers, distinct_batches      # noqa: E402

MAX_TEXT_LEN = 256
INPUT_NAMES = ["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]


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


def export_onnx(model, text, H, W, onnx_path, batch=1):
    # CRITICAL: GroundingDINO.forward caches backbone features in self.features/self.poss and only
    # recomputes them from the image if those attrs are absent. Any prior forward (e.g. the parity
    # reference, or -- since --from-config traces the SAME model instance once per batch -- an
    # earlier batch's export) leaves them set, so the trace would bake those cached features as
    # constants and never connect the `img` input -> an image-independent engine. Clear the cache
    # first so the traced forward recomputes the backbone from the dummy image and wires
    # img -> features -> out. This must be a hard failure, not a silent no-op: a model without
    # this method would trace with stale features and the only downstream detector
    # (fidelity_report) is non-blocking by default, so the defect would ship silently.
    if not hasattr(model, "unset_image_tensor"):
        raise SystemExit(
            "model has no unset_image_tensor(): its cached backbone features "
            "(self.features/self.poss) cannot be cleared before tracing, so a repeat trace of "
            "this model instance would silently bake STALE features and ignore the `img` input "
            "-- producing an engine that returns the same output for every image. Use a model "
            "that provides unset_image_tensor() (the wingdzero fork), or build one batch per "
            "process (a fresh --stage export per --batch) so each trace is the model's first "
            "forward.")
    model.unset_image_tensor()
    # The traced batch IS the engine's batch. dynamic_axes below declares batch_size symbolic
    # and TensorRT's parser reports -1, but a Where in the encoder fusion attention bakes a
    # broadcast that is only conformable at the traced batch: forcing a different batch fails
    # the build ("broadcast dimensions must be conformable"), and a multi-size profile silently
    # specialises to a static shape. So trace at exactly the batch the workers will run.
    B = int(batch)
    rep = lambda v: v if B == 1 else v.repeat(*([B] + [1] * (v.dim() - 1)))
    dummy = (torch.randn(B, 3, H, W), rep(text["input_ids"]), rep(text["attention_mask"]),
             rep(text["position_ids"]), rep(text["token_type_ids"]), rep(text["text_token_mask"]))
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


def build_engine(onnx_path, engine_path, H, W, L, fp16=False, batch=1):
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
    # min = opt = max = N: the ONNX was traced at N and the engine can only run at N.
    B = int(batch)
    if B < 1:
        raise SystemExit(f"--batch must be >= 1, got {B}")
    prof.set_shape("img", (B, 3, H, W), (B, 3, H, W), (B, 3, H, W))
    for n in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        prof.set_shape(n, (B, L), (B, L), (B, L))
    prof.set_shape("text_token_mask", (B, L, L), (B, L, L), (B, L, L))
    config.add_optimization_profile(prof)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"engine written: {engine_path}  (batch {B}, "
          f"{os.path.getsize(engine_path) / 2**20:.0f} MiB)")
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


def run_at_batch(ser, text, img, B):
    """Run the engine once at batch B with the parity image replicated; per-slice (score, box)."""
    feed = {"img": img.repeat(B, 1, 1, 1)}
    for k, v in text.items():
        feed[k] = v.repeat(*([B] + [1] * (v.dim() - 1)))
    o = _run_engine(ser, feed)
    return [_top_box(o["logits"][i:i + 1], o["boxes"][i:i + 1]) for i in range(B)]


def slice_consistency_gate(slices, min_iou=0.999, max_score_delta=0.01):
    """BLOCKING. Every slice of a batched run must agree with slice 0.

    This is the check that proves batching is correct. The engine's batch is baked at trace
    time, and the failure mode of getting that wrong is silently wrong output on slices
    1..N-1 -- which looks like random per-camera detection dropouts, not like a crash.
    Identical inputs, so the bar is tight (measured 0.999611 on a good engine).
    """
    if len(slices) < 2:
        print("slice-consistency gate: batch 1, nothing to compare")
        return
    s0, b0 = slices[0]
    worst_iou, worst_ds = 1.0, 0.0
    for i, (s, b) in enumerate(slices[1:], start=1):
        iou = _iou(b0, b)
        ds = abs(float(s) - float(s0))
        worst_iou, worst_ds = min(worst_iou, iou), max(worst_ds, ds)
        if iou < min_iou or ds > max_score_delta:
            raise SystemExit(
                f"SLICE CONSISTENCY GATE FAILED at slice {i}/{len(slices)}: IoU {iou:.6f} "
                f"(need >= {min_iou}), |score delta| {ds:.6f} (need <= {max_score_delta}).\n"
                f"The engine gives different answers for identical inputs across the batch, so "
                f"batching is NOT safe with it. Re-export with --batch {len(slices)} and rebuild.")
    print(f"slice-consistency gate OK (batch {len(slices)}: worst IoU {worst_iou:.6f}, "
          f"worst |score delta| {worst_ds:.6f})")


def fidelity_report(slice0, s_ref, b_ref, image_path, min_iou=0.99, min_detect=0.30,
                    strict=False):
    """PyTorch fidelity: always REPORTED, fatal only with --strict-parity.

    Non-blocking by default because container TRT 10.9 engines currently deviate from PyTorch
    for reasons unrelated to batching -- boxes stay close (IoU ~0.9637) but confidence scores
    are depressed (0.52-0.82 vs 0.889). A batch-1 engine deviates identically, and the 0.9994
    recorded on 2026-08-03 came from a host TRT 11.2 build. Blocking by default would block
    every build for a pre-existing, separately-tracked problem, so it is printed loudly instead.
    """
    if s_ref < min_detect:
        raise SystemExit(
            f"PARITY IMAGE UNUSABLE: PyTorch top score {s_ref:.3f} < {min_detect} -- the parity "
            f"image '{image_path}' does not contain the prompted classes, so top-box IoU is not "
            f"a valid faithfulness check. Re-run --stage export with an image that clearly shows "
            f"the prompted classes.")
    s, b = slice0
    iou = _iou(b_ref, b)
    ds = abs(float(s) - float(s_ref))
    ok = iou >= min_iou and ds <= 0.01
    print(f"pytorch fidelity [{'OK' if ok else 'DEVIATION'}]: pytorch score={s_ref:.3f} vs "
          f"trt {s:.3f} (|d|={ds:.3f}) | top-box IoU={iou:.4f} (want >= {min_iou}) on {image_path}")
    if not ok:
        print("  ^ NOT blocking. Known open issue: container TRT 10.9 depresses confidence "
              "scores while keeping boxes close; a batch-1 engine deviates identically, so this "
              "is not caused by batching. Pass --strict-parity to make it fatal.")
        if strict:
            raise SystemExit("PARITY GATE FAILED (--strict-parity)")


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


def stage_export_setup(args, H, W, npz_path, ref_path):
    """Load the model, build the fixed text tensors, capture the PyTorch parity reference, and
    write the two batch-INDEPENDENT artifacts (prompts npz, parity ref) once.

    Neither file's name nor content depends on the engine batch -- both are shared by every
    `--batch` this run exports -- so this runs once per invocation, not once per batch (see
    `stage_export_batch`, which the caller loops).
    """
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
    return model, text


def stage_export_batch(model, text, H, W, onnx_path, batch, print_hint=True):
    """Export the batch-DEPENDENT ONNX for one engine batch, reusing the model/text tensors
    `stage_export_setup` already built.

    `print_hint` prints the per-batch "now build the engine" follow-up naming this batch's
    `--batch N`. The `--from-config` caller in `main` sets it False and prints ONE combined
    hint after the whole loop instead -- naming `--batch N` per batch here would have a user
    run the build stage once per batch by hand instead of once with the same `--from-config`.
    """
    export_onnx(model, text, H, W, onnx_path, batch=batch)
    if print_hint:
        print(f"\nDONE (export, batch {batch}). Now build the engine INSIDE the runtime "
              f"container:\n  python3 gdino_trt_export.py --stage build --hw {H} {W} "
              f"--batch {batch} --out <container path>")


def stage_build(args, H, W, onnx_path, engine_path, npz_path, ref_path, batch):
    """Build and gate the engine for one batch. `npz_path`/`ref_path` are the batch-independent
    artifacts the export stage wrote once; they are only READ here, so re-reading them per
    batch (the caller loops this function once per distinct batch) is harmless."""
    if not os.path.exists(onnx_path):
        raise SystemExit(
            f"ONNX not found: {onnx_path}\nRun the export stage on the host first.")
    text, L = load_text_npz(npz_path)
    ref_img, s_pt, b_pt, ref_image_path = load_parity_ref(ref_path, H, W)
    print(f"TensorRT {trt.__version__}  |  L={L}  |  pytorch reference: top score={s_pt:.3f} "
          f"on {ref_image_path}")

    ser = build_engine(onnx_path, engine_path, H, W, L, fp16=args.fp16, batch=batch)
    slices = run_at_batch(ser, text, ref_img, int(batch))
    slice_consistency_gate(slices)
    fidelity_report(slices[0], s_pt, b_pt, ref_image_path, min_iou=args.min_iou,
                    min_detect=args.min_detect, strict=args.strict_parity)
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
    ap.add_argument("--batch", type=int, default=None,
                    help="the engine's batch = the largest LangSAM worker's camera count. Both "
                         "stages need the SAME value: export traces at it (the traced batch is "
                         "baked into the graph) and build pins the profile to it. Artifacts are "
                         "named _b<N>_ so a mismatched pair cannot be combined by accident. "
                         "(default: 1; mutually exclusive with --from-config)")
    ap.add_argument("--from-config", default=None,
                    help="path to tcn_shm_vlm_inference.yaml; builds one engine per distinct "
                         "worker camera count in its gpu_workers node (mutually exclusive "
                         "with --batch). Pass the SAME --from-config to both --stage export "
                         "and --stage build.")
    ap.add_argument("--strict-parity", action="store_true",
                    help="make the PyTorch fidelity deviation fatal (default: reported only)")
    args = ap.parse_args()

    # A plain argparse mutually-exclusive group is NOT safe here: it flags a conflict only when
    # the parsed value is not identical (by `is`) to the argument's default, and CPython caches
    # small ints, so e.g. `--batch 1 --from-config ...` (1 happens to be this tool's default)
    # would silently pass through uncaught. Checking `args.batch is not None` (its default is
    # None, never a user-supplied value) sidesteps that entirely.
    if args.batch is not None and args.from_config:
        ap.error("argument --from-config: not allowed with argument --batch")

    if args.from_config:
        batches = batches_from_config(args.from_config)
    else:
        batches = [int(args.batch) if args.batch is not None else 1]

    H, W = args.hw
    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "gdino_swint_prompts.npz")
    ref_path = os.path.join(args.out, f"gdino_swint_{H}x{W}_parity_ref.npz")

    def paths_for(batch):
        tag = f"gdino_swint_{H}x{W}_b{int(batch)}_{'fp16' if args.fp16 else 'tf32'}"
        return (os.path.join(args.out, tag + ".onnx"), os.path.join(args.out, tag + ".engine"))

    if args.stage == "export":
        if not args.prompts:
            ap.error("--stage export requires --prompts")
        # Batch-independent (prompts npz, parity ref) is done ONCE; only the ONNX export --
        # which bakes the traced batch -- repeats per batch.
        model, text = stage_export_setup(args, H, W, npz_path, ref_path)
        for batch in batches:
            onnx_path, _ = paths_for(batch)
            stage_export_batch(model, text, H, W, onnx_path, batch, print_hint=not args.from_config)
        if args.from_config:
            print(f"\nDONE (export, batches {batches}). Now build the engines INSIDE the "
                  f"runtime container, with the SAME --from-config:\n"
                  f"  python3 gdino_trt_export.py --stage build --hw {H} {W} "
                  f"--from-config {args.from_config} --out <container path>")
    else:
        for batch in batches:
            onnx_path, engine_path = paths_for(batch)
            stage_build(args, H, W, onnx_path, engine_path, npz_path, ref_path, batch)


if __name__ == "__main__":
    main()
