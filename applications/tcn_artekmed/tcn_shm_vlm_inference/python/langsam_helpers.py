# SPDX-License-Identifier: Apache-2.0
"""Pure, dependency-light LangSAM helpers (numpy-only) so they unit-test on the host.

The array function is array-module-agnostic: pass ``xp=cupy`` at runtime and ``xp=numpy``
in tests. Re-exported from ``langsam_common`` for convenience.
"""

import numpy as np


def resolve_workers(multicam_cfg, all_color_cameras):
    """Resolve the per-GPU worker assignment.

    A worker is ``{"device": int, "cameras": [port, ...]}``. Empty/missing ``workers`` ->
    a single worker on device 0 processing all color cameras.
    """
    workers = (multicam_cfg or {}).get("workers") or []
    if not workers:
        return [{"device": 0, "cameras": list(all_color_cameras)}]
    return [
        {"device": int(w.get("device", 0)), "cameras": list(w.get("cameras") or [])}
        for w in workers
    ]


def class_id_map(prompts):
    """Normalized prompt -> 1-based class id (0 reserved for background)."""
    return {str(p).strip().lower(): i + 1 for i, p in enumerate(prompts)}


def class_id_for_label(label, cmap):
    """Class id for a detected label: exact match, then substring, else 0 (background)."""
    key = str(label).strip().lower()
    if key in cmap:
        return cmap[key]
    for known, cid in cmap.items():
        if known and (known in key or key in known):
            return cid
    return 0


def build_label_map(masks, labels, scores, cmap, height, width, xp=np):
    """(M,H,W) masks + labels + scores -> (H,W) uint8 class-label map.

    Detections are painted in ascending score order so the highest-confidence class wins on
    overlap. Unknown labels (class id 0) are skipped.
    """
    label_map = xp.zeros((height, width), dtype=xp.uint8)
    if masks is None or len(masks) == 0:
        return label_map
    order = xp.argsort(scores)
    order = [int(i) for i in (order.tolist() if hasattr(order, "tolist") else order)]
    for j in order:
        cid = class_id_for_label(labels[j], cmap)
        if cid == 0:
            continue
        label_map[masks[j] > 0] = cid
    return label_map


# Packed panoptic encoding: value = (class_id << 8) | instance_id; 0 = background.
PANOPTIC_CLASS_SHIFT = 8
PANOPTIC_INSTANCE_MASK = 0xFF


def panoptic_class(value):
    """Class id from a packed panoptic value (scalar or array)."""
    return value >> PANOPTIC_CLASS_SHIFT


def panoptic_instance(value):
    """Instance id from a packed panoptic value (scalar or array)."""
    return value & PANOPTIC_INSTANCE_MASK


def build_panoptic_map(masks, labels, scores, cmap, height, width, xp=np):
    """(M,H,W) masks + labels + scores -> (H,W) uint16 panoptic map.

    Each value packs ``(class_id << 8) | instance_id`` (0 = background). Instances are
    numbered per class by descending score (instance 1 = most confident); painting is done in
    ascending score order so the most confident detection wins on overlap. Instance ids are
    per-frame (not temporally stable). Unknown labels (class id 0) are skipped.
    """
    pmap = xp.zeros((height, width), dtype=xp.uint16)
    if masks is None or len(masks) == 0:
        return pmap
    order = xp.argsort(scores)
    order = [int(i) for i in (order.tolist() if hasattr(order, "tolist") else order)]
    inst_count = {}
    value_of = {}
    for j in reversed(order):                       # descending score: number instances
        cid = class_id_for_label(labels[j], cmap)
        if cid == 0:
            value_of[j] = 0
            continue
        inst_count[cid] = inst_count.get(cid, 0) + 1
        value_of[j] = (cid << PANOPTIC_CLASS_SHIFT) | min(inst_count[cid], PANOPTIC_INSTANCE_MASK)
    for j in order:                                 # ascending score: highest wins overlap
        v = value_of[j]
        if v:
            pmap[masks[j] > 0] = v
    return pmap


def gdino_postprocess(logits, boxes, token_class_ids, num_classes,
                      box_threshold, img_h, img_w, xp=np):
    """Grounding DINO raw outputs -> detections, using a fixed token->class map.

    logits (Q,256), boxes (Q,4) cxcywh in [0,1], token_class_ids (256,) with class id
    (1..num_classes) for prompt tokens else 0. Returns (boxes_xyxy_px, class_ids, scores) for
    queries whose best per-class score exceeds box_threshold. Array-module-agnostic (xp).
    """
    probs = 1.0 / (1.0 + xp.exp(-logits))                       # (Q,256)
    tcid = xp.asarray(token_class_ids)
    Q = probs.shape[0]
    class_score = xp.zeros((Q, num_classes + 1), dtype=probs.dtype)  # col 0 = background/unused
    for c in range(1, num_classes + 1):
        mask = (tcid == c)
        if bool(mask.any()):
            class_score[:, c] = probs[:, mask].max(axis=1)
    best_cls = class_score[:, 1:].argmax(axis=1) + 1            # (Q,) 1-based
    best_score = class_score[xp.arange(Q), best_cls]            # (Q,)
    keep = best_score > box_threshold
    b = boxes[keep]
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = xp.stack([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                     (cx + w / 2) * img_w, (cy + h / 2) * img_h], axis=1)
    return xyxy, best_cls[keep], best_score[keep]


def build_class_token_masks(token_class_ids, num_classes, xp=np):
    """(num_classes, 256) bool; row c-1 marks the token slots belonging to class c.

    Hoisted out of the per-frame path. `token_class_ids` is fixed for a given prompt set, so
    these masks -- and the per-class `.any()` check the per-image path ran for every camera on
    every frame, each a device->host sync -- are computed once, when the active prompt set
    changes. See GDinoTrtDetector.set_prompts.
    """
    tcid = xp.asarray(token_class_ids)
    return xp.stack([(tcid == c) for c in range(1, int(num_classes) + 1)], axis=0)


def gdino_postprocess_batch(logits, boxes, class_masks, img_hw, xp=np):
    """Batched Grounding DINO decode -- adds NO synchronisation.

    logits (N,Q,256), boxes (N,Q,4) cxcywh in [0,1], class_masks (C,256) bool from
    build_class_token_masks, img_hw a list of N (h,w) giving each camera's ORIGINAL pixel size.

    Returns (xyxy (N,Q,4) in pixels, best_cls (N,Q) 1-based, best_score (N,Q)) for ALL queries;
    the caller thresholds on `best_score` after a single device->host transfer. Deliberately
    returns unfiltered arrays: boolean indexing would force cupy to size the output on the
    host, which is one of the syncs this whole change exists to remove.
    """
    probs = 1.0 / (1.0 + xp.exp(-logits))                          # (N,Q,256)
    # xp.where(mask, probs, 0.0) instead of probs[..., mask]: sigmoids are strictly > 0, so
    # masking with 0 yields the same maximum as selecting the class's columns, while an empty
    # class scores exactly 0 -- matching gdino_postprocess -- and neither indexes nor syncs.
    per_class = xp.stack(
        [xp.where(class_masks[c], probs, 0.0).max(axis=-1) for c in range(class_masks.shape[0])],
        axis=-1)                                                   # (N,Q,C)
    best_idx = per_class.argmax(axis=-1)                           # (N,Q) 0-based
    best_cls = best_idx + 1                                        # 1-based; 0 is background
    best_score = xp.take_along_axis(per_class, best_idx[..., None], axis=-1)[..., 0]

    h = xp.asarray([float(a) for a, _ in img_hw]).reshape(-1, 1)
    w = xp.asarray([float(b) for _, b in img_hw]).reshape(-1, 1)
    cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = xp.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], axis=-1)   # (N,Q,4) pixels
    return xyxy, best_cls, best_score


def build_prompt_remap(baked_prompts, active_prompts):
    """Baked class ids -> active class ids, for a prompt set the engine can already express.

    The TRT engine bakes the prompt TOKENS (input_ids, text_token_mask, fixed L), but the
    prompt->class mapping lives entirely in token_class_ids, where class i+1 is baked prompt i
    (see docs/gdino_trt_export.py build_text). So any subset and/or reordering of the baked
    prompts is expressible by renumbering alone -- no tokenizer, no re-export.

    Returns an int64 array of length len(baked_prompts)+1: index 0 (background) maps to 0, and
    index i+1 maps to the 1-based position of baked prompt i in active_prompts, or 0 if that
    prompt was dropped. Apply as `remap[token_class_ids]`.

    Raises ValueError if active_prompts is empty, contains duplicates after normalisation, or
    contains a term that is not baked into the engine -- that term's tokens simply are not in
    the engine's input_ids, so it requires a re-export + rebuild.
    """
    def _norm(p):
        return str(p).strip().lower()

    baked = [_norm(p) for p in baked_prompts]
    active = [_norm(p) for p in active_prompts]
    if not active:
        raise ValueError("active prompt set is empty; at least one prompt is required")
    if len(set(active)) != len(active):
        raise ValueError(f"duplicate prompts after normalisation: {active}")
    unknown = [p for p in active if p not in baked]
    if unknown:
        raise ValueError(
            f"prompts {unknown} are not baked into the GDINO TRT engine (baked: {baked}). "
            f"Only a subset or reordering of the baked prompts can be applied at runtime; a new "
            f"term needs a re-export and rebuild:\n"
            f"  host:      python3 gdino_trt_export.py --stage export --prompts {' '.join(active)} ...\n"
            f"  container: python3 gdino_trt_export.py --stage build --out /srv/models/active/groundingdino")
    remap = np.zeros(len(baked) + 1, np.int64)
    for i, p in enumerate(baked):
        remap[i + 1] = active.index(p) + 1 if p in active else 0
    return remap
