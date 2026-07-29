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
