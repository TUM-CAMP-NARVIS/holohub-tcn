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
