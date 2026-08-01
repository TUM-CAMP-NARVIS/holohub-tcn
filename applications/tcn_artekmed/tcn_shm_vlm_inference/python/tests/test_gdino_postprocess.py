# SPDX-License-Identifier: Apache-2.0
"""Host tests for the lean Grounding DINO post-process (numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import gdino_postprocess


def _sig_inv(p):  # logit that yields target prob p
    return np.log(p / (1 - p))


def test_gdino_postprocess_thresholds_classes_and_scales():
    # 256 text tokens; tokens 1,2 -> class 1 (floor); token 4 -> class 2 (person); rest 0
    tcid = np.zeros(256, np.int64); tcid[1] = 1; tcid[2] = 1; tcid[4] = 2
    Q = 3
    logits = np.full((Q, 256), _sig_inv(0.01), np.float32)   # baseline low
    logits[0, 1] = _sig_inv(0.90)     # query0 -> floor, score .90
    logits[1, 4] = _sig_inv(0.80)     # query1 -> person, score .80
    logits[2, 2] = _sig_inv(0.10)     # query2 -> floor .10 (below threshold)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2],
                      [0.25, 0.25, 0.1, 0.1],
                      [0.9, 0.9, 0.1, 0.1]], np.float32)
    bx, cls, sc = gdino_postprocess(logits, boxes, tcid, num_classes=2,
                                    box_threshold=0.3, img_h=100, img_w=200, xp=np)
    assert list(cls) == [1, 2]                       # query2 dropped
    assert np.allclose(sc, [0.90, 0.80], atol=1e-4)
    # query0 box cxcywh (.5,.5,.2,.2) on 200x100 -> xyxy pixels
    assert np.allclose(bx[0], [80, 40, 120, 60], atol=1e-3)


def test_gdino_postprocess_empty_when_all_below():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    logits = np.full((5, 256), _sig_inv(0.05), np.float32)
    boxes = np.tile(np.array([0.5, 0.5, 0.1, 0.1], np.float32), (5, 1))
    bx, cls, sc = gdino_postprocess(logits, boxes, tcid, 1, 0.3, 100, 100, xp=np)
    assert len(bx) == 0 and len(cls) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, repr(e))
    print(f"{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
