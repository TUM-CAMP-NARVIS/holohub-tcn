# SPDX-License-Identifier: Apache-2.0
"""Host tests for the lean Grounding DINO post-process (numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import gdino_postprocess, build_class_token_masks, gdino_postprocess_batch


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


def test_build_class_token_masks_marks_each_class():
    tcid = np.zeros(256, np.int64); tcid[1] = 1; tcid[2] = 1; tcid[4] = 2
    m = build_class_token_masks(tcid, num_classes=2, xp=np)
    assert m.shape == (2, 256)
    assert list(np.nonzero(m[0])[0]) == [1, 2]      # class 1 -> tokens 1,2
    assert list(np.nonzero(m[1])[0]) == [4]         # class 2 -> token 4
    assert m.dtype == np.bool_


def test_build_class_token_masks_empty_class_is_all_false():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    m = build_class_token_masks(tcid, num_classes=2, xp=np)
    assert m[0].any() and not m[1].any()


def _random_case(rng, n, q=40, c=2):
    tcid = np.zeros(256, np.int64)
    tcid[1] = 1; tcid[2] = 1
    if c >= 2:
        tcid[4] = 2
    logits = rng.normal(0, 3, size=(n, q, 256)).astype(np.float32)
    boxes = rng.uniform(0.15, 0.85, size=(n, q, 4)).astype(np.float32)
    boxes[..., 2:] *= 0.2                      # keep w/h small so boxes stay in frame
    return tcid, logits, boxes


def test_batch_matches_per_image_reference():
    """The batched decode + threshold must equal looping the per-image gdino_postprocess."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5):
        for thr in (0.3, 0.5, 0.9):
            tcid, logits, boxes = _random_case(rng, n)
            hw = [(100 + 7 * i, 200 + 11 * i) for i in range(n)]      # distinct per camera
            masks = build_class_token_masks(tcid, 2, xp=np)
            xyxy, bcls, bscore = gdino_postprocess_batch(logits, boxes, masks, hw, xp=np)
            for i in range(n):
                exp_bx, exp_cls, exp_sc = gdino_postprocess(
                    logits[i], boxes[i], tcid, 2, thr, hw[i][0], hw[i][1], xp=np)
                keep = np.nonzero(bscore[i] > thr)[0]
                assert list(bcls[i][keep]) == list(exp_cls), (n, thr, i)
                assert np.allclose(bscore[i][keep], exp_sc, atol=1e-6), (n, thr, i)
                assert np.allclose(xyxy[i][keep], exp_bx, atol=1e-3), (n, thr, i)


def test_batch_handles_zero_detections():
    tcid = np.zeros(256, np.int64); tcid[1] = 1
    logits = np.full((2, 5, 256), _sig_inv(0.05), np.float32)
    boxes = np.tile(np.array([0.5, 0.5, 0.1, 0.1], np.float32), (2, 5, 1))
    masks = build_class_token_masks(tcid, 1, xp=np)
    xyxy, bcls, bscore = gdino_postprocess_batch(logits, boxes, masks, [(100, 100)] * 2, xp=np)
    assert xyxy.shape == (2, 5, 4) and bcls.shape == (2, 5)
    assert not (bscore > 0.3).any()


def test_batch_empty_class_scores_zero_like_reference():
    """A prompt with no tokens (e.g. dropped by a remap) must score 0, never win argmax."""
    tcid = np.zeros(256, np.int64); tcid[1] = 1          # class 2 has no tokens
    logits = np.full((1, 3, 256), _sig_inv(0.02), np.float32)
    logits[0, 0, 1] = _sig_inv(0.95)
    masks = build_class_token_masks(tcid, 2, xp=np)
    _, bcls, bscore = gdino_postprocess_batch(logits, boxes=np.full((1, 3, 4), 0.5, np.float32),
                                              class_masks=masks, img_hw=[(10, 10)], xp=np)
    assert bcls[0, 0] == 1 and abs(float(bscore[0, 0]) - 0.95) < 1e-4


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
