# SPDX-License-Identifier: Apache-2.0
"""Host-runnable tests for the panoptic-paint reformulation (design
docs/specs/2026-08-10-panoptic-cuda-design.md, §1).

The existing `build_panoptic_map` (numpy path, `xp=np`) is the ORACLE. These tests assert that

    paint_panoptic_np(masks, *plan_panoptic_paint(labels, scores, cmap, xp=np), H, W)

is byte-identical (`np.array_equal`) to `build_panoptic_map(masks, labels, scores, cmap, H, W,
xp=np)` for randomised inputs, including tied scores -- the case the priority reformulation
exists to get right. This is the highest-value test in the whole task: it validates the
reformulation, not just the numpy transcription of the kernel.

Compatible with pytest; also runnable via the __main__ block, matching the other tests in this
directory (test_langsam_multicam.py etc.).
"""

import os
import sys

import numpy as np

# Make the app's python/ dir importable regardless of how the tests are launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsam_helpers import (
    class_id_map,
    build_panoptic_map,
    plan_panoptic_paint,
    paint_panoptic_np,
    PANOPTIC_INSTANCE_MASK,
)


def _oracle_and_candidate(masks, labels, scores, cmap, H, W):
    oracle = build_panoptic_map(masks, labels, scores, cmap, H, W, xp=np)
    values, priorities = plan_panoptic_paint(labels, scores, cmap, xp=np)
    candidate = paint_panoptic_np(masks, values, priorities, H, W)
    return oracle, candidate


def _assert_matches_oracle(masks, labels, scores, cmap, H, W, msg=""):
    oracle, candidate = _oracle_and_candidate(masks, labels, scores, cmap, H, W)
    assert candidate.dtype == np.uint16, f"{msg}: dtype {candidate.dtype}"
    assert oracle.dtype == np.uint16, f"{msg}: oracle dtype {oracle.dtype}"
    assert np.array_equal(oracle, candidate), (
        f"{msg}: mismatch, {int(np.sum(oracle != candidate))} differing pixels "
        f"out of {oracle.size}"
    )


def _random_detections(rng, H, W, M, prompts, mask_density=0.35, unknown_prob=0.0,
                       tie_prob=0.0, empty_mask_prob=0.0):
    """Randomised (masks, labels, scores) for M detections over an HxW map."""
    labels = []
    for _ in range(M):
        if unknown_prob > 0 and rng.random() < unknown_prob:
            labels.append("totally_unknown_thing")
        else:
            labels.append(str(rng.choice(prompts)))
    scores = rng.random(M)
    if tie_prob > 0 and M > 1:
        # Force some scores to collide with a neighbour's, on purpose -- ties are the case
        # the priority reformulation exists to get right.
        for i in range(1, M):
            if rng.random() < tie_prob:
                scores[i] = scores[i - 1]
    masks = (rng.random((M, H, W)) < mask_density).astype(np.uint8)
    if empty_mask_prob > 0:
        for j in range(M):
            if rng.random() < empty_mask_prob:
                masks[j] = 0
    return masks, labels, scores


# --- overlapping masks: higher score must win -------------------------------------------------

def test_overlap_highest_score_wins_simple():
    cmap = class_id_map(["person"])
    H = W = 4
    a = np.zeros((H, W), np.uint8); a[0:3, 0:3] = 1   # low score
    b = np.zeros((H, W), np.uint8); b[1:4, 1:4] = 1   # high score, must win the overlap
    masks = np.stack([a, b])
    labels = ["person", "person"]
    scores = np.array([0.2, 0.95])
    _assert_matches_oracle(masks, labels, scores, cmap, H, W, "overlap_simple")


def test_overlap_highest_score_wins_random_trials():
    rng = np.random.default_rng(1234)
    prompts = ["floor", "person", "robot", "tool"]
    cmap = class_id_map(prompts)
    for trial in range(30):
        H = int(rng.integers(4, 20))
        W = int(rng.integers(4, 20))
        M = int(rng.integers(1, 15))
        masks, labels, scores = _random_detections(rng, H, W, M, prompts, mask_density=0.4)
        _assert_matches_oracle(masks, labels, scores, cmap, H, W, f"overlap_trial_{trial}")


# --- tied scores: the case the priority reformulation exists to get right ----------------------

def test_tied_scores_all_identical():
    cmap = class_id_map(["person"])
    H = W = 6
    rng = np.random.default_rng(42)
    M = 8
    masks = (rng.random((M, H, W)) < 0.5).astype(np.uint8)
    labels = ["person"] * M
    scores = np.full(M, 0.5)   # every score identical: ties everywhere
    _assert_matches_oracle(masks, labels, scores, cmap, H, W, "tied_all_identical")


def test_tied_scores_random_trials():
    rng = np.random.default_rng(777)
    prompts = ["floor", "person", "robot"]
    cmap = class_id_map(prompts)
    for trial in range(30):
        H = int(rng.integers(4, 16))
        W = int(rng.integers(4, 16))
        M = int(rng.integers(2, 20))
        masks, labels, scores = _random_detections(
            rng, H, W, M, prompts, mask_density=0.4, tie_prob=0.5)
        _assert_matches_oracle(masks, labels, scores, cmap, H, W, f"tied_trial_{trial}")


def test_tied_scores_two_classes_same_pixel_deterministic():
    # Two detections, SAME score, overlapping mask, DIFFERENT classes: argsort must break the
    # tie the same (stable) way in both the oracle and the candidate, or this fails.
    cmap = class_id_map(["a", "b"])
    H = W = 3
    m0 = np.ones((H, W), np.uint8)
    m1 = np.ones((H, W), np.uint8)
    masks = np.stack([m0, m1])
    labels = ["a", "b"]
    scores = np.array([0.5, 0.5])
    _assert_matches_oracle(masks, labels, scores, cmap, H, W, "tied_two_classes")


# --- unknown labels mixed with known ------------------------------------------------------------

def test_unknown_labels_mixed_with_known():
    rng = np.random.default_rng(99)
    prompts = ["floor", "person"]
    cmap = class_id_map(prompts)
    for trial in range(20):
        H = int(rng.integers(4, 14))
        W = int(rng.integers(4, 14))
        M = int(rng.integers(3, 15))
        masks, labels, scores = _random_detections(
            rng, H, W, M, prompts, mask_density=0.4, unknown_prob=0.4)
        _assert_matches_oracle(masks, labels, scores, cmap, H, W, f"unknown_trial_{trial}")


def test_all_labels_unknown_is_all_background():
    cmap = class_id_map(["person"])
    H = W = 5
    masks = np.ones((3, H, W), np.uint8)
    labels = ["lamp", "chair", "table"]
    scores = np.array([0.1, 0.5, 0.9])
    oracle, candidate = _oracle_and_candidate(masks, labels, scores, cmap, H, W)
    assert int(oracle.max()) == 0
    assert np.array_equal(oracle, candidate)


# --- a detection whose mask is entirely empty ---------------------------------------------------

def test_empty_mask_detection_never_paints():
    cmap = class_id_map(["person"])
    H = W = 5
    empty = np.zeros((H, W), np.uint8)
    full = np.zeros((H, W), np.uint8); full[1:4, 1:4] = 1
    masks = np.stack([empty, full])
    labels = ["person", "person"]
    # Give the EMPTY mask the highest score, so if it painted anything the bug would show.
    scores = np.array([0.99, 0.1])
    _assert_matches_oracle(masks, labels, scores, cmap, H, W, "empty_mask_highest_score")


def test_empty_mask_detections_random_trials():
    rng = np.random.default_rng(55)
    prompts = ["floor", "person", "robot"]
    cmap = class_id_map(prompts)
    for trial in range(20):
        H = int(rng.integers(4, 14))
        W = int(rng.integers(4, 14))
        M = int(rng.integers(2, 12))
        masks, labels, scores = _random_detections(
            rng, H, W, M, prompts, mask_density=0.4, empty_mask_prob=0.3)
        _assert_matches_oracle(masks, labels, scores, cmap, H, W, f"empty_mask_trial_{trial}")


# --- M == 0 -> all-zero map ----------------------------------------------------------------------

def test_zero_detections_is_all_zero_map():
    cmap = class_id_map(["person"])
    H, W = 7, 9
    masks = np.empty((0, H, W), np.uint8)
    labels = []
    scores = np.array([])
    oracle, candidate = _oracle_and_candidate(masks, labels, scores, cmap, H, W)
    assert oracle.shape == (H, W) and candidate.shape == (H, W)
    assert int(oracle.max()) == 0 and int(candidate.max()) == 0
    assert np.array_equal(oracle, candidate)
    assert candidate.dtype == np.uint16


def test_zero_detections_via_none_masks():
    # This is exactly how the production call sites build the "no detections on this camera"
    # default map: build_panoptic_map(None, [], None, cmap, h, w, xp=cp).
    cmap = class_id_map(["person"])
    H, W = 5, 5
    oracle = build_panoptic_map(None, [], None, cmap, H, W, xp=np)
    values, priorities = plan_panoptic_paint([], None, cmap, xp=np)
    candidate = paint_panoptic_np(None, values, priorities, H, W)
    assert values.shape == (0,) and priorities.shape == (0,)
    assert np.array_equal(oracle, candidate)


# --- more instances of one class than PANOPTIC_INSTANCE_MASK allows (clamp path) -----------------

def test_instance_count_clamped_to_panoptic_instance_mask():
    cmap = class_id_map(["person"])
    H = W = 20
    # Many more detections of the same class than PANOPTIC_INSTANCE_MASK (255) allows, each a
    # single-pixel mask at a distinct location so nothing overlaps and every instance survives
    # to be checked individually.
    M = PANOPTIC_INSTANCE_MASK + 20
    rng = np.random.default_rng(3)
    masks = np.zeros((M, H, W), np.uint8)
    coords = [(i % H, (i * 7) % W) for i in range(M)]  # distinct-ish, may collide occasionally
    for j, (r, c) in enumerate(coords):
        masks[j, r, c] = 1
    labels = ["person"] * M
    # Descending score order 0..M-1 so detection j is the (j+1)-th most confident (instance j+1,
    # clamped at PANOPTIC_INSTANCE_MASK).
    scores = np.linspace(1.0, 0.0, M)
    _assert_matches_oracle(masks, labels, scores, cmap, H, W, "instance_clamp")


def test_instance_count_clamp_random_trials():
    rng = np.random.default_rng(4321)
    prompts = ["thing"]
    cmap = class_id_map(prompts)
    for trial in range(5):
        H = W = 12
        M = PANOPTIC_INSTANCE_MASK + int(rng.integers(1, 30))
        masks, labels, scores = _random_detections(
            rng, H, W, M, prompts, mask_density=0.05, tie_prob=0.2)
        _assert_matches_oracle(masks, labels, scores, cmap, H, W, f"instance_clamp_trial_{trial}")


# --- dtype is uint16 in every case (folded into _assert_matches_oracle / explicit checks above,
# and re-asserted here as a standalone check across a spread of shapes/M). ------------------------

def test_dtype_is_uint16_across_varied_shapes():
    cmap = class_id_map(["a", "b", "c"])
    rng = np.random.default_rng(8)
    for H, W, M in [(1, 1, 0), (1, 1, 1), (5, 3, 4), (33, 17, 0), (10, 10, 50)]:
        masks, labels, scores = _random_detections(rng, H, W, M, ["a", "b", "c"], mask_density=0.5)
        values, priorities = plan_panoptic_paint(labels, scores, cmap, xp=np)
        candidate = paint_panoptic_np(masks if M else None, values, priorities, H, W)
        assert candidate.dtype == np.uint16, (H, W, M, candidate.dtype)
        assert values.dtype == np.uint16, (H, W, M, values.dtype)


# --- priorities sanity: must be a permutation of 0..M-1 (uniqueness is what makes the tie
# argument in the design doc correct) --------------------------------------------------------------

def test_priorities_are_a_permutation_of_range_m():
    rng = np.random.default_rng(21)
    prompts = ["a", "b"]
    cmap = class_id_map(prompts)
    for trial in range(10):
        M = int(rng.integers(0, 25))
        labels = [str(rng.choice(prompts)) for _ in range(M)]
        scores = rng.random(M)
        _values, priorities = plan_panoptic_paint(labels, scores, cmap, xp=np)
        assert sorted(priorities.tolist()) == list(range(M)), (trial, M, priorities.tolist())


if __name__ == "__main__":
    # Plain-python runner for hosts without pytest.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
