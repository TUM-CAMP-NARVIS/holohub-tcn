# SPDX-License-Identifier: Apache-2.0
"""Host-runnable unit tests for the pure multi-camera LangSAM helpers.

Imports from ``langsam_helpers`` (numpy-only) rather than ``langsam_common`` so they run on
a host without cupy/torch/sam2. Compatible with pytest; also runnable via the __main__ block.
"""

import os
import sys

import numpy as np

# Make the app's python/ dir importable regardless of how the tests are launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsam_helpers import (
    resolve_workers,
    class_id_map,
    class_id_for_label,
    build_label_map,
    build_panoptic_map,
    panoptic_class,
    panoptic_instance,
    mask_name,
    plan_batched_decode,
)

ALL = ["camera01_colorimage", "camera02_colorimage", "camera03_colorimage"]


def test_empty_config_defaults_to_single_gpu0_worker():
    assert resolve_workers(None, ALL) == [{"device": 0, "cameras": ALL}]
    assert resolve_workers({}, ALL) == [{"device": 0, "cameras": ALL}]
    assert resolve_workers({"workers": []}, ALL) == [{"device": 0, "cameras": ALL}]


def test_explicit_workers_preserved_and_device_coerced():
    cfg = {"workers": [
        {"device": "0", "cameras": ["camera01_colorimage"]},
        {"device": 1, "cameras": ["camera02_colorimage", "camera03_colorimage"]},
    ]}
    assert resolve_workers(cfg, ALL) == [
        {"device": 0, "cameras": ["camera01_colorimage"]},
        {"device": 1, "cameras": ["camera02_colorimage", "camera03_colorimage"]},
    ]


def test_class_id_map_is_one_based():
    assert class_id_map(["floor", "person", "robot"]) == {"floor": 1, "person": 2, "robot": 3}


def test_class_id_for_label_exact_substring_and_unknown():
    cmap = class_id_map(["floor", "person", "robot"])
    assert class_id_for_label("Floor", cmap) == 1       # case-insensitive exact
    assert class_id_for_label("a person", cmap) == 2    # substring (prompt in label)
    assert class_id_for_label("lamp", cmap) == 0        # unknown -> background


def test_build_label_map_higher_score_wins_overlap():
    cmap = class_id_map(["floor", "person"])
    H = W = 4
    m_floor = np.zeros((H, W), bool); m_floor[0:3, 0:3] = True    # floor, low score
    m_person = np.zeros((H, W), bool); m_person[1:4, 1:4] = True  # person, high score
    masks = np.stack([m_floor, m_person])
    lm = build_label_map(masks, ["floor", "person"], np.array([0.5, 0.9]), cmap, H, W, xp=np)
    assert lm.dtype == np.uint8
    assert lm[0, 0] == 1       # floor only
    assert lm[3, 3] == 2       # person only
    assert lm[2, 2] == 2       # overlap -> higher score (person) wins
    assert lm[3, 0] == 0       # background


def test_build_label_map_empty_is_all_background():
    cmap = class_id_map(["floor"])
    lm = build_label_map(np.empty((0, 4, 4)), [], np.array([]), cmap, 4, 4, xp=np)
    assert lm.shape == (4, 4) and int(lm.max()) == 0


def test_build_label_map_skips_unknown_labels():
    cmap = class_id_map(["floor"])
    H = W = 3
    mask = np.ones((1, H, W), bool)
    lm = build_label_map(mask, ["lamp"], np.array([0.9]), cmap, H, W, xp=np)  # lamp unknown
    assert int(lm.max()) == 0  # nothing painted


def test_build_panoptic_map_packs_class_and_per_class_instances():
    cmap = class_id_map(["floor", "person"])
    H = W = 6
    # two person instances (disjoint) + one floor
    p1 = np.zeros((H, W), bool); p1[0:2, 0:2] = True
    p2 = np.zeros((H, W), bool); p2[0:2, 4:6] = True
    fl = np.zeros((H, W), bool); fl[4:6, 0:2] = True
    masks = np.stack([p1, p2, fl])
    labels = ["person", "person", "floor"]
    scores = np.array([0.9, 0.6, 0.8])       # p1 most confident person, then p2
    pm = build_panoptic_map(masks, labels, scores, cmap, H, W, xp=np)
    assert pm.dtype == np.uint16
    # class ids in high byte: person=2, floor=1
    assert panoptic_class(pm[0, 0]) == 2 and panoptic_instance(pm[0, 0]) == 1   # top person
    assert panoptic_class(pm[0, 4]) == 2 and panoptic_instance(pm[0, 4]) == 2   # 2nd person
    assert panoptic_class(pm[4, 0]) == 1 and panoptic_instance(pm[4, 0]) == 1   # floor inst 1
    assert pm[3, 3] == 0                                                        # background


def test_build_panoptic_map_overlap_highest_score_wins():
    cmap = class_id_map(["person"])
    H = W = 4
    a = np.zeros((H, W), bool); a[0:3, 0:3] = True   # score 0.4
    b = np.zeros((H, W), bool); b[1:4, 1:4] = True   # score 0.9 (wins overlap, instance 1)
    pm = build_panoptic_map(np.stack([a, b]), ["person", "person"], np.array([0.4, 0.9]),
                            cmap, H, W, xp=np)
    assert panoptic_instance(pm[2, 2]) == 1          # overlap -> most-confident instance (b)
    assert panoptic_class(pm[2, 2]) == 1


def test_maskname_strips_colorimage_suffix():
    assert mask_name("camera01_colorimage") == "camera01_mask"


def test_maskname_without_colorimage_suffix_just_appends():
    assert mask_name("camera01") == "camera01_mask"


def test_plan_batched_decode_typical():
    total, box_img = plan_batched_decode([2, 3])
    assert total == 5
    assert box_img.tolist() == [0, 0, 1, 1, 1]


def test_plan_batched_decode_single_image():
    total, box_img = plan_batched_decode([4])
    assert total == 4
    assert box_img.tolist() == [0, 0, 0, 0]


def test_plan_batched_decode_zero_in_middle():
    total, box_img = plan_batched_decode([2, 0, 1])
    assert total == 3
    assert box_img.tolist() == [0, 0, 2]


def test_plan_batched_decode_all_zeros():
    total, box_img = plan_batched_decode([0, 0])
    assert total == 0
    assert box_img.tolist() == []


def test_plan_batched_decode_empty():
    total, box_img = plan_batched_decode([])
    assert total == 0
    assert box_img.tolist() == []


def test_plan_batched_decode_negative_raises():
    try:
        plan_batched_decode([2, -1])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plan_batched_decode_dtype_is_int64():
    _, box_img = plan_batched_decode([2, 3])
    assert box_img.dtype == np.int64


def test_plan_batched_decode_matches_independent_oracle():
    # Independent oracle: NOT a restatement of the implementation, just the definition of
    # "which image did box j come from" spelled out the obvious (slow) way.
    for counts in ([1], [3], [2, 3], [1, 1, 1], [5, 0, 2, 0, 1], [0], [7, 4, 9, 1]):
        total, box_img = plan_batched_decode(counts)
        oracle = np.concatenate([np.full(c, i) for i, c in enumerate(counts)]) \
            if sum(counts) > 0 else np.empty(0, dtype=np.int64)
        assert total == sum(counts)
        assert box_img.tolist() == oracle.tolist(), (counts, box_img.tolist(), oracle.tolist())


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
