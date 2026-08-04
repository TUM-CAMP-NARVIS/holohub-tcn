# SPDX-License-Identifier: Apache-2.0
"""Host tests for the baked->active Grounding DINO prompt remap (numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import build_prompt_remap


def _tcid():
    """token_class_ids for baked prompts ["floor", "person"]: class 1 -> tokens 1,2; class 2 -> 4."""
    t = np.zeros(256, np.int64); t[1] = 1; t[2] = 1; t[4] = 2
    return t


def test_identity_is_a_noop():
    remap = build_prompt_remap(["floor", "person"], ["floor", "person"])
    assert list(remap) == [0, 1, 2]
    assert np.array_equal(remap[_tcid()], _tcid())


def test_normalisation_ignores_case_and_whitespace():
    remap = build_prompt_remap(["floor", "person"], ["  Floor ", "PERSON"])
    assert list(remap) == [0, 1, 2]


def test_reorder_permutes_class_ids():
    remap = build_prompt_remap(["floor", "person"], ["person", "floor"])
    assert list(remap) == [0, 2, 1]           # baked floor(1) -> 2, baked person(2) -> 1
    out = remap[_tcid()]
    assert list(np.nonzero(out == 2)[0]) == [1, 2]   # floor tokens now class 2
    assert list(np.nonzero(out == 1)[0]) == [4]      # person tokens now class 1


def test_subset_zeroes_the_dropped_class_tokens():
    remap = build_prompt_remap(["floor", "person"], ["person"])
    assert list(remap) == [0, 0, 1]
    out = remap[_tcid()]
    assert not (out == 2).any()                       # no class 2 left
    assert list(np.nonzero(out == 1)[0]) == [4]       # person became class 1
    assert out[1] == 0 and out[2] == 0                # floor tokens dropped to background


def test_unbaked_term_raises_and_names_the_baked_set():
    try:
        build_prompt_remap(["floor", "person"], ["floor", "robot"])
    except ValueError as e:
        assert "robot" in str(e) and "floor" in str(e)
        return
    raise AssertionError("expected ValueError for an unbaked prompt")


def test_duplicate_after_normalisation_raises():
    try:
        build_prompt_remap(["floor", "person"], ["floor", "Floor"])
    except ValueError as e:
        assert "duplicate" in str(e).lower()
        return
    raise AssertionError("expected ValueError for duplicate prompts")


def test_empty_active_set_raises():
    try:
        build_prompt_remap(["floor", "person"], [])
    except ValueError as e:
        assert "empty" in str(e).lower()
        return
    raise AssertionError("expected ValueError for an empty prompt set")


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
