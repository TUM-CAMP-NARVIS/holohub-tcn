# SPDX-License-Identifier: Apache-2.0
"""Host tests for the gpu_workers topology helpers (numpy-free)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import (resolve_workers, worker_batch, distinct_batches,
                             worker_engine_path)

CFG = {"workers": [
    {"device": 0, "cameras": ["camera01_colorimage", "camera02_colorimage"]},
    {"device": 1, "cameras": ["camera03_colorimage", "camera04_colorimage",
                              "camera05_colorimage"]},
]}
ALL = ["camera0%d_colorimage" % i for i in range(1, 6)]


def test_resolve_workers_reads_the_node():
    w = resolve_workers(CFG, ALL)
    assert [x["device"] for x in w] == [0, 1]
    assert len(w[0]["cameras"]) == 2 and len(w[1]["cameras"]) == 3


def test_empty_config_falls_back_to_one_worker_on_device_0():
    for cfg in (None, {}, {"workers": []}):
        w = resolve_workers(cfg, ALL)
        assert w == [{"device": 0, "cameras": ALL}], cfg


def test_batch_is_derived_from_the_camera_count():
    w = resolve_workers(CFG, ALL)
    assert worker_batch(w[0]) == 2 and worker_batch(w[1]) == 3


def test_distinct_batches_are_sorted_and_deduplicated():
    assert distinct_batches(resolve_workers(CFG, ALL)) == [2, 3]
    same = [{"device": 0, "cameras": ["a", "b"]}, {"device": 1, "cameras": ["c", "d"]}]
    assert distinct_batches(same) == [2]


def test_engine_path_substitutes_batch():
    assert worker_engine_path("/m/gdino_b{batch}_tf32.engine", 2) == "/m/gdino_b2_tf32.engine"


def test_engine_path_without_placeholder_is_a_noop():
    """Lets a single-engine config keep working while only some batches exist."""
    p = "/m/gdino_b3_tf32.engine"
    assert worker_engine_path(p, 2) == p


def test_engine_path_rejects_a_still_templated_result():
    try:
        worker_engine_path("/m/gdino_{size}_b{batch}.engine", 2)
    except ValueError as e:
        assert "size" in str(e) or "{" in str(e)
        return
    raise AssertionError("expected ValueError for an unresolved placeholder")


def test_engine_path_accepts_none():
    assert worker_engine_path(None, 2) is None


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
