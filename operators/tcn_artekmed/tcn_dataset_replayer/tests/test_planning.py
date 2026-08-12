"""Host tests for `_planning.py` (pure numpy-free frame-plan and timestamp arithmetic).

Run directly: `python3 operators/tcn_artekmed/tcn_dataset_replayer/tests/test_planning.py`

Every expectation here is built independently of `plan_frame_sequence`'s implementation
(hand-written lists, or `itertools.cycle`/`islice` for the repeat case) -- never by restating
the slicing logic under test.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _planning import (DEFAULT_INTERVAL_NS, is_strictly_increasing, loop_span_ns,
                       plan_frame_sequence)

CONTIGUOUS = [0, 1, 2, 3, 4, 5]
# Non-contiguous frame numbers: an index/number confusion (e.g. returning the selected
# *indices* instead of the dataset frame *numbers*) fails loudly against this list, because
# index i and frame number NONCONTIG[i] never coincide except at i == 0.
NONCONTIG = [10, 20, 30, 40]


def test_defaults_return_every_frame_in_order():
    assert plan_frame_sequence(CONTIGUOUS) == [0, 1, 2, 3, 4, 5]


def test_count_truncates_the_selection():
    assert plan_frame_sequence(CONTIGUOUS, count=3) == [0, 1, 2]


def test_step_and_count_combine():
    assert plan_frame_sequence(CONTIGUOUS, step=2, count=3) == [0, 2, 4]


def test_loop_wraps_to_the_requested_tick_count():
    # Independent oracle: cycle the *expected* selection with itertools, not the code under test.
    expected = list(itertools.islice(itertools.cycle([0, 1, 2]), 8))
    assert expected == [0, 1, 2, 0, 1, 2, 0, 1]  # sanity-check the oracle itself
    got = plan_frame_sequence(CONTIGUOUS, count=3, loop=True, ticks=8)
    assert got == expected


def test_loop_false_truncates_rather_than_pads():
    # Selection is all 6 frames; loop=False must never manufacture extra entries even though
    # ticks=100 asks for more than exist.
    got = plan_frame_sequence(CONTIGUOUS, loop=False, ticks=100)
    assert got == [0, 1, 2, 3, 4, 5]
    assert len(got) == 6


def test_start_is_an_index_not_a_frame_number():
    # start=1 on NONCONTIG must select frame number 20 (index 1), never frame number 1.
    got = plan_frame_sequence(NONCONTIG, start=1, count=2)
    assert got == [20, 30]
    assert 1 not in got


def test_returned_values_are_frame_numbers_on_a_noncontiguous_list():
    got = plan_frame_sequence(NONCONTIG, start=0, count=None, step=1, loop=False, ticks=None)
    assert got == [10, 20, 30, 40]


def test_start_and_step_combine_on_noncontiguous_list():
    got = plan_frame_sequence(NONCONTIG, start=1, step=2, count=None)
    assert got == [20, 40]


def test_loop_wraps_on_a_noncontiguous_selection():
    expected = list(itertools.islice(itertools.cycle([10, 20]), 5))
    assert expected == [10, 20, 10, 20, 10]
    got = plan_frame_sequence(NONCONTIG, start=0, count=2, loop=True, ticks=5)
    assert got == expected


def test_start_beyond_the_end_raises_value_error():
    try:
        plan_frame_sequence(CONTIGUOUS, start=6)
    except ValueError as e:
        assert "start" in str(e).lower()
        return
    raise AssertionError("expected ValueError for start beyond the end")


def test_count_zero_raises_value_error():
    try:
        plan_frame_sequence(CONTIGUOUS, count=0)
    except ValueError as e:
        assert "count" in str(e).lower()
        return
    raise AssertionError("expected ValueError for count=0")


def test_empty_frame_numbers_raises_value_error():
    try:
        plan_frame_sequence([])
    except ValueError as e:
        assert "frame_numbers" in str(e).lower() or "empty" in str(e).lower()
        return
    raise AssertionError("expected ValueError for an empty frame_numbers list")


def test_step_below_one_raises_value_error():
    for bad_step in (0, -1):
        try:
            plan_frame_sequence(CONTIGUOUS, step=bad_step)
        except ValueError as e:
            assert "step" in str(e).lower()
            continue
        raise AssertionError(f"expected ValueError for step={bad_step}")


def test_count_below_zero_raises_value_error():
    try:
        plan_frame_sequence(CONTIGUOUS, count=-1)
    except ValueError as e:
        assert "count" in str(e).lower()
        return
    raise AssertionError("expected ValueError for count=-1")


def test_no_loop_no_ticks_returns_full_selection_unpadded():
    got = plan_frame_sequence(CONTIGUOUS, count=3, loop=False, ticks=None)
    assert got == [0, 1, 2]


def test_loop_with_ticks_shorter_than_selection_truncates():
    # loop=True but ticks smaller than one full cycle: still just the first `ticks` entries.
    got = plan_frame_sequence(CONTIGUOUS, count=4, loop=True, ticks=2)
    assert got == [0, 1]



# --- loop_span_ns / is_strictly_increasing --------------------------------------------------------
# The property that matters: replaying the plan with `k * span` added must produce a strictly
# increasing timestamp sequence across passes. Tests assert that directly where they can, rather
# than restating the extent-plus-interval formula.

MS = 1_000_000


def _replayed(stamps, span, passes):
    """The timestamps a looping replay would emit over `passes` passes."""
    return [t + k * span for k in range(passes) for t in stamps]


def test_span_keeps_uniform_cadence_strictly_increasing_across_passes():
    stamps = [0, 33 * MS, 66 * MS, 99 * MS]
    assert is_strictly_increasing(_replayed(stamps, loop_span_ns(stamps), passes=4))


def test_span_keeps_irregular_cadence_strictly_increasing_across_passes():
    # Gaps 10, 5, 40 ms: the span must clear the LARGEST stamp, using the smallest gap as the
    # step, so an uneven export loops just as safely as an even one.
    stamps = [100 * MS, 110 * MS, 115 * MS, 155 * MS]
    assert is_strictly_increasing(_replayed(stamps, loop_span_ns(stamps), passes=3))


def test_span_equals_extent_plus_smallest_gap():
    stamps = [1000, 1003, 1010]            # extent 10, smallest gap 3
    assert loop_span_ns(stamps) == 13


def test_span_is_order_independent():
    stamps = [50 * MS, 10 * MS, 30 * MS]
    assert loop_span_ns(stamps) == loop_span_ns(sorted(stamps))


def test_single_stamp_falls_back_to_default_interval():
    assert loop_span_ns([12345]) == DEFAULT_INTERVAL_NS


def test_empty_falls_back_to_default_interval():
    assert loop_span_ns([]) == DEFAULT_INTERVAL_NS


def test_all_stamps_equal_falls_back_to_default_interval():
    # Extent 0 and no positive gap: the span is the fallback interval alone, which is still enough
    # to separate one pass from the next.
    assert loop_span_ns([7, 7, 7]) == DEFAULT_INTERVAL_NS
    assert is_strictly_increasing([7 + k * loop_span_ns([7, 7, 7]) for k in range(3)])


def test_custom_default_interval_is_honoured():
    assert loop_span_ns([1], default_interval_ns=500) == 500


def test_strictly_increasing_detects_plan_order_not_sorted_order():
    # The point of checking in plan order: this sequence sorts fine but replays out of order.
    assert not is_strictly_increasing([30 * MS, 10 * MS, 20 * MS])
    assert is_strictly_increasing([10 * MS, 20 * MS, 30 * MS])


def test_strictly_increasing_rejects_duplicates_and_accepts_short_sequences():
    assert not is_strictly_increasing([5, 5])
    assert is_strictly_increasing([5])
    assert is_strictly_increasing([])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, repr(e))
        except Exception as e:
            bad += 1; print("ERROR", fn.__name__, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
