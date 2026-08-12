"""Pure, numpy-free frame-plan arithmetic for `TcnDatasetReplayerOp`.

Kept separate from `dataset_replayer_op.py` so it can be imported and tested on a host that
has neither `holoscan` nor `cupy` nor `artekmed_dataset_reader` installed -- this is where
off-by-one / index-vs-frame-number bugs live, and they are cheap to catch here.
"""
from typing import List, Optional, Sequence


def plan_frame_sequence(
    frame_numbers: Sequence[int],
    start: int = 0,
    count: Optional[int] = None,
    step: int = 1,
    loop: bool = True,
    ticks: Optional[int] = None,
) -> List[int]:
    """Source frame numbers a replay run will emit, in order.

    `frame_numbers` is the dataset's available frames (ascending). `start` is an INDEX into
    that list, not a frame number -- ``plan_frame_sequence([10, 20, 30], start=1)`` begins at
    frame number 20, not frame number 1. Returns the selected frames repeated to `ticks`
    entries when `loop`, else the selection truncated to at most `ticks` (never padded).

    Raises:
        ValueError: if `frame_numbers` is empty, `start` is out of range, `step` < 1, or
            `count` is not None and < 1.
    """
    frame_numbers = list(frame_numbers)
    if not frame_numbers:
        raise ValueError("plan_frame_sequence: frame_numbers is empty; dataset has no frames")
    if step < 1:
        raise ValueError(f"plan_frame_sequence: step must be >= 1, got {step}")
    if count is not None and count < 1:
        raise ValueError(f"plan_frame_sequence: count must be >= 1 or None, got {count}")
    if start < 0 or start >= len(frame_numbers):
        raise ValueError(
            f"plan_frame_sequence: start index {start} is out of range for "
            f"{len(frame_numbers)} available frame(s) (start is an INDEX, not a frame number)"
        )

    selection = frame_numbers[start::step]
    if count is not None:
        selection = selection[:count]

    if not selection:
        # Unreachable given the checks above (start is in range, so frame_numbers[start::step]
        # always yields at least one element), but guard anyway so a future refactor of the
        # bounds checks above cannot silently divide by zero below.
        raise ValueError(
            "plan_frame_sequence: selection is empty after applying start/step/count"
        )

    if ticks is None:
        return list(selection)

    if not loop:
        return list(selection[:ticks])

    n = len(selection)
    full_cycles, remainder = divmod(ticks, n)
    return selection * full_cycles + selection[:remainder]


#: Fallback inter-frame spacing (30 fps) when a cadence cannot be measured. Matches the synthetic
#: timestamps `TcnDatasetReplayerOp` invents for an export that carries none.
DEFAULT_INTERVAL_NS = 33_333_333


def loop_span_ns(stamps: Sequence[int], default_interval_ns: int = DEFAULT_INTERVAL_NS) -> int:
    """Nanoseconds to add per completed loop so replayed timestamps keep advancing.

    A consumer that keys on acquisition time treats a timestamp as a frame's identity, so it must
    reject one that does not advance -- two frames cannot BE the same frame. Replaying a dataset's
    real capture times verbatim therefore makes every pass after the first look like a repeat, and a
    looping run only ever synchronises pass 0.

    The span is the timestamps' extent plus one inter-frame interval, so the first frame of the next
    pass lands one interval after the last frame of this one, as if capture had simply continued.
    The interval is the smallest positive gap between consecutive stamps in time order, which is the
    dataset's own cadence; `default_interval_ns` covers the degenerate cases (fewer than two stamps,
    or all stamps equal).

    Order-independent: `stamps` may be in any order, since only the extent and the gaps matter.
    """
    ordered = sorted(stamps)
    if len(ordered) < 2:
        return default_interval_ns
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    interval = min(gaps) if gaps else default_interval_ns
    return (ordered[-1] - ordered[0]) + interval


def is_strictly_increasing(values: Sequence[int]) -> bool:
    """Whether `values` strictly increases in the order given.

    Checked in PLAN order, not sorted order: sorting is what hides the defect this detects -- a
    frame plan whose capture times disagree with its frame order. No loop offset can fix that, since
    the frames collide within a single pass.
    """
    return all(b > a for a, b in zip(values, values[1:]))
