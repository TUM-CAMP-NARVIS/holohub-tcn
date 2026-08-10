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
