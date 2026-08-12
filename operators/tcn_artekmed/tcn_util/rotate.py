"""180-degree rotation, as a plain function so it can be used inline as well as as an operator.

Kept dependency-free (works on numpy or cupy, imports neither) so it is host-testable and so callers
inside another operator's `compute()` can use it without going through the graph.

`RotateImage180Op` in this package is the graph-node form; `tcn_langsam` calls the function directly,
because it rotates individual cameras out of a multi-camera entity and inserting a graph node per
camera would mean splitting and re-merging the entity for no gain.
"""


def rotate180(arr):
    """Rotate an array 180 degrees in its two leading (spatial) axes.

    Works for `(H, W)` maps and `(H, W, C)` images. The channel axis is deliberately untouched --
    reversing it would swap colour channels, not orientation.

    Returns a reversed **view**, not a copy: callers that need contiguous memory (anything handing the
    result to a tensor consumer) must ask for it with `ascontiguousarray`.

    A 180-degree rotation is its own inverse and involves no resampling or interpolation, which is
    what makes "rotate in, rotate the result back out" exactly reversible. No other angle has that
    property with array slicing alone.
    """
    return arr[::-1, ::-1]
