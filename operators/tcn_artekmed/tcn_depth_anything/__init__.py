"""Depth-Anything monocular depth sub-flows (V2 and V3).

Each variant is a self-contained subgraph: preprocess -> TRT inference -> postprocess to metric
depth, optionally scaled by a camera's real intrinsics from the device context. They share no code
with each other on purpose -- the two models differ in output layout and in how metric scale is
recovered, and collapsing them into one parameterised subgraph made both harder to read.

Imports are lazy so that pulling in one variant does not construct the other's TRT stack.
"""

__all__ = [
    "DA2PostprocessorOp",
    "DA2MetricProcessingSubgraph",
    "DA3PostprocessorOp",
    "DA3MetricProcessingSubgraph",
]


def __getattr__(name):
    if name in ("DA2PostprocessorOp", "DA2MetricProcessingSubgraph"):
        from . import da2
        return getattr(da2, name)
    if name in ("DA3PostprocessorOp", "DA3MetricProcessingSubgraph"):
        from . import da3
        return getattr(da3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
