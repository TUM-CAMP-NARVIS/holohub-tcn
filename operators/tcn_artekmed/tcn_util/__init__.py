"""
TCN utility operators with optional C++ acceleration.

When the C++ pybind11 operator bindings are available (built via holohub),
this module uses them with thin wrappers that preserve the pure-Python
calling convention.  Otherwise it falls back to the pure-Python operators.
"""

import logging as _logging

_log = _logging.getLogger(__name__)
_USE_CPP = {}

# ── StreamSplitterOp ──
try:
    from holohub.tcn_stream_splitter import TcnStreamSplitterOp as _CppStreamSplitterOp

    class StreamSplitterOp(_CppStreamSplitterOp):
        """C++ StreamSplitter with Python-compatible constructor."""
        def __init__(self, fragment, cuda_stream_pool, channel_names, *args, **kwargs):
            super().__init__(fragment, cuda_stream_pool, *args,
                             channel_names=channel_names, **kwargs)

    _USE_CPP["StreamSplitterOp"] = True
except ImportError:
    from .StreamSplitterOp import StreamSplitterOp
    _USE_CPP["StreamSplitterOp"] = False

# ── StreamMergerOp ──
try:
    from holohub.tcn_stream_merger import TcnStreamMergerOp as _CppStreamMergerOp

    class StreamMergerOp(_CppStreamMergerOp):
        """C++ StreamMerger with Python-compatible constructor."""
        def __init__(self, fragment, cuda_stream_pool, input_port_names,
                     input_message_name, output_message_name, fuse_buffers,
                     *args, **kwargs):
            super().__init__(fragment, cuda_stream_pool, *args,
                             input_port_names=input_port_names,
                             input_message_name=input_message_name,
                             output_message_name=output_message_name,
                             fuse_buffers=fuse_buffers, **kwargs)

    _USE_CPP["StreamMergerOp"] = True
except ImportError:
    from .StreamMergerOp import StreamMergerOp
    _USE_CPP["StreamMergerOp"] = False

# ── FlattenTensorOp ──
try:
    from holohub.tcn_flatten_tensor import TcnFlattenTensorOp as _CppFlattenTensorOp

    class FlattenTensorOp(_CppFlattenTensorOp):
        """C++ FlattenTensor with Python-compatible constructor.

        The C++ binding does not accept ``allocator`` as a keyword argument,
        so we move it into the positional resource args.
        """
        def __init__(self, fragment, *args, allocator=None, **kwargs):
            pos_args = list(args)
            if allocator is not None:
                pos_args.append(allocator)
            super().__init__(fragment, *pos_args, **kwargs)

    _USE_CPP["FlattenTensorOp"] = True
except ImportError:
    from .FlattenTensorOp import FlattenTensorOp
    _USE_CPP["FlattenTensorOp"] = False

# ── DepthImageMaxDistanceOp ──
try:
    from holohub.tcn_depthimage_max_distance import TcnDepthImageMaxDistanceOp as _CppMaxDistOp

    class DepthImageMaxDistanceOp(_CppMaxDistOp):
        """C++ DepthImageMaxDistance with Python-compatible constructor.

        Extracts ``cuda_stream_pool`` from kwargs and passes it as a positional
        resource arg (the C++ binding does not have a ``cuda_stream_pool`` kwarg).
        """
        def __init__(self, fragment, *args, cuda_stream_pool=None, **kwargs):
            pos_args = list(args)
            if cuda_stream_pool is not None:
                pos_args.append(cuda_stream_pool)
            super().__init__(fragment, *pos_args, **kwargs)

    _USE_CPP["DepthImageMaxDistanceOp"] = True
except ImportError:
    from .DepthImageMaxDistanceOp import DepthImageMaxDistanceOp
    _USE_CPP["DepthImageMaxDistanceOp"] = False

# ── DepthImageForegroundBackgroundMaskOp ──
try:
    from holohub.tcn_depthimage_fgbg_mask import TcnDepthImageFgbgMaskOp as _CppFgbgOp

    class DepthImageForegroundBackgroundMaskOp(_CppFgbgOp):
        """C++ DepthImageFgbgMask with Python-compatible constructor."""
        def __init__(self, fragment, *args, cuda_stream_pool=None, **kwargs):
            pos_args = list(args)
            if cuda_stream_pool is not None:
                pos_args.append(cuda_stream_pool)
            super().__init__(fragment, *pos_args, **kwargs)

    _USE_CPP["DepthImageForegroundBackgroundMaskOp"] = True
except ImportError:
    from .DepthImageForegroundBackgroundMaskOp import DepthImageForegroundBackgroundMaskOp
    _USE_CPP["DepthImageForegroundBackgroundMaskOp"] = False

# ── DepthImageApplyMaskOp ──
try:
    from holohub.tcn_depthimage_apply_mask import TcnDepthImageApplyMaskOp as _CppApplyMaskOp

    class DepthImageApplyMaskOp(_CppApplyMaskOp):
        """C++ DepthImageApplyMask with Python-compatible constructor."""
        def __init__(self, fragment, *args, cuda_stream_pool=None, **kwargs):
            pos_args = list(args)
            if cuda_stream_pool is not None:
                pos_args.append(cuda_stream_pool)
            super().__init__(fragment, *pos_args, **kwargs)

    _USE_CPP["DepthImageApplyMaskOp"] = True
except ImportError:
    from .DepthImageApplyMaskOp import DepthImageApplyMaskOp
    _USE_CPP["DepthImageApplyMaskOp"] = False

# ── ConvertBgraToRgbaOp ──
try:
    from holohub.tcn_convert_bgra_to_rgba import TcnConvertBgraToRgbaOp as ConvertBgraToRgbaOp
    _USE_CPP["ConvertBgraToRgbaOp"] = True
except ImportError:
    from .ConvertBgraToRgbaOp import ConvertBgraToRgbaOp
    _USE_CPP["ConvertBgraToRgbaOp"] = False

# Log which backend is in use
for _name, _cpp in _USE_CPP.items():
    _log.debug("%-45s %s", _name, "C++" if _cpp else "Python")



from .RotateImage180Op import RotateImage180Op

__all__ = ["StreamMergerOp", "StreamSplitterOp", "FlattenTensorOp",
           "DepthImageMaxDistanceOp", "DepthImageForegroundBackgroundMaskOp",
           "DepthImageApplyMaskOp", "ConvertBgraToRgbaOp", "RotateImage180Op"]

# ── frame identity (acquisition timestamps + tensor-map filtering) ──
# Pure Python with no C++ counterpart: these wrap Holoscan's own accessors.
from .frame_identity import (  # noqa: E402
    NO_ACQ_TIMESTAMP,
    NON_TENSOR_COMPONENTS,
    acq_timestamp,
    acq_timestamp_consensus,
    tensor_names,
)

# 180-degree rotation as a plain function, for callers that need it inline rather than as a
# graph node (see rotate.py).
from .rotate import rotate180  # noqa: E402
