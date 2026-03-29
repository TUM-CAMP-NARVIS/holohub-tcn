# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import the holoscan modules we'll depend on
import holoscan.core
import holoscan.gxf

# Load the python binding
try:
    from ._tcn_zenoh_receiver import TcnZenohReceiverOp
    from ._tcn_zenoh_receiver import ZenohStreamConfig
    from ._tcn_zenoh_receiver import ZenohSession
    from ._tcn_zenoh_receiver import open_zenoh_session
    from ._tcn_zenoh_receiver import discover_streams
except ImportError as e:
    pybind11_hsdk_err = 'unknown base type "holoscan::'

    if not pybind11_hsdk_err in str(e):
        raise e

    note = """
- Holoscan SDK >= 3.3.0: make sure to link your bindings against 'holoscan::pybind11'.
- Holoscan SDK < 3.3.0: use the same compiler version as your installation of the Holoscan SDK.

See https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator_python_bindings.html#pybind11-abi-compatibility for details.
"""

    if hasattr(e, "add_note"):
        e.add_note(note)
        raise e

    e = ImportError(e.msg + "\n" + note).with_traceback(e.__traceback__)
    raise e from None


# Register types with the SDK
try:
    from ._tcn_zenoh_receiver import register_types as _register_types

    try:
        from holoscan.core import io_type_registry
    except ImportError as e:
        import warnings
        warnings.warn(
            "`holoscan.core.io_type_registry` is unavailable in Holoscan SDK < 2.1.0. "
            "To use a user-defined `register_types` function, you must upgrade Holoscan SDK."
        )
        raise e

    _register_types(io_type_registry)
except ImportError as e:
    pass
