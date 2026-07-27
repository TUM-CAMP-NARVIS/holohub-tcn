"""
CUDA/CuPy utility functions for GPU buffer operations.

CuPy is an optional dependency. Functions in this module will raise ImportError
at call time if CuPy is not installed.
"""
import numpy as np
import slangpy as spy


def copy_cupy_array_into_slangpy_buffer(cupy_array, slang_buffer, shape):
    """
    Copy data from a CuPy array into a Slangpy buffer using CUDA interop.

    Args:
        cupy_array: Source CuPy array (can be numpy array, will be converted)
        slang_buffer: Destination Slangpy buffer
        shape: Shape of the data

    Raises:
        ImportError: If CuPy is not installed.
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for CUDA buffer operations. "
            "Install it with: pip install slangpy-renderer[cuda]"
        )

    # Ensure we have a cupy array
    if isinstance(cupy_array, np.ndarray):
        cupy_array = cp.asarray(cupy_array)

    # Get the CUDA device pointer from slangpy buffer
    slang_cuda_ptr = slang_buffer.device_address

    # Create a cupy array view of the slangpy buffer
    slang_view = cp.ndarray(
        shape=shape,
        dtype=cupy_array.dtype,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(slang_cuda_ptr, slang_buffer.size, owner=None),
            offset=0
        )
    )

    # Copy data from cupy array to slangpy buffer
    slang_view[:] = cupy_array.reshape(shape)
