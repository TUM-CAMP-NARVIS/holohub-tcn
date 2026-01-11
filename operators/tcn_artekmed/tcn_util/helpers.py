from holoscan.pose_tree import Pose3
import numpy as np
import cupy as cp
import torch
import slangpy as spy


# Map numpy/cupy dtype -> slangpy.DataType (adjust to your formats)
DTYPE_TO_SLANG = {
    cp.uint8:  spy.DataType.uint8,
    cp.int8:   spy.DataType.int8,
    cp.uint16: spy.DataType.uint16,
    cp.int16:  spy.DataType.int16,
    cp.uint32:  spy.DataType.uint32,
    cp.int32:  spy.DataType.int32,
    cp.float16: spy.DataType.float16,
    cp.float32: spy.DataType.float32,
}

def copy_cupy_array_into_slangpy_buffer(src_cp: cp.ndarray, dst_buf: spy.Buffer, shape, dtype=None):
    """
    src_cp: CuPy ndarray wrapping Holoscan tensor device memory (e.g. HxWxC).
    dst_buf: preallocated slangpy.Buffer sized for shape*dtype
    shape: tuple/list, e.g. (H, W, C)
    dtype: optional, defaults to src_cp.dtype
    """
    if dtype is None:
        dtype = src_cp.dtype

    # 1) Source: CuPy -> Torch (zero-copy view via DLPack)
    # torch.utils.dlpack.from_dlpack accepts a dlpack capsule.
    src_t = torch.utils.dlpack.from_dlpack(
        src_cp.toDlpack()
    )

    # Ensure expected shape/dtype (reshape is view if compatible)
    src_t = src_t.reshape(shape)
    if src_t.dtype != torch.from_numpy(cp.empty((), dtype=dtype).get()).dtype:
        # Usually you want to avoid implicit conversions; do it explicitly if needed.
        src_t = src_t.to(dtype=torch.__dict__[str(dtype).split('.')[-1]])

    # 2) Destination: SlangPy buffer -> Torch tensor (view into dst buffer memory)
    slang_dtype = DTYPE_TO_SLANG.get(dtype.type)
    if slang_dtype is None:
        raise TypeError(f"Unsupported dtype {dtype} for this d2d copy.")

    dst_t = dst_buf.to_torch(type=slang_dtype, shape=list(shape))

    # 3) Device-to-device copy
    # This enqueues a CUDA memcpy on the current PyTorch stream.
    dst_t.copy_(src_t)

    # Optional: if the next stage is NOT using PyTorch stream semantics,
    # you may need to synchronize here.
    # torch.cuda.synchronize()



def pose3_to_matrix4x4(pose: Pose3) -> np.ndarray:
    """
    Converts a Holoscan Pose3 object to a 4x4 rigid transformation matrix.

    Args:
        pose: The holoscan.pose_tree.Pose3 object containing rotation and translation.

    Returns:
        A 4x4 float32 numpy array representing the rigid transform.
    """
    # Create a 4x4 identity matrix
    mat = np.eye(4, dtype=np.float32)

    # Set the upper-left 3x3 to the rotation matrix
    # pose.rotation.matrix() returns a 3x3 ndarray
    mat[:3, :3] = pose.rotation.matrix()

    # Set the first three elements of the last column to the translation
    # pose.translation returns a 3-element vector/ndarray
    mat[:3, 3] = pose.translation

    return mat