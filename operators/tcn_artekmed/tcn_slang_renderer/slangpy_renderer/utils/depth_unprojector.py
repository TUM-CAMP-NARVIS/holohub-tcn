"""
GPU Depth-to-Pointcloud converter using XY Lookup Table compute shaders.

Converts 16-bit depth images (known intrinsics, Brown-Conrady distortion) into
structured pointclouds entirely on the GPU. Two-stage pipeline:

1. XY Lookup Table (compute once per camera): iteratively un-distorts each pixel
   to produce a unit-ray direction (x, y) per pixel via Newton-Raphson solver.
2. Depth -> Pointcloud (compute every frame): multiply each pixel's ray by depth,
   and optionally project each 3D point into a color camera's image plane to
   produce normalized UV texture coordinates.

Ported from: https://github.com/TUM-CAMP-NARVIS/xylt
Color projection ported from: tcn_depthimage_backprojection_kernel.cu
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import slangpy as spy


# Thread group size matching the shader's [numthreads(16, 16, 1)]
_GROUP_SIZE = 16


def _group_count(total: int, group_size: int) -> int:
    return math.ceil(total / group_size)


# ============================================================================
# Camera parameter data classes
# ============================================================================


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters with Brown-Conrady distortion model.

    Matches the ``CameraIntrinsics`` struct in ``depth_unproject.slang``.
    Reusable for both depth and color cameras.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    radial_distortion: Sequence[float] = field(default_factory=lambda: [0.0] * 6)
    tangential_distortion: Sequence[float] = field(default_factory=lambda: [0.0, 0.0])
    max_radius: float = 1.7

    @property
    def k(self) -> list[float]:
        """Radial distortion coefficients [k1..k6], zero-padded to 6."""
        return (list(self.radial_distortion) + [0.0] * 6)[:6]

    @property
    def p(self) -> list[float]:
        """Tangential distortion coefficients [p1, p2], zero-padded to 2."""
        return (list(self.tangential_distortion) + [0.0] * 2)[:2]


@dataclass
class DepthParameters:
    """
    Complete depth camera configuration for GPU unprojection.

    Matches the ``DepthParameters`` struct in ``depth_unproject.slang``.
    """

    width: int
    height: int
    intrinsics: CameraIntrinsics
    depth_scale: float = 0.001


@dataclass
class ColorProjectionParameters:
    """
    Parameters for projecting depth-camera 3D points into a color camera's image plane.

    Matches the ``ColorProjectionParams`` struct in ``depth_unproject.slang``.
    """

    width: int
    height: int
    intrinsics: CameraIntrinsics
    depth_to_color: np.ndarray  # (4, 4) float64 or float32, depth camera -> color camera

    @classmethod
    def from_calibration(
        cls,
        color_params: dict,
        color2depth_transform: dict,
    ) -> ColorProjectionParameters:
        """
        Build from calibration JSON sections.

        Args:
            color_params: The ``color_parameters`` dict (has fx, fy, cx, cy,
                width, height, radial_distortion, tangential_distortion, metric_radius).
            color2depth_transform: The ``color2depth_transform`` dict with
                ``translation`` {x, y, z} and ``rotation`` {x, y, z, w} quaternion.
                This is the color-to-depth transform; it will be inverted internally
                to get the depth-to-color transform needed by the shader.
        """
        rot = color2depth_transform["rotation"]
        trans = color2depth_transform["translation"]
        c2d = _rigid_transform_to_matrix(
            rotation_xyzw=(rot["x"], rot["y"], rot["z"], rot["w"]),
            translation_xyz=(trans["x"], trans["y"], trans["z"]),
        )
        d2c = _invert_rigid_transform(c2d)
        return cls(
            width=color_params["width"],
            height=color_params["height"],
            intrinsics=CameraIntrinsics(
                fx=color_params["fx"],
                fy=color_params["fy"],
                cx=color_params["cx"],
                cy=color_params["cy"],
                radial_distortion=color_params["radial_distortion"],
                tangential_distortion=color_params["tangential_distortion"],
                max_radius=color_params["metric_radius"],
            ),
            depth_to_color=d2c,
        )


# ============================================================================
# Rigid transform helpers
# ============================================================================


def _quaternion_to_rotation_matrix(
    x: float, y: float, z: float, w: float
) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rigid_transform_to_matrix(
    rotation_xyzw: tuple[float, float, float, float],
    translation_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Build a 4x4 rigid body transform matrix from quaternion + translation."""
    x, y, z, w = rotation_xyzw
    R = _quaternion_to_rotation_matrix(x, y, z, w)
    t = np.array(translation_xyz, dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _invert_rigid_transform(M: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid body transform: M_inv = [[R^T, -R^T @ t], [0, 0, 0, 1]]."""
    R = M[:3, :3]
    t = M[:3, 3]
    M_inv = np.eye(4, dtype=np.float64)
    M_inv[:3, :3] = R.T
    M_inv[:3, 3] = -R.T @ t
    return M_inv


# ============================================================================
# ShaderCursor binding helpers
# ============================================================================


def _bind_intrinsics(cursor: spy.ShaderCursor, cam: CameraIntrinsics) -> None:
    """Bind a CameraIntrinsics to a ShaderCursor pointing at a CameraIntrinsics struct."""
    k = cam.k
    p = cam.p
    cursor.fx = cam.fx
    cursor.fy = cam.fy
    cursor.cx = cam.cx
    cursor.cy = cam.cy
    cursor.k1 = k[0]
    cursor.k2 = k[1]
    cursor.k3 = k[2]
    cursor.k4 = k[3]
    cursor.k5 = k[4]
    cursor.k6 = k[5]
    cursor.p1 = p[0]
    cursor.p2 = p[1]
    cursor.max_radius = cam.max_radius


def _bind_depth_params(cursor: spy.ShaderCursor, params: DepthParameters) -> None:
    """Bind a DepthParameters to a ShaderCursor pointing at a DepthParameters struct."""
    cursor.width = params.width
    cursor.height = params.height
    _bind_intrinsics(cursor.intrinsics, params.intrinsics)
    cursor.depth_scale = params.depth_scale


def _bind_color_projection_params(
    cursor: spy.ShaderCursor, params: ColorProjectionParameters | None
) -> None:
    """Bind a ColorProjectionParams to a ShaderCursor. Binds safe defaults when None."""
    if params is not None:
        cursor.width = params.width
        cursor.height = params.height
        _bind_intrinsics(cursor.intrinsics, params.intrinsics)
        cursor.depth_to_color = params.depth_to_color
    else:
        # Bind safe defaults — these won't be used (compute_texcoords will be False)
        cursor.width = 1
        cursor.height = 1
        _bind_intrinsics(
            cursor.intrinsics,
            CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        )
        cursor.depth_to_color = np.eye(4, dtype=np.float64)


# ============================================================================
# Main class
# ============================================================================


class DepthUnprojector:
    """
    GPU depth-to-pointcloud converter with Brown-Conrady distortion handling
    and optional color camera UV projection.

    On construction, compiles two compute kernels and dispatches the XY lookup
    table computation once. Then ``unproject()`` can be called every frame to
    convert a uint16 depth image into a float3 pointcloud buffer and (optionally)
    a float2 UV texcoord buffer.

    The output ``position_buffer`` is a ``StructuredBuffer<float3>`` that can be
    passed directly to ``PointcloudSpritesRenderer`` via a ``Pointcloud`` renderable.
    """

    def __init__(
        self,
        device: spy.Device,
        params: DepthParameters,
        color_params: ColorProjectionParameters | None = None,
    ):
        """
        Initialize the depth unprojector and compute the XY lookup table.

        Args:
            device: SlangPy device (must have shader include paths configured).
            params: Depth camera parameters (dimensions, intrinsics, depth scale).
            color_params: Optional color camera parameters. When provided, each
                ``unproject()`` call will also compute normalized UV texture
                coordinates for projecting depth points into the color image.
        """
        self.m_device = device
        self.m_params = params
        self.m_color_params = color_params

        # --- Compile compute kernels ---
        xy_program = device.load_program(
            "depth_unproject.slang", ["compute_xy_table"]
        )
        self.m_xy_kernel = device.create_compute_kernel(xy_program)

        pc_program = device.load_program(
            "depth_unproject.slang", ["compute_pointcloud"]
        )
        self.m_pc_kernel = device.create_compute_kernel(pc_program)

        normals_program = device.load_program(
            "depth_unproject.slang", ["compute_normals"]
        )
        self.m_normals_kernel = device.create_compute_kernel(normals_program)

        # --- Create GPU resources ---

        w, h = params.width, params.height

        # XY table: float2 per pixel (8 bytes per element)
        self.m_xy_table_buffer = device.create_buffer(
            element_count=w * h,
            struct_size=8,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        # Pointcloud output: float3 per pixel (12 bytes per element)
        self.m_position_buffer = device.create_buffer(
            element_count=w * h,
            struct_size=12,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        # Texcoord output: float2 per pixel (8 bytes per element)
        self.m_texcoord_buffer = device.create_buffer(
            element_count=w * h,
            struct_size=8,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        # Normal output: float3 per pixel (12 bytes per element)
        self.m_normal_buffer = device.create_buffer(
            element_count=w * h,
            struct_size=12,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        # Depth texture: r16_uint, uploaded every frame
        self.m_depth_texture = device.create_texture(
            format=spy.Format.r16_uint,
            width=w,
            height=h,
            usage=spy.TextureUsage.shader_resource,
        )

        # --- Dispatch XY table computation (once) ---
        self._dispatch_xy_table()

    def _dispatch_xy_table(self) -> None:
        """Compute the XY lookup table via GPU compute pass."""
        w, h = self.m_params.width, self.m_params.height
        groups_x = _group_count(w, _GROUP_SIZE)
        groups_y = _group_count(h, _GROUP_SIZE)

        command_encoder = self.m_device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_enc:
            shader_obj = pass_enc.bind_pipeline(self.m_xy_kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj)
            _bind_depth_params(cursor.depth_params, self.m_params)
            cursor.xy_table = self.m_xy_table_buffer
            pass_enc.dispatch_compute(spy.uint3(groups_x, groups_y, 1))
        self.m_device.submit_command_buffer(command_encoder.finish())
        self.m_device.wait()

    def _dispatch_pointcloud(self) -> None:
        """Compute the pointcloud (and optionally UVs) from depth texture and XY table."""
        w, h = self.m_params.width, self.m_params.height
        groups_x = _group_count(w, _GROUP_SIZE)
        groups_y = _group_count(h, _GROUP_SIZE)

        command_encoder = self.m_device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_enc:
            shader_obj = pass_enc.bind_pipeline(self.m_pc_kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj)
            _bind_depth_params(cursor.depth_params, self.m_params)
            cursor.depth_texture = self.m_depth_texture
            cursor.xy_table_in = self.m_xy_table_buffer
            cursor.pointcloud = self.m_position_buffer
            cursor.texcoords = self.m_texcoord_buffer
            cursor.compute_texcoords = self.m_color_params is not None
            _bind_color_projection_params(cursor.color_params, self.m_color_params)
            pass_enc.dispatch_compute(spy.uint3(groups_x, groups_y, 1))
        self.m_device.submit_command_buffer(command_encoder.finish())
        self.m_device.wait()

    def _dispatch_normals(self) -> None:
        """Compute per-pixel surface normals from the structured pointcloud grid."""
        w, h = self.m_params.width, self.m_params.height
        groups_x = _group_count(w, _GROUP_SIZE)
        groups_y = _group_count(h, _GROUP_SIZE)

        command_encoder = self.m_device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_enc:
            shader_obj = pass_enc.bind_pipeline(self.m_normals_kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj)
            _bind_depth_params(cursor.depth_params, self.m_params)
            cursor.pointcloud = self.m_position_buffer
            cursor.normals = self.m_normal_buffer
            pass_enc.dispatch_compute(spy.uint3(groups_x, groups_y, 1))
        self.m_device.submit_command_buffer(command_encoder.finish())
        self.m_device.wait()

    def unproject(self, depth_image: np.ndarray) -> spy.Buffer:
        """
        Convert a uint16 depth image to a float3 pointcloud on the GPU.

        When color_params were provided at construction, also computes normalized
        UV texture coordinates for each point.

        Args:
            depth_image: (H, W) uint16 numpy array with raw depth values.

        Returns:
            The position buffer (StructuredBuffer<float3>, width*height elements).
            This is the same buffer reference every call — contents are overwritten.
        """
        assert depth_image.dtype == np.uint16, (
            f"Expected uint16 depth image, got {depth_image.dtype}"
        )
        assert depth_image.shape == (self.m_params.height, self.m_params.width), (
            f"Expected shape ({self.m_params.height}, {self.m_params.width}), "
            f"got {depth_image.shape}"
        )

        # Upload depth image to GPU texture
        self.m_depth_texture.copy_from_numpy(depth_image)

        # Dispatch pointcloud computation
        self._dispatch_pointcloud()

        # Dispatch normal computation
        self._dispatch_normals()

        return self.m_position_buffer

    @property
    def has_color_projection(self) -> bool:
        """Whether color camera UV projection is enabled."""
        return self.m_color_params is not None

    @property
    def position_buffer(self) -> spy.Buffer:
        """The output pointcloud buffer (stable reference, reused every frame)."""
        return self.m_position_buffer

    @property
    def texcoord_buffer(self) -> spy.Buffer:
        """The output texcoord buffer (stable reference, reused every frame)."""
        return self.m_texcoord_buffer

    @property
    def normal_buffer(self) -> spy.Buffer:
        """The output normal buffer (stable reference, reused every frame)."""
        return self.m_normal_buffer

    @property
    def num_points(self) -> int:
        """Total number of points (width * height)."""
        return self.m_params.width * self.m_params.height

    def to_numpy(self) -> np.ndarray:
        """
        Read back the pointcloud buffer as a numpy array.

        Returns:
            (H, W, 3) float32 numpy array of 3D point positions.
        """
        raw = self.m_position_buffer.to_numpy()
        return raw.view(np.float32).reshape(
            self.m_params.height, self.m_params.width, 3
        )

    def texcoords_to_numpy(self) -> np.ndarray:
        """
        Read back the texcoord buffer as a numpy array.

        Returns:
            (H, W, 2) float32 numpy array of normalized UV coordinates.
        """
        raw = self.m_texcoord_buffer.to_numpy()
        return raw.view(np.float32).reshape(
            self.m_params.height, self.m_params.width, 2
        )

    def normals_to_numpy(self) -> np.ndarray:
        """
        Read back the normal buffer as a numpy array.

        Returns:
            (H, W, 3) float32 numpy array of per-point normal vectors.
        """
        raw = self.m_normal_buffer.to_numpy()
        return raw.view(np.float32).reshape(
            self.m_params.height, self.m_params.width, 3
        )

    def xy_table_to_numpy(self) -> np.ndarray:
        """
        Read back the XY lookup table as a numpy array.

        Returns:
            (H, W, 2) float32 numpy array of unit-ray (x, y) directions.
        """
        raw = self.m_xy_table_buffer.to_numpy()
        return raw.view(np.float32).reshape(
            self.m_params.height, self.m_params.width, 2
        )
