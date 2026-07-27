"""
OffscreenContext - Headless rendering context for testing and batch rendering.

Parallel to SlangWindow but without any window, UI, or event loop.
Creates a Vulkan device, offscreen render targets, and all renderers.
"""
import logging
import math
import os
from pathlib import Path
from typing import Optional

import slangpy as spy
import numpy as np

from .renderables import Renderable, Pointcloud, Mesh, ColoredMesh
from .renderers import (
    PointcloudRenderer,
    PointcloudSpritesRenderer,
    MeshRenderer,
    ColoredMeshRenderer,
)

log = logging.getLogger(__name__)

# Shared with window.py — keep in sync
OUTPUT_FORMAT = spy.Format.rgba8_unorm


def vulkan_rh_zo_perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """
    Canonical Vulkan-compatible perspective projection:
    - Right-handed, camera looks down -Z
    - NDC depth range: 0..1 (Vulkan ZO)
    Matrix returned in ROW-MAJOR form.
    """
    fovy = math.radians(fov_y_deg)
    f = 1.0 / math.tan(0.5 * fovy)
    A = far / (near - far)
    B = (far * near) / (near - far)

    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a right-handed look-at view matrix (row-major)."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float64)
    M[0, 0:3] = s
    M[1, 0:3] = u
    M[2, 0:3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


class OffscreenContext:
    """
    Headless rendering context — no window, no UI, no event loop.

    Creates a Vulkan device with shader include paths, offscreen color + depth
    textures, and all four renderer types. Renders to a numpy array.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        enable_cuda_interop: bool = False,
        assets_path: str = None,
        device_id: int = 0,
    ):
        """
        Initialize the offscreen rendering context.

        Args:
            width: Render target width in pixels.
            height: Render target height in pixels.
            enable_cuda_interop: If True, set up CUDA/Vulkan interop (requires CuPy).
            assets_path: Path to shader/model assets (defaults to package assets).
            device_id: Cuda device id (defaults to 0)
        """
        self.width = width
        self.height = height

        # Resolve asset root
        if assets_path is None:
            asset_root_dir = Path(__file__).parent / "assets"
        else:
            asset_root_dir = Path(assets_path)
        self.asset_root_dir = asset_root_dir

        # Build device kwargs
        device_kwargs = dict(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=True,
            compiler_options={
                "include_paths": [
                    str(asset_root_dir / "shaders"),
                    os.path.join(os.path.dirname(spy.__file__), "slang"),
                ],
                "debug_info": spy.SlangDebugInfoLevel.maximal,
                "optimization": spy.SlangOptimizationLevel.none,
            },
        )

        if enable_cuda_interop:
            try:
                import cupy as cp
            except ImportError:
                raise ImportError(
                    "CuPy is required for CUDA interop. "
                    "Install it with: pip install slangpy-renderer[cuda]"
                )
            with cp.cuda.Device(device_id):
                _ = cp.zeros((1,), dtype=cp.uint8)
                device_handle = spy.get_cuda_current_context_native_handles()
            device_kwargs["enable_cuda_interop"] = True
            device_kwargs["existing_device_handles"] = device_handle

        self.device = spy.Device(**device_kwargs)

        # Create offscreen render targets
        self.color_texture = self.device.create_texture(
            format=OUTPUT_FORMAT,
            width=width,
            height=height,
            usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
        )
        self.depth_texture = self.device.create_texture(
            format=spy.Format.d32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.depth_stencil,
        )

        # Create renderers (stateless, shared across renderables)
        self.mesh_renderer = MeshRenderer(self.device, OUTPUT_FORMAT)
        self.pointcloud_renderer = PointcloudRenderer(self.device, OUTPUT_FORMAT)
        self.pointcloud_sprites_renderer = PointcloudSpritesRenderer(
            self.device, OUTPUT_FORMAT
        )
        self.colored_mesh_renderer = ColoredMeshRenderer(self.device, OUTPUT_FORMAT)

        # Scene graph: name -> renderable
        self._renderables: dict[str, Renderable] = {}

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_renderable(
        self, name: str, renderable: Renderable, pose: np.ndarray = None
    ) -> str:
        """Add a renderable to the scene, associating the correct renderer."""
        if name in self._renderables:
            raise ValueError(f"Renderable with name '{name}' already exists")

        if isinstance(renderable, Mesh):
            renderable.renderer = self.mesh_renderer
        elif isinstance(renderable, Pointcloud):
            renderable.renderer = self.pointcloud_renderer
        elif isinstance(renderable, ColoredMesh):
            renderable.renderer = self.colored_mesh_renderer

        if pose is not None:
            renderable.pose = pose

        self._renderables[name] = renderable
        return name

    def remove_renderable(self, name: str):
        """Remove a renderable from the scene."""
        self._renderables.pop(name, None)

    def get_renderable(self, name: str) -> Optional[Renderable]:
        """Get a renderable by name."""
        return self._renderables.get(name)

    def clear(self):
        """Remove all renderables from the scene."""
        self._renderables.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_frame(
        self,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        extra_args: dict = None,
    ) -> np.ndarray:
        """
        Render all visible renderables and return the color buffer as numpy.

        Args:
            view_matrix: 4x4 camera view matrix.
            proj_matrix: 4x4 projection matrix.
            clear_color: RGBA clear color (default black).
            extra_args: Extra shader parameters (renderStaticColor, pointSize, etc.).

        Returns:
            (height, width, 4) uint8 numpy array (RGBA).
        """
        if extra_args is None:
            extra_args = {"renderStaticColor": True, "pointSize": 3.0}

        window_size = (self.width, self.height)

        # Sync GPU for all visible renderables
        for renderable in self._renderables.values():
            if renderable.visible:
                renderable.sync_gpu()

        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self.color_texture.create_view(),
                        "clear_value": list(clear_color),
                        "load_op": spy.LoadOp.clear,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": self.depth_texture.create_view(),
                    "depth_clear_value": 1.0,
                    "depth_load_op": spy.LoadOp.clear,
                    "depth_store_op": spy.StoreOp.store,
                    "depth_read_only": False,
                },
            }
        ) as pass_encoder:
            for renderable in self._renderables.values():
                if not renderable.visible:
                    continue
                renderable.render(
                    pass_encoder, window_size, view_matrix, proj_matrix, extra_args
                )

        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait()

        return self.color_texture.to_numpy().astype(np.uint8)

    def read_depth(self) -> np.ndarray:
        """
        Return the depth buffer from the last render as a float32 numpy array.

        Returns:
            (height, width) float32 numpy array.
        """
        return self.depth_texture.to_numpy()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def default_view_matrix(
        self,
        eye: tuple = (3.0, 3.0, 3.0),
        center: tuple = (0.0, 0.0, 0.0),
        up: tuple = (0.0, 1.0, 0.0),
    ) -> np.ndarray:
        """Compute a look-at view matrix."""
        return look_at(np.array(eye), np.array(center), np.array(up))

    def default_proj_matrix(
        self, fov: float = 60.0, near: float = 0.1, far: float = 100.0
    ) -> np.ndarray:
        """Compute a Vulkan perspective projection matrix."""
        aspect = float(self.width) / float(self.height)
        return vulkan_rh_zo_perspective(fov, aspect, near, far)
