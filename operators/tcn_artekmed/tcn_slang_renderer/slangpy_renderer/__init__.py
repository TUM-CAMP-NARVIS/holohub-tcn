"""
Slangpy Renderer - Standalone 3D rendering library based on Slangpy/Vulkan.

This library provides a high-level interface for rendering 3D content using Slangpy
with support for meshes, point clouds, and various rendering primitives.

Main Components:
- SlangWindow: Main rendering window with scene management
- Renderables: Pointcloud, Mesh, ColoredMesh
- Renderers: PointcloudRenderer, MeshRenderer, ColoredMeshRenderer, PointcloudSpritesRenderer
- Controllers: ArcBall, FirstPersonView camera controllers

Example Usage:
    ```python
    from slangpy_renderer import SlangWindow, Pointcloud
    import numpy as np
    import cupy as cp

    # Create window
    window = SlangWindow(1024, 768, "My Renderer")

    # Create a pointcloud
    positions = np.random.randn(100, 100, 3).astype(np.float32)
    texcoords = np.random.rand(100, 100, 2).astype(np.float32)
    image = np.random.rand(480, 640, 4).astype(np.float32)

    pointcloud = Pointcloud(
        device=window.get_device(),
        positions=positions,
        texcoords=texcoords,
        image=image,
        sync_gpu=True
    )

    # Add to scene
    window.add_renderable("my_pointcloud", pointcloud)

    # Run rendering loop
    window.run()
    ```
"""

__version__ = "0.1.0"

# Main window class
from .window import SlangWindow

# Renderable objects
from .renderables import Renderable, Pointcloud, Mesh, ColoredMesh

# Renderers
from .renderers import (
    PointcloudRenderer,
    PointcloudSpritesRenderer,
    PointcloudSurfelRenderer,
    MeshRenderer,
    ColoredMeshRenderer
)

# Offscreen rendering
from .offscreen import OffscreenContext

# Camera controllers
from .controllers import ArcBall, FirstPersonView

# Utility functions and camera parameter types
from .utils import (
    copy_cupy_array_into_slangpy_buffer,
    CameraIntrinsics,
    ColorProjectionParameters,
    DepthParameters,
    DepthUnprojector,
)

__all__ = [
    # Version
    "__version__",

    # Main window
    "SlangWindow",

    # Renderables
    "Renderable",
    "Pointcloud",
    "Mesh",
    "ColoredMesh",

    # Renderers
    "PointcloudRenderer",
    "PointcloudSpritesRenderer",
    "PointcloudSurfelRenderer",
    "MeshRenderer",
    "ColoredMeshRenderer",

    # Offscreen rendering
    "OffscreenContext",

    # Controllers
    "ArcBall",
    "FirstPersonView",

    # Utils & camera parameter types
    "copy_cupy_array_into_slangpy_buffer",
    "CameraIntrinsics",
    "ColorProjectionParameters",
    "DepthParameters",
    "DepthUnprojector",
]
