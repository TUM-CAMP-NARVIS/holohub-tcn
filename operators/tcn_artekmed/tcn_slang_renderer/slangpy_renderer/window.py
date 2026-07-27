"""
SlangWindow - Main rendering window class with scene management.
Standalone version without Holoscan dependencies.
"""
import logging
import os
import threading
import time
import math
from typing import Callable, Optional
from pathlib import Path

import slangpy as spy
import numpy as np

from .renderables import Renderable, Pointcloud, Mesh, ColoredMesh
from .renderers import (
    PointcloudRenderer,
    PointcloudSpritesRenderer,
    PointcloudSurfelRenderer,
    MeshRenderer,
    ColoredMeshRenderer
)
from .controllers import ArcBall, FirstPersonView

log = logging.getLogger(__name__)


def vulkan_rh_zo_perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Canonical Vulkan-compatible perspective projection:
    - Right-handed
    - Camera looks down -Z in view space
    - NDC depth range: 0..1 (Vulkan ZO)
    Matrix is returned in ROW-MAJOR form.
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


class SlangWindow:
    """
    Main rendering window for Slangpy-based visualization.

    Provides:
    - Window creation and event handling
    - Scene management (add/remove/update renderables)
    - Camera control (arcball or FPV)
    - Rendering pipeline management
    - Thread-safe updates from external data sources
    """

    def __init__(self,
                 width: int,
                 height: int,
                 title: str,
                 resizeable: bool = True,
                 close_callback: Callable = None,
                 assets_path: str = None,
                 device_id: int = 0):
        """
        Initialize the rendering window.

        Args:
            width: Window width
            height: Window height
            title: Window title
            resizeable: Whether window can be resized
            close_callback: Callback function when window closes
            assets_path: Path to shader/model assets (defaults to package assets)
            device_id: Cuda device id (defaults to 0)
        """
        self.ui = None
        self.close_callback = close_callback

        self.window = spy.Window(width, height, title, resizable=resizeable)

        # Determine asset path
        if assets_path is None:
            asset_root_dir = Path(__file__).parent / "assets"
        else:
            asset_root_dir = Path(assets_path)
        self.asset_root_dir = asset_root_dir

        # Initialize CUDA/Vulkan interop
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "CuPy is required for SlangWindow (CUDA/Vulkan interop). "
                "Install it with: pip install slangpy-renderer[cuda]"
            )
        with cp.cuda.Device(device_id):
            _ = cp.zeros((1,), dtype=cp.uint8)  # forces context creation if not existent
            device_handle = spy.get_cuda_current_context_native_handles()

        self.device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=True,
            enable_cuda_interop=True,
            existing_device_handles=device_handle,
            compiler_options={
                "include_paths": [
                    str(asset_root_dir / "shaders"),
                    os.path.join(os.path.dirname(spy.__file__), "slang"),
                ],
                "debug_info": spy.SlangDebugInfoLevel.maximal,
                "optimization": spy.SlangOptimizationLevel.none,
            },
        )

        self.surface = self.device.create_surface(self.window)
        self.surface.configure(self.window.width, self.window.height)
        output_format = self.surface.config.format

        # Create renderers (stateless, shared by all renderables)
        self.mesh_renderer = MeshRenderer(self.device, output_format)
        self.pointcloud_renderer = PointcloudRenderer(self.device, output_format)
        self.pointcloud_sprites_renderer = PointcloudSpritesRenderer(self.device, output_format)
        self.pointcloud_surfel_renderer = PointcloudSurfelRenderer(self.device, output_format)
        self.colored_mesh_renderer = ColoredMeshRenderer(self.device, output_format)

        # Scene management
        self._renderables = {}  # name -> Renderable
        self._staged_renderables = {}
        self._next_id = 0

        # Camera setup
        self.camera_pos = np.asarray([5, 5, 5], dtype=np.float32)
        self.camera_target = np.asarray([0, 0, 0], dtype=np.float32)
        self.camera_up = np.asarray([0, 1, 0], dtype=np.float32)
        self.fov = 60.0

        self.model_pose = spy.math.float3(0., 0., 0.)

        self.near_plane = 1.0
        self.far_plane = 50.0
        self.timer = time.perf_counter()

        self.arc_ball = ArcBall(self.camera_pos, self.camera_target, self.camera_up, self.fov, (width, height))
        self.current_mouse_button_down = None
        self.arc_ball_needs_init = False

        self.window.on_keyboard_event = self._on_window_keyboard_event
        self.window.on_mouse_event = self._on_window_mouse_event
        self.window.on_resize = self.handle_resize

        # UI variables
        self._render_static_colors = True
        self._point_size = 3.0

        # Create UI
        self.create_depth_texture()
        self.setup_ui()
        self.dirty = False
        self.surface_texture = None
        self._mouse_pos = None

        # Sync rendering with data input
        self._cv = threading.Condition()
        self._should_render = True
        self._running = True

        # Hookable events
        self.on_keyboard_event: Optional[Callable[[spy.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[spy.MouseEvent], None]] = None

        # Load a default mesh for testing (can be removed later)
        # model_path = asset_root_dir / "models" / "cube.obj"
        # if model_path.exists():
        #     default_mesh = Mesh.from_obj(self.device, str(model_path))
        #     self.add_renderable("default_mesh", default_mesh)

    def set_model_pose(self, pose: spy.math.float3):
        """Set pose for all renderables (helper method)."""
        self.model_pose = pose
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = pose[0]
        transform[1, 3] = pose[1]
        transform[2, 3] = pose[2]
        for renderable in self._renderables.values():
            renderable.pose = transform

    def setup_ui(self):
        """Setup ImGui-based UI controls."""
        self.ui = spy.ui.Context(self.device)

        window = spy.ui.Window(
            self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(300, 300)
        )

        spy.ui.CheckBox(window, "Render Static Color", self._render_static_colors,
                       lambda v: setattr(self, "_render_static_colors", v))
        spy.ui.InputFloat(window, "Point Size", self._point_size,
                         lambda v: setattr(self, "_point_size", v))
        spy.ui.InputFloat3(window, "Model Pose", self.model_pose, self.set_model_pose)

    def add_renderable(self, name: str, renderable: Renderable, pose: np.ndarray = None) -> str:
        """
        Add a renderable object to the scene.

        Args:
            name: Unique name for this renderable
            renderable: Renderable object (Mesh, Pointcloud, etc.)
            pose: Optional 4x4 transformation matrix (defaults to identity)

        Returns:
            The name used to identify this renderable

        Raises:
            ValueError: If name already exists
        """
        if name in self._renderables:
            raise ValueError(f"Renderable with name '{name}' already exists")

        # Associate renderer with the renderable
        if isinstance(renderable, Mesh):
            renderable.renderer = self.mesh_renderer
        elif isinstance(renderable, Pointcloud):
            renderable.renderer = self.pointcloud_renderer
        elif isinstance(renderable, ColoredMesh):
            renderable.renderer = self.colored_mesh_renderer

        if pose is not None:
            renderable.pose = pose

        self._staged_renderables[name] = renderable
        return name

    def remove_renderable(self, name: str):
        """Remove a renderable object from the scene."""
        if name in self._renderables:
            del self._renderables[name]

    def get_renderable(self, name: str) -> Optional[Renderable]:
        """Get a renderable by name."""
        return self._renderables.get(name) or self._staged_renderables.get(name)

    def set_pose(self, name: str, pose: np.ndarray):
        """Set the 6D pose of a renderable object."""
        if name in self._renderables:
            self._renderables[name].pose = pose

    def set_visible(self, name: str, visible: bool):
        """Set the visibility of a renderable object."""
        if name in self._renderables:
            self._renderables[name].visible = visible

    def get_view_matrix(self) -> np.ndarray:
        """Compute the current view matrix from camera parameters."""
        view_pose = self.arc_ball.view_matrix()
        return view_pose

    def get_projection_matrix(self) -> np.ndarray:
        """Compute the current projection matrix from camera parameters."""
        aspect = float(self.window.width) / float(self.window.height)
        proj_matrix = vulkan_rh_zo_perspective(self.fov, aspect, self.near_plane, self.far_plane)
        return proj_matrix

    def get_device(self):
        """Get the Slangpy device."""
        return self.device

    def close(self):
        """Close the window and stop rendering."""
        self._running = False
        self.window.close()
        if self.close_callback is not None:
            self.close_callback()

    def handle_resize(self, width, height):
        """Handle window resize events."""
        self.dirty = True

    def create_depth_texture(self):
        """Create depth buffer texture."""
        self.depth_texture = self.device.create_texture(
            format=spy.Format.d32_float,
            width=self.window.width,
            height=self.window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    def resize(self):
        """Resize rendering buffers after window resize."""
        del self.depth_texture
        del self.surface_texture
        self.device.wait()
        self.surface.configure(self.window.width, self.window.height)
        self.create_depth_texture()

    def _on_visibility_changed(self, name: str, value: bool):
        """Internal callback for UI visibility toggles."""
        if name in self._renderables:
            self._renderables[name].visible = value

    def _on_window_keyboard_event(self, event: spy.KeyboardEvent):
        """Handle keyboard events."""
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.close()
                return

            key_str = chr(event.key.value)
            if key_str in [str(i+1) for i in range(9)]:
                idx = int(key_str) - 1
                keys = list(sorted(self._renderables.keys()))
                if idx < len(keys):
                    self.set_visible(keys[idx], not self._renderables[keys[idx]].visible)

        if self.on_keyboard_event:
            self.on_keyboard_event(event)
        else:
            self.ui.handle_keyboard_event(event)

    def _on_window_mouse_event(self, event: spy.MouseEvent):
        """Handle mouse events."""
        if event.type == spy.MouseEventType.button_down:
            log.debug(f"Mouse button down {event.pos} {event.mods} {event.button}")
            if self.current_mouse_button_down != event.button:
                self.arc_ball_needs_init = True
            self.current_mouse_button_down = event.button
        elif event.type == spy.MouseEventType.button_up:
            log.debug(f"Mouse button up {event.pos} {event.mods} {event.button}")
            self.current_mouse_button_down = None
        elif event.type == spy.MouseEventType.move:
            pos = (int(event.pos.x), int(event.pos.y))
            if self.current_mouse_button_down == spy.MouseButton.left:
                if self.arc_ball_needs_init:
                    self.arc_ball_needs_init = False
                    self.arc_ball.init_transformation(pos)

                if event.mods == spy.KeyModifierFlags.shift:
                    self.arc_ball.translate(pos)
                else:
                    self.arc_ball.rotate(pos)
                self.request_redraw()
        elif event.type == spy.MouseEventType.scroll:
            log.debug(f"mouse scroll {event}")
            delta = event.scroll.y / 5.0
            if event.mods == spy.KeyModifierFlags.shift:
                delta /= 3.0
            self.arc_ball.zoom(delta)
            self.request_redraw()

        self.ui.handle_mouse_event(event)

    def request_redraw(self):
        """Request a redraw of the scene (thread-safe)."""
        with self._cv:
            self._should_render = True
            self._cv.notify()

    def run(self):
        """Main rendering loop."""
        while self._running:
            self.window.process_events()

            if self.window.should_close():
                self._running = False
                break

            staged_renderables = {}
            staged_renderables, self._staged_renderables = self._staged_renderables, staged_renderables

            for k,v in staged_renderables.items():
                if k in self._renderables:
                    log.warning("skipping duplicate renderable %s", k)
                    continue
                self._renderables[k] = v

            with self._cv:
                if not self._should_render:
                    self._cv.wait(timeout=0.01)
                should_render = self._should_render
                self._should_render = False

            if not should_render:
                continue

            # Render the frame
            window_size = (self.window.width, self.window.height)
            if self.dirty:
                self.resize()
                self.arc_ball.reshape(window_size)
                self.dirty = False

            self.arc_ball.update_transformation()

            self.surface_texture = self.surface.acquire_next_image()
            if not self.surface_texture:
                continue

            command_encoder = self.device.create_command_encoder()

            # Compute camera matrices once per frame
            view_matrix = self.get_view_matrix()
            proj_matrix = self.get_projection_matrix()

            # Sync GPU buffers for all dirty renderables before rendering
            for name, renderable in self._renderables.items():
                if renderable.visible:
                    renderable.sync_gpu()

            # Begin single render pass for all renderables
            with command_encoder.begin_render_pass(
                {
                    "color_attachments": [
                        {
                            "view": self.surface_texture.create_view(),
                            "clear_value": [0.0, 0.0, 0.0, 1.0],
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
                # Render all visible renderables in a single pass
                for name, renderable in self._renderables.items():
                    if not renderable.visible:
                        continue

                    # Use the render method from the renderable (which delegates to its renderer)
                    renderable.render(
                        pass_encoder,
                        window_size,
                        view_matrix,
                        proj_matrix,
                        extra_args={
                            "renderStaticColor": self._render_static_colors,
                            "pointSize": self._point_size,
                        }
                    )

            self.device.submit_command_buffer(command_encoder.finish())
            self.surface.present()
