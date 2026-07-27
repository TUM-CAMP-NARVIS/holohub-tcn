"""
Colored mesh renderable data class (e.g., for axes, wireframes).
"""
import slangpy as spy
import numpy as np
import threading

from .base import Renderable


class ColoredMesh(Renderable):
    """
    Colored mesh visualization (e.g., coordinate axes, wireframes).
    Renders colored line segments or triangles.
    """

    @staticmethod
    def create_axis3d(device: spy.Device, scale: float = 1.0):
        """
        Create a 3D coordinate axis with the given scale.

        Args:
            device: Slang device
            scale: Scale factor for the axes (default: 1.0)

        Returns:
            ColoredMesh instance representing XYZ axes
        """
        # Static geometry data for the reference frame
        # Indices for line segments (9 lines total)
        INDICES = np.array([
            0, 1,
            1, 2,  # X axis
            1, 3,

            4, 5,
            5, 6,  # Y axis
            5, 7,

            8, 9,
            9, 10,  # Z axis
            9, 11
        ], dtype=np.uint16)

        # Vertices with positions and colors (RGB for X, Y, Z axes)
        VERTICES_POSITION = np.array([
            # X axis (red)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.9, -0.1, 0.0],

            # Y axis (green)
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [-0.1, 0.9, 0.0],

            # Z axis (blue)
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
            [-0.1, 0.0, 0.9]
        ], dtype=np.float32)

        VERTICES_COLOR = np.array([
            # X axis (red)
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],

            # Y axis (green)
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],

            # Z axis (blue)
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        positions = VERTICES_POSITION * scale
        colors = VERTICES_COLOR.copy()
        indices = INDICES.copy()

        return ColoredMesh(device, positions, colors, indices, sync_gpu=True)

    def __init__(self,
                 device: spy.Device,
                 positions: np.ndarray,
                 colors: np.ndarray,
                 indices: np.ndarray,
                 sync_gpu: bool = False):
        """
        Initialize a colored mesh renderable.

        Args:
            device: Slang device
            positions: Vertex positions (Nx3)
            colors: Vertex colors (Nx3)
            indices: Line/triangle indices
            sync_gpu: If True, create GPU buffers immediately
        """
        super().__init__(device)
        self.buffer_lock = threading.Lock()
        self.renderer = None  # Will be set by ColoredMeshRenderer

        # Pending updates storage
        self._pending_data = {
            'positions': positions if not sync_gpu else None,
            'colors': colors if not sync_gpu else None,
            'indices': indices if not sync_gpu else None,
        }

        # GPU buffers
        self.position_buffer = None
        self.color_buffer = None
        self.index_buffer = None

        # Rendering info
        self.index_format = spy.IndexFormat.uint16
        self.index_count = indices.size if indices is not None else 0

        self._is_dirty = False

        if sync_gpu:
            # Initialize buffers immediately
            self.position_buffer = device.create_buffer(
                size=positions.nbytes,
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                data=positions,
            )

            self.color_buffer = device.create_buffer(
                size=colors.nbytes,
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                data=colors,
            )

            self.index_buffer = device.create_buffer(
                size=indices.nbytes,
                usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
                data=indices,
            )

    @property
    def has_geometry(self):
        """Check if all required geometry buffers are available."""
        return (self.position_buffer is not None and
                self.color_buffer is not None and
                self.index_buffer is not None)

    @property
    def is_dirty(self):
        return self._is_dirty

    def update(self,
               positions: np.ndarray = None,
               colors: np.ndarray = None,
               indices: np.ndarray = None):
        """
        Thread-safe: Call this from any thread to stage data for the next frame.

        Args:
            positions: Updated vertex positions (Nx3)
            colors: Updated vertex colors (Nx3)
            indices: Updated line/triangle indices
        """
        with self.buffer_lock:
            if positions is not None:
                self._pending_data['positions'] = positions
            if colors is not None:
                self._pending_data['colors'] = colors
            if indices is not None:
                self._pending_data['indices'] = indices
            self._is_dirty = True

    def sync_gpu(self):
        """
        Call this once per frame from the main rendering thread
        before dispatching shaders.
        """
        with self.buffer_lock:
            if self.is_dirty:
                if self._pending_data['positions'] is not None:
                    data = self._pending_data['positions']
                    if self.position_buffer is not None and self.position_buffer.size == data.nbytes:
                        self.position_buffer.copy_from_numpy(data)
                    else:
                        self.position_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['positions'] = None

                if self._pending_data['colors'] is not None:
                    data = self._pending_data['colors']
                    if self.color_buffer is not None and self.color_buffer.size == data.nbytes:
                        self.color_buffer.copy_from_numpy(data)
                    else:
                        self.color_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['colors'] = None

                if self._pending_data['indices'] is not None:
                    data = self._pending_data['indices']
                    self.index_count = data.size
                    if self.index_buffer is not None and self.index_buffer.size == data.nbytes:
                        self.index_buffer.copy_from_numpy(data)
                    else:
                        self.index_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['indices'] = None

                self._is_dirty = False

    def render(self,
               pass_encoder: spy.RenderPassEncoder,
               window_size: tuple[int, int],
               view_matrix: np.ndarray,
               proj_matrix: np.ndarray,
               extra_args: dict = None,
               ):
        """
        Render this colored mesh using its associated renderer.

        Args:
            pass_encoder: Active render pass encoder
            window_size: Window dimensions (width, height)
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            extra_args: Additional rendering parameters
        """
        if self.renderer is not None:
            self.renderer.render(
                pass_encoder,
                self,
                window_size,
                view_matrix,
                proj_matrix,
                self.pose,
                extra_args,
            )
