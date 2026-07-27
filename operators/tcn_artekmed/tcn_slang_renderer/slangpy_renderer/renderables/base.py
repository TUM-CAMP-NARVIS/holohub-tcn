"""
Base renderable class for all render-able objects in the scene.
Provides common interface for managing 6D pose, visibility, and GPU synchronization.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import slangpy as spy


class Renderable(ABC):
    """
    Abstract base class for all renderable objects in the scene.

    Provides common interface for managing:
    - 6D pose (4x4 transformation matrix)
    - Visibility state
    - GPU buffer synchronization
    - Data updates (thread-safe)
    """

    def __init__(self, device: spy.Device):
        self.device = device
        self._pose = np.eye(4, dtype=np.float32)
        self._visible = True

    @property
    def pose(self) -> np.ndarray:
        """Get the 4x4 transformation matrix for this object."""
        return self._pose

    @pose.setter
    def pose(self, value):
        """Set the 4x4 transformation matrix for this object."""
        # Handle glm.mat4x4 objects
        if hasattr(value, 'to_list'):
            # Convert glm matrix to numpy array
            matrix_list = value.to_list()
            # glm uses column-major order, convert to row-major for numpy
            self._pose = np.array(matrix_list, dtype=np.float32).T
        elif isinstance(value, np.ndarray):
            if value.shape != (4, 4):
                raise ValueError("Pose must be a 4x4 matrix")
            self._pose = value.astype(np.float32)
        else:
            # Try to convert to numpy array
            arr = np.array(value, dtype=np.float32)
            if arr.shape != (4, 4):
                raise ValueError("Pose must be a 4x4 matrix")
            self._pose = arr

    @property
    def visible(self) -> bool:
        """Get visibility state."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Set visibility state."""
        self._visible = value

    @abstractmethod
    def update(self, **kwargs):
        """
        Thread-safe method to stage data updates.
        Called from any thread (e.g., data processing pipeline).
        """
        pass

    @abstractmethod
    def sync_gpu(self):
        """
        Synchronize staged data to GPU buffers.
        Must be called from the rendering thread before drawing.
        """
        pass

    @abstractmethod
    def render(self, pass_encoder: spy.RenderPassEncoder,
               window_size: tuple[int, int],
               view_matrix: np.ndarray,
               proj_matrix: np.ndarray,
               extra_args: dict = None,
               ):
        """
        Render this object within an active render pass.
        Called from the rendering thread during the render loop.

        Args:
            pass_encoder: Active render pass encoder
            window_size: (width, height) of the render target
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            extra_args: Optional: Additional arguments for rendering customization
        """
        pass
