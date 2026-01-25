import numpy as np
from typing import Tuple, Optional
from pyglm import glm
import logging

log = logging.getLogger(__name__)

class FirstPersonView:
    """
    FPV camera controller implemented using pyglm.
    Maintains the 'ArcBall' class name and method signatures to avoid breaking SlangWindow.
    """

    def __init__(self,
                 camera_position: np.ndarray,
                 view_center: np.ndarray,
                 up_dir: np.ndarray,
                 fov_degrees: float,
                 window_size: Tuple[int, int]):
        self._window_size = glm.ivec2(window_size[0], window_size[1])
        self._fov = fov_degrees
        
        # State
        self._eye = glm.vec3(camera_position[0], camera_position[1], camera_position[2])
        self._up = glm.vec3(up_dir[0], up_dir[1], up_dir[2])
        
        # Calculate initial yaw/pitch from view_center
        direction = glm.normalize(glm.vec3(view_center[0], view_center[1], view_center[2]) - self._eye)
        self._pitch = glm.degrees(glm.asin(direction.y))
        self._yaw = glm.degrees(glm.atan2(direction.z, direction.x))
        
        self._prev_mouse_pos = glm.vec2(0, 0)
        self._mouse_sensitivity = 0.1
        self._move_speed = 0.05
        
        # Initial state for reset
        self._init_eye = glm.vec3(self._eye)
        self._init_yaw = self._yaw
        self._init_pitch = self._pitch

    def set_view_parameters(self, eye: np.ndarray, view_center: np.ndarray, up_dir: np.ndarray):
        self._eye = glm.vec3(eye[0], eye[1], eye[2])
        direction = glm.normalize(glm.vec3(view_center[0], view_center[1], view_center[2]) - self._eye)
        self._pitch = glm.degrees(glm.asin(direction.y))
        self._yaw = glm.degrees(glm.atan2(direction.z, direction.x))
        self._up = glm.vec3(up_dir[0], up_dir[1], up_dir[2])

    def reset(self):
        self._eye = glm.vec3(self._init_eye)
        self._yaw = self._init_yaw
        self._pitch = self._init_pitch

    def reshape(self, window_size: Tuple[int, int]):
        self._window_size = glm.ivec2(window_size[0], window_size[1])

    def init_transformation(self, mouse_pos: Tuple[int, int]):
        self._prev_mouse_pos = glm.vec2(mouse_pos[0], mouse_pos[1])

    def rotate(self, mouse_pos: Tuple[int, int]):
        """Rotates camera (Yaw/Pitch) based on mouse movement."""
        curr_pos = glm.vec2(mouse_pos[0], mouse_pos[1])
        offset = curr_pos - self._prev_mouse_pos
        self._prev_mouse_pos = curr_pos

        self._yaw += offset.x * self._mouse_sensitivity
        self._pitch -= offset.y * self._mouse_sensitivity # Inverted Y for intuitive feel

        # Constrain pitch to avoid flipping
        self._pitch = glm.clamp(self._pitch, -89.0, 89.0)

    def translate(self, mouse_pos: Tuple[int, int]):
        """Translates camera along local Right (X) and Up (Y) axes."""
        curr_pos = glm.vec2(mouse_pos[0], mouse_pos[1])
        offset = curr_pos - self._prev_mouse_pos
        self._prev_mouse_pos = curr_pos

        front, right, up = self._get_camera_basis()
        
        # Scale movement by window size to feel consistent
        self._eye -= right * (offset.x * self._move_speed)
        self._eye += up * (offset.y * self._move_speed)

    def zoom(self, delta: float):
        """Moves camera along local Forward (Z) axis."""
        front, _, _ = self._get_camera_basis()
        self._eye += front * (delta * self._move_speed * 10.0)

    def update_transformation(self) -> bool:
        """Always return true as we update view_matrix on the fly, or just return True for compatibility."""
        return True

    def view_matrix(self) -> np.ndarray:
        front, _, _ = self._get_camera_basis()
        view = glm.lookAt(self._eye, self._eye + front, self._up)
        # Convert pyglm matrix to numpy array (column-major to row-major layout check)
        return np.array(view)

    def _get_camera_basis(self):
        """Calculates local camera axes based on yaw and pitch."""
        front = glm.vec3()
        front.x = glm.cos(glm.radians(self._yaw)) * glm.cos(glm.radians(self._pitch))
        front.y = glm.sin(glm.radians(self._pitch))
        front.z = glm.sin(glm.radians(self._yaw)) * glm.cos(glm.radians(self._pitch))
        front = glm.normalize(front)
        
        right = glm.normalize(glm.cross(front, self._up))
        up = glm.normalize(glm.cross(right, front))
        return front, right, up

    # Keep other props for compatibility if needed
    @property
    def lagging(self): return 0.0
    def set_lagging(self, val): pass