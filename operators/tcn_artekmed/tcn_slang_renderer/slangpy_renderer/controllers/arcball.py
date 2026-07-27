"""
ArcBall Camera Controller - Pure numpy implementation without Holoscan dependencies.

Based on the Magnum ArcBall implementation, adapted to use pure numpy for all operations.
"""

import numpy as np
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)


class ArcBall:
    """
    ArcBall camera controller with smooth lagging and intuitive mouse controls.

    Supports:
    - Rotation via arcball metaphor
    - Translation (panning)
    - Zooming
    - Smooth interpolation (lagging) for fluid camera motion
    """

    def __init__(self,
                 camera_position: np.ndarray,
                 view_center: np.ndarray,
                 up_dir: np.ndarray,
                 fov_degrees: float,
                 window_size: Tuple[int, int]):
        """
        Initialize the ArcBall controller.

        Args:
            camera_position: Camera eye position [x, y, z]
            view_center: Point the camera is looking at [x, y, z]
            up_dir: Up direction vector [x, y, z]
            fov_degrees: Field of view in degrees
            window_size: Window dimensions (width, height)
        """
        self._fov = np.radians(fov_degrees)
        self._window_size = np.array(window_size, dtype=np.int32)

        self._prev_mouse_pos_ndc = np.zeros(2, dtype=np.float32)
        self._lagging = 0.0

        # Target transformations (where camera is moving to)
        self._target_position = np.zeros(3, dtype=np.float32)
        self._target_rotation = np.array([0, 0, 0, 1], dtype=np.float32)  # quaternion [x, y, z, w]
        self._target_zooming = 0.0

        # Current transformations (interpolated)
        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        self._current_zooming = 0.0

        # Initial transformations (for reset)
        self._position_t0 = np.zeros(3, dtype=np.float32)
        self._rotation_t0 = np.array([0, 0, 0, 1], dtype=np.float32)
        self._zooming_t0 = 0.0

        # View transformations
        self._view_matrix = np.eye(4, dtype=np.float32)
        self._inverse_view_matrix = np.eye(4, dtype=np.float32)

        # Initialize camera parameters
        self.set_view_parameters(camera_position, view_center, up_dir)

    def set_view_parameters(self,
                           eye: np.ndarray,
                           view_center: np.ndarray,
                           up_dir: np.ndarray):
        """
        Set the camera view parameters.

        Args:
            eye: Camera eye position [x, y, z]
            view_center: Point to look at [x, y, z]
            up_dir: Up direction vector [x, y, z]
        """
        eye = np.array(eye, dtype=np.float32)
        view_center = np.array(view_center, dtype=np.float32)
        up_dir = np.array(up_dir, dtype=np.float32)

        # Compute camera basis vectors
        direction = view_center - eye
        z_axis = direction / np.linalg.norm(direction)
        x_axis = np.cross(z_axis, up_dir)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(z_axis, y_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Create rotation matrix (column vectors) Left-Handed CS
        rotation_matrix = np.column_stack([x_axis, y_axis, -z_axis])

        # Convert rotation matrix to quaternion
        self._target_rotation = self._matrix_to_quaternion(rotation_matrix.T)
        self._target_position = -view_center
        self._target_zooming = -np.linalg.norm(direction)

        # Initialize current and initial states
        self._current_position = self._target_position.copy()
        self._current_rotation = self._target_rotation.copy()
        self._current_zooming = self._target_zooming

        self._position_t0 = self._target_position.copy()
        self._rotation_t0 = self._target_rotation.copy()
        self._zooming_t0 = self._target_zooming

        self._update_internal_transformations()

    def reset(self):
        """Reset camera to initial position, view center, and up direction."""
        self._target_position = self._position_t0.copy()
        self._target_rotation = self._rotation_t0.copy()
        self._target_zooming = self._zooming_t0

    def reshape(self, window_size: Tuple[int, int]):
        """Update screen size after window has been resized."""
        self._window_size = np.array(window_size, dtype=np.int32)

    def set_lagging(self, lagging: float):
        """
        Set the amount of lagging for smooth camera motion.

        Args:
            lagging: Lagging factor in [0, 1). Higher values = slower motion.
        """
        assert 0.0 <= lagging < 1.0, "Lagging must be in [0, 1)"
        self._lagging = lagging

    @property
    def lagging(self) -> float:
        """Get the current lagging factor."""
        return self._lagging

    def init_transformation(self, mouse_pos: Tuple[int, int]):
        """
        Initialize transformation with first mouse position.
        Call this in mouse pressed event.

        Args:
            mouse_pos: Screen coordinates (x, y)
        """
        self._prev_mouse_pos_ndc = self._screen_coord_to_ndc(mouse_pos)

    def rotate(self, mouse_pos: Tuple[int, int]):
        """
        Rotate camera from previous to current mouse position.

        Args:
            mouse_pos: Current screen coordinates (x, y)
        """
        mouse_pos_ndc = self._screen_coord_to_ndc(mouse_pos)

        current_q = self._ndc_to_arcball(mouse_pos_ndc)
        prev_q = self._ndc_to_arcball(self._prev_mouse_pos_ndc)
        self._prev_mouse_pos_ndc = mouse_pos_ndc

        # Compose rotations: current * prev * target
        q_rotation = self._quaternion_multiply(
            self._quaternion_multiply(current_q, prev_q),
            self._target_rotation
        )
        q_rotation = q_rotation / np.linalg.norm(q_rotation)

        self._target_rotation = q_rotation

    def translate(self, mouse_pos: Tuple[int, int]):
        """
        Translate camera from previous to current mouse position.

        Args:
            mouse_pos: Current screen coordinates (x, y)
        """
        mouse_pos_ndc = self._screen_coord_to_ndc(mouse_pos)
        translation_ndc = mouse_pos_ndc - self._prev_mouse_pos_ndc
        self._prev_mouse_pos_ndc = mouse_pos_ndc
        self.translate_delta(translation_ndc)

    def translate_delta(self, translation_ndc: np.ndarray):
        """
        Translate camera by delta amount in NDC coordinates.

        Args:
            translation_ndc: Translation in NDC space [-1, 1]
        """
        # Half size of screen viewport at view center perpendicular to view direction
        hh = abs(self._target_zooming) * np.tan(self._fov * 0.5)
        hw = hh * (self._window_size[0] / self._window_size[1])

        # Transform translation to world space
        translation_world = np.array([
            translation_ndc[0] * hw,
            translation_ndc[1] * hh,
            0.0
        ], dtype=np.float32)

        # Apply inverse view rotation to get world-space translation
        rotation_matrix = self._quaternion_to_matrix(self._current_rotation)
        inv_rotation = rotation_matrix.T  # Transpose for inverse of rotation matrix
        translation_world = inv_rotation @ translation_world

        self._target_position += translation_world

    def zoom(self, delta: float):
        """
        Zoom the camera (positive = zoom in, negative = zoom out).

        Args:
            delta: Zoom amount
        """
        self._target_zooming += delta

    def update_transformation(self) -> bool:
        """
        Update any unfinished transformation due to lagging.

        Returns:
            True if camera matrices have changed, False otherwise
        """
        # Compute differences
        diff_position = self._target_position - self._current_position
        diff_rotation_q = self._target_rotation - self._current_rotation
        diff_zooming = self._target_zooming - self._current_zooming

        d_position = np.dot(diff_position, diff_position)
        d_rotation = np.dot(diff_rotation_q, diff_rotation_q)
        d_zooming = diff_zooming * diff_zooming

        # Nothing changed
        if d_position < 1.0e-10 and d_rotation < 1.0e-10 and d_zooming < 1.0e-10:
            return False

        # Nearly done: jump directly to target
        if d_position < 1.0e-6 and d_rotation < 1.0e-6 and d_zooming < 1.0e-6:
            self._current_position = self._target_position.copy()
            self._current_rotation = self._target_rotation.copy()
            self._current_zooming = self._target_zooming
        else:
            # Interpolate between current and target
            t = 1.0 - self._lagging
            self._current_position = self._lerp(self._current_position, self._target_position, t)
            self._current_zooming = self._lerp(self._current_zooming, self._target_zooming, t)
            self._current_rotation = self._slerp(
                self._current_rotation,
                self._target_rotation,
                t
            )

        self._update_internal_transformations()
        return True

    def view_distance(self) -> float:
        """Return distance from camera position to view center."""
        return abs(self._target_zooming)

    def view_matrix(self) -> np.ndarray:
        """Get camera's view matrix (4x4)."""
        return self._view_matrix.copy()

    def inverse_view_matrix(self) -> np.ndarray:
        """Get camera's inverse view matrix (4x4)."""
        return self._inverse_view_matrix.copy()

    # Private helper methods

    def _update_internal_transformations(self):
        """Update the internal view and inverse view transformations."""
        # Build view transformation: T(zoom) * R(rotation) * T(position)

        # Create translation matrices
        t_position = np.eye(4, dtype=np.float32)
        t_position[:3, 3] = self._current_position

        t_zoom = np.eye(4, dtype=np.float32)
        t_zoom[2, 3] = self._current_zooming

        # Create rotation matrix from quaternion
        r_rotation = np.eye(4, dtype=np.float32)
        r_rotation[:3, :3] = self._quaternion_to_matrix(self._current_rotation)

        # Compose: view = t_zoom @ r_rotation @ t_position
        self._view_matrix = t_zoom @ r_rotation @ t_position
        self._inverse_view_matrix = np.linalg.inv(self._view_matrix)

    def _screen_coord_to_ndc(self, mouse_pos: Tuple[int, int]) -> np.ndarray:
        """
        Transform screen coordinates to NDC (Normalized Device Coordinates).
        Top-left = [-1, 1], bottom-right = [1, -1]

        Args:
            mouse_pos: Screen coordinates (x, y)

        Returns:
            NDC coordinates [x, y]
        """
        x = mouse_pos[0] * 2.0 / self._window_size[0] - 1.0
        y = 1.0 - 2.0 * mouse_pos[1] / self._window_size[1]
        return np.array([x, y], dtype=np.float32)

    def _ndc_to_arcball(self, p: np.ndarray) -> np.ndarray:
        """
        Project a point in NDC onto the arcball sphere.

        Args:
            p: NDC coordinates [x, y]

        Returns:
            Quaternion [x, y, z, w]
        """
        dist = np.dot(p, p)

        # Point is on sphere
        if dist <= 1.0:
            return np.array([p[0], p[1], np.sqrt(1.0 - dist), 0.0], dtype=np.float32)

        # Point is outside sphere - project onto edge
        proj = p / np.linalg.norm(p)
        return np.array([proj[0], proj[1], 0.0, 0.0], dtype=np.float32)

    def _matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to quaternion [x, y, z, w].

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Quaternion [x, y, z, w]
        """
        m = rotation_matrix
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w], dtype=np.float32)

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        Args:
            q: Quaternion [x, y, z, w]

        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = q

        m = np.zeros((3, 3), dtype=np.float32)
        m[0, 0] = 1 - 2*y*y - 2*z*z
        m[0, 1] = 2*x*y - 2*z*w
        m[0, 2] = 2*x*z + 2*y*w
        m[1, 0] = 2*x*y + 2*z*w
        m[1, 1] = 1 - 2*x*x - 2*z*z
        m[1, 2] = 2*y*z - 2*x*w
        m[2, 0] = 2*x*z - 2*y*w
        m[2, 1] = 2*y*z + 2*x*w
        m[2, 2] = 1 - 2*x*x - 2*y*y

        return m

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.

        Args:
            q1: First quaternion [x, y, z, w]
            q2: Second quaternion [x, y, z, w]

        Returns:
            Product quaternion [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dtype=np.float32)

    @staticmethod
    def _lerp(a, b, t):
        """Linear interpolation between a and b."""
        return a * (1.0 - t) + b * t

    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two quaternions (shortest path).

        Args:
            q1: Start quaternion [x, y, z, w]
            q2: End quaternion [x, y, z, w]
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion
        """
        # Normalize inputs
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Compute dot product
        dot = np.dot(q1, q2)

        # If negative, use -q2 for shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result = result / np.linalg.norm(result)
            return result

        # Standard slerp
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        result = w1 * q1 + w2 * q2
        result = result / np.linalg.norm(result)
        return result
