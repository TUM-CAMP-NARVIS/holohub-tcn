import re
import logging
from typing import Any, Dict, Optional

import holoscan as hs
import numpy as np
from holohub.tcn_depthimage_backprojection import TcnDepthImageBackprojectionOp
from holohub.tcn_depthimage_backprojection._tcn_depthimage_backprojection import CameraModel, DistortionType, \
    RigidTransform, CameraParameters, make_rigid_transform
from holoscan.core import DefaultFragmentService
from holoscan.operators import HolovizOp
from pyxylt import create_xy_lookup_table_from_intrinsics, IntrinsicParameters

log = logging.getLogger("TcnDeviceContext")

class DeviceContextService(DefaultFragmentService):
    """A simple fragment service that holds an integer value."""

    def __init__(self, device_contexts: Dict[str, Any]):
        super().__init__()
        self._device_contexts = device_contexts
        self.portname_cameraname_match = re.compile("^(camera[0-9]+)_.*$")

    def get_camera_name_from_port_name(self, port_name: str) -> Optional[str]:
        m = self.portname_cameraname_match.match(port_name)
        if m is not None:
            return m.group(1)
        return None

    def raw_context(self) -> Dict[str, Any]:
        """Get the value stored in the service."""
        return self._device_contexts

    def get_device_context(self, camera_name: str) -> Dict[str, Any]:
        if camera_name not in self._device_contexts:
            log.error(f"no camera found with name: {camera_name}")
            return None
        return self._device_contexts[camera_name]

    def get_device_calibration(self, camera_name: str) -> Dict[str, Any]:
        ctx = self.get_device_context(camera_name)
        if ctx is None:
            return None
        if "calibration" not in ctx:
            log.error(f"no calibration found for camera_name: {camera_name}")
            return None
        return ctx["calibration"]

    def get_depth_camera_model(self, camera_name: str) -> Optional[CameraModel]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "depthCameraParameters" not in calib:
            log.error(f"no depthCameraParameters found for camera_name: {camera_name}")
            return None
        params = calib["depthCameraParameters"]
        return self._camera_model_from_dict(params)

    def get_color_camera_model(self, camera_name: str) -> Optional[CameraModel]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "colorCameraParameters" not in calib:
            log.error(f"no colorCameraParameters found for camera_name: {camera_name}")
            return None
        params = calib["colorCameraParameters"]
        return self._camera_model_from_dict(params)

    def _camera_model_from_dict(self, params) -> CameraModel:
        model = CameraModel()
        model.distortion_type = DistortionType.Brown
        model.dimensions.x = params["width"]
        model.dimensions.y = params["height"]
        model.focal_length.x = params["fovX"]
        model.focal_length.y = params["fovY"]
        model.principal_point.x = params["cX"]
        model.principal_point.y = params["cY"]
        model.skew_value = 1.0
        model.distortion_coefficients = [
            params["distortionParams"]["k1"],
            params["distortionParams"]["k2"],
            params["distortionParams"]["tx"],
            params["distortionParams"]["ty"],
            params["distortionParams"]["k3"],
            params["distortionParams"]["k4"],
            params["distortionParams"]["k5"],
            params["distortionParams"]["k6"],
        ]

        return model

    def get_xy_table_intrinsics(self, camera_name: str) -> Optional[IntrinsicParameters]:
        model = self.get_depth_camera_model(camera_name)
        if model is None:
            return None
        intrinsics = IntrinsicParameters()
        intrinsics.fov_x = float(model.focal_length.x)
        intrinsics.fov_y = float(model.focal_length.y)
        intrinsics.c_x = float(model.principal_point.x)
        intrinsics.c_y = float(model.principal_point.y)
        intrinsics.width = int(model.dimensions.x)
        intrinsics.height = int(model.dimensions.y)
        intrinsics.tangential_distortion = [
            model.distortion_coefficients[2],
            model.distortion_coefficients[3],
        ]
        intrinsics.radial_distortion = [
            model.distortion_coefficients[0],
            model.distortion_coefficients[1],
            model.distortion_coefficients[4],
            model.distortion_coefficients[5],
            model.distortion_coefficients[6],
            model.distortion_coefficients[7],
        ]
        return intrinsics

    def get_xy_table(self, camera_name: str) -> Optional[np.ndarray]:
        intrinsics : Optional[IntrinsicParameters] = self.get_xy_table_intrinsics(camera_name)
        if intrinsics is None:
            return None
        log.info(f"Creating xy-table for {camera_name}")
        xy_lookup_table = create_xy_lookup_table_from_intrinsics(intrinsics)

        if not xy_lookup_table:
            raise RuntimeError(f"Failed to create XY lookup table for camera {camera_name}")
        if xy_lookup_table.width == 0 or xy_lookup_table.height == 0 or len(xy_lookup_table.data) == 0:
            raise RuntimeError(f"XY lookup table for camera {camera_name} is empty despite success=True")

        return np.array(xy_lookup_table.data, dtype=np.float32).reshape((xy_lookup_table.height, xy_lookup_table.width, 2))

    def get_depth_extrinsics(self, camera_name: str) -> Optional[RigidTransform]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "cameraPose" not in calib:
            log.error(f"no cameraPose found for camera_name: {camera_name}")
            return None
        params = calib["cameraPose"]
        return self._pose3d_from_dict(params)

    def get_color_to_depth(self, camera_name: str) -> Optional[RigidTransform]:
        calib = self.get_device_calibration(camera_name)
        if calib is None:
            return None
        if "color2depthTransform" not in calib:
            log.error(f"no color2depthTransform found for camera_name: {camera_name}")
            return None
        params = calib["color2depthTransform"]
        return self._pose3d_from_dict(params)


    def _pose3d_from_dict(self, params) -> Optional[RigidTransform]:
        return make_rigid_transform(
            np.asarray([
                params["translation"]["x"],
                params["translation"]["y"],
                params["translation"]["z"],
            ]),
            np.asarray([
                params["rotation"]["x"],
                params["rotation"]["y"],
                params["rotation"]["z"],
                params["rotation"]["w"],
            ])
        )

