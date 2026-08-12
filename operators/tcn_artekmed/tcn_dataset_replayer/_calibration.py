"""Pure conversion from an artekmed export's calibration JSON to a Holoscan device-context dict.

Kept separate from `dataset_replayer_op.py` so it can be imported and tested on a host that has
neither `holoscan` nor `cupy` -- same reason as `_planning.py`.

Why this exists: `compose()` sets `device_contexts = {}` for `source: dataset`, so the replay path
has no calibration, no xy_table, and therefore no backprojection -- on the one source that is
deterministic. The export ships per-camera calibration with exactly the fields the device-context
dict needs, under snake_case names instead of the SHM path's camelCase. Converting it lets
`DeviceContextService`, `XYLookupTableSourceOp`, `tcn_depthimage_backprojection` and
`tcn_label_sampler` all run unmodified on replayed frames.

The target shape is defined by `camera_device_info_from_dict` in
`operators/tcn_artekmed/tcn_device_context/python/device_context.cpp`; that function is the
authority, and every key below is read by it.
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence

#: Depth encoding of the capture format (Azure Kinect / Orbbec write millimetres). The export does
#: NOT record this, so it is an assumption rather than data -- it matches
#: `depthimage_backprojection.depth_units_per_meter` in tcn_shm_receiver.yaml, and a wrong value
#: scales every unprojected point linearly.
DEFAULT_DEPTH_UNITS_PER_METER = 1000.0

#: Nominal capture rate. Only reported; nothing in the backprojection path reads it.
DEFAULT_FRAME_RATE = 30.0

#: Eigen serialises a column vector as m00/m10/m20.
_VEC3_KEYS = ("m00", "m10", "m20")


def _vec3(node: Dict[str, Any], where: str) -> Dict[str, float]:
    missing = [k for k in _VEC3_KEYS if k not in node]
    if missing:
        raise ValueError(f"{where}: translation is missing {missing}; expected Eigen keys "
                         f"{list(_VEC3_KEYS)}")
    return {"x": float(node["m00"]), "y": float(node["m10"]), "z": float(node["m20"])}


def _quat(node: Dict[str, Any], where: str) -> Dict[str, float]:
    missing = [k for k in "xyzw" if k not in node]
    if missing:
        raise ValueError(f"{where}: rotation is missing {missing}")
    return {k: float(node[k]) for k in "xyzw"}


def _transform(node: Dict[str, Any], where: str) -> Dict[str, Any]:
    return {"translation": _vec3(node["translation"], where),
            "rotation": _quat(node["rotation"], where)}


def _camera_parameters(node: Dict[str, Any], where: str) -> Dict[str, Any]:
    """One camera's intrinsics in device-context form.

    Distortion ordering is the interesting part. The export stores six radial coefficients
    (`radial_distortion.m00..m50` = k1..k6, the rational model) and two tangential ones
    (`tangential_distortion.m00/m10` = p1/p2). `camera_model_from_dict` reads Brown coefficients in
    the order **k1, k2, tx, ty, k3, k4, k5, k6** -- the tangential pair sits in the MIDDLE. Emitting
    them in the export's own order instead would put p1/p2 where k3/k4 belong, which distorts
    plausibly rather than obviously.
    """
    radial = node.get("radial_distortion", {})
    tangential = node.get("tangential_distortion", {})
    radial_keys = ("m00", "m10", "m20", "m30", "m40", "m50")
    missing = [k for k in radial_keys if k not in radial]
    if missing:
        raise ValueError(f"{where}: radial_distortion is missing {missing}; the Brown rational "
                         f"model needs six coefficients")
    for k in ("m00", "m10"):
        if k not in tangential:
            raise ValueError(f"{where}: tangential_distortion is missing {k}")

    k = [float(radial[key]) for key in radial_keys]
    return {
        "width": int(node["width"]),
        "height": int(node["height"]),
        "fovX": float(node["fov_x"]),
        "fovY": float(node["fov_y"]),
        "cX": float(node["c_x"]),
        "cY": float(node["c_y"]),
        "distortionParams": {
            "k1": k[0], "k2": k[1],
            "tx": float(tangential["m00"]), "ty": float(tangential["m10"]),
            "k3": k[2], "k4": k[3], "k5": k[4], "k6": k[5],
        },
    }


def device_context_from_export(
    calibration: Dict[str, Any],
    depth_units_per_meter: float = DEFAULT_DEPTH_UNITS_PER_METER,
    frame_rate: float = DEFAULT_FRAME_RATE,
    where: str = "calibration",
) -> Dict[str, Any]:
    """Convert one parsed `calibration/<camera>.json` into a device-context dict.

    Accepts either the file's top level (which wraps everything in `value0`, cereal's convention) or
    the unwrapped body, so a caller need not know which it holds.
    """
    body = calibration.get("value0", calibration)
    for key in ("depth_parameters", "color_parameters", "camera_pose", "color2depth_transform"):
        if key not in body:
            raise ValueError(f"{where}: missing '{key}'; this does not look like an artekmed "
                             f"camera calibration")

    # A camera the capture marked invalid has calibration fields present but meaningless. Silently
    # using it produces a point cloud that is wrong in a way only visible against ground truth.
    if not bool(body.get("is_valid", True)):
        raise ValueError(f"{where}: the export marks this camera as not valid; its calibration "
                         f"cannot be used for backprojection")

    return {
        "calibration": {
            "depthCameraParameters": _camera_parameters(body["depth_parameters"],
                                                        f"{where}.depth_parameters"),
            "colorCameraParameters": _camera_parameters(body["color_parameters"],
                                                        f"{where}.color_parameters"),
            "cameraPose": _transform(body["camera_pose"], f"{where}.camera_pose"),
            # Passed through in the export's own direction. The device context inverts it on demand
            # (`get_color_to_depth_inv`), which is what backprojection's `depth_to_color` wants.
            "color2depthTransform": _transform(body["color2depth_transform"],
                                               f"{where}.color2depth_transform"),
        },
        "depthUnitsPerMeter": float(depth_units_per_meter),
        "isValid": True,
        "frameRate": float(frame_rate),
    }


def calibration_path(dataset_path: str, camera_id: str) -> str:
    """Where a camera's calibration lives inside an export."""
    return os.path.join(dataset_path, "calibration", f"{camera_id}.json")


def load_device_contexts(
    dataset_path: str,
    cameras: Sequence[str],
    depth_units_per_meter: float = DEFAULT_DEPTH_UNITS_PER_METER,
    frame_rate: float = DEFAULT_FRAME_RATE,
    missing_ok: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Device-context dict for `cameras`, ready for `DeviceContextService.create()`.

    Raises for a camera whose calibration is absent or unreadable unless `missing_ok`, in which case
    it is omitted. Omission is not silent at the call site: the caller sees a short dict and must
    decide, which is why the geometric path refuses to build for a camera it cannot calibrate.
    """
    contexts: Dict[str, Dict[str, Any]] = {}
    for camera_id in cameras:
        path = calibration_path(dataset_path, camera_id)
        if not os.path.isfile(path):
            if missing_ok:
                continue
            raise FileNotFoundError(
                f"no calibration for camera {camera_id!r} at {path}; the mask/depth join needs "
                f"per-camera intrinsics and extrinsics")
        with open(path, "r") as fh:
            raw = json.load(fh)
        contexts[camera_id] = device_context_from_export(
            raw, depth_units_per_meter=depth_units_per_meter, frame_rate=frame_rate, where=path)
    return contexts
