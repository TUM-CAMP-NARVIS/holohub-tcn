"""Host tests for `_calibration.py` (export calibration -> device-context dict).

Run directly: `python3 operators/tcn_artekmed/tcn_dataset_replayer/tests/test_calibration.py`

Expectations are written against the CONSUMER's contract -- the keys and distortion ordering that
`camera_device_info_from_dict` in `tcn_device_context/python/device_context.cpp` reads -- not by
restating the converter. The distortion ordering is the part worth testing: the export's own order
and the consumer's differ, and getting it wrong distorts plausibly rather than obviously.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _calibration import (DEFAULT_DEPTH_UNITS_PER_METER, device_context_from_export,
                          load_device_contexts)

# Distinct, recognisable values so a mis-mapped field is identifiable by its value alone.
RADIAL = {"m00": 1.0, "m10": 2.0, "m20": 3.0, "m30": 4.0, "m40": 5.0, "m50": 6.0}   # k1..k6
TANGENTIAL = {"m00": 0.5, "m10": 0.25}                                              # p1, p2


def _params(width, height, fx, fy, cx, cy):
    return {"width": width, "height": height, "fov_x": fx, "fov_y": fy, "c_x": cx, "c_y": cy,
            "radial_distortion": dict(RADIAL), "tangential_distortion": dict(TANGENTIAL)}


def _transform(tx, ty, tz, qx, qy, qz, qw):
    return {"translation": {"m00": tx, "m10": ty, "m20": tz},
            "rotation": {"x": qx, "y": qy, "z": qz, "w": qw}}


EXPORT = {"value0": {
    "depth_parameters": _params(640, 576, 505.0, 504.9, 326.6, 331.0),
    "color_parameters": _params(1920, 1080, 1120.7, 1120.8, 946.4, 519.3),
    "camera_pose": _transform(-0.23, -1.35, -1.24, 0.04, 0.90, -0.44, -0.07),
    "color2depth_transform": _transform(0.032, -0.0008, 0.0022, 0.051, 0.0049, 0.0012, 0.999),
    "is_valid": True,
}}


def test_intrinsics_land_on_the_consumers_key_names():
    ctx = device_context_from_export(EXPORT)
    depth = ctx["calibration"]["depthCameraParameters"]
    assert depth["width"] == 640 and depth["height"] == 576
    assert depth["fovX"] == 505.0 and depth["fovY"] == 504.9
    assert depth["cX"] == 326.6 and depth["cY"] == 331.0
    color = ctx["calibration"]["colorCameraParameters"]
    assert color["width"] == 1920 and color["height"] == 1080
    assert color["fovX"] == 1120.7 and color["cY"] == 519.3


def test_tangential_coefficients_sit_between_k2_and_k3():
    # The whole point of the mapping: Brown order is k1, k2, tx, ty, k3, k4, k5, k6.
    d = device_context_from_export(EXPORT)["calibration"]["depthCameraParameters"]["distortionParams"]
    assert d["k1"] == RADIAL["m00"]
    assert d["k2"] == RADIAL["m10"]
    assert d["tx"] == TANGENTIAL["m00"]
    assert d["ty"] == TANGENTIAL["m10"]
    assert d["k3"] == RADIAL["m20"]
    assert d["k4"] == RADIAL["m30"]
    assert d["k5"] == RADIAL["m40"]
    assert d["k6"] == RADIAL["m50"]


def test_all_eight_distortion_keys_are_present():
    d = device_context_from_export(EXPORT)["calibration"]["colorCameraParameters"]["distortionParams"]
    assert set(d) == {"k1", "k2", "tx", "ty", "k3", "k4", "k5", "k6"}


def test_transforms_convert_eigen_vector_keys_to_xyz():
    ctx = device_context_from_export(EXPORT)
    pose = ctx["calibration"]["cameraPose"]
    assert pose["translation"] == {"x": -0.23, "y": -1.35, "z": -1.24}
    assert pose["rotation"]["w"] == -0.07          # negative w must survive, not be normalised away
    c2d = ctx["calibration"]["color2depthTransform"]
    assert c2d["translation"]["x"] == 0.032


def test_unwrapped_body_is_accepted_too():
    assert device_context_from_export(EXPORT["value0"]) == device_context_from_export(EXPORT)


def test_depth_units_default_and_override():
    assert device_context_from_export(EXPORT)["depthUnitsPerMeter"] == DEFAULT_DEPTH_UNITS_PER_METER
    assert device_context_from_export(EXPORT, depth_units_per_meter=1.0)["depthUnitsPerMeter"] == 1.0


def test_invalid_camera_is_refused():
    bad = copy.deepcopy(EXPORT)
    bad["value0"]["is_valid"] = False
    try:
        device_context_from_export(bad)
    except ValueError as e:
        assert "not valid" in str(e)
        return
    raise AssertionError("an is_valid=False camera was accepted")


def test_missing_radial_coefficient_is_refused():
    bad = copy.deepcopy(EXPORT)
    del bad["value0"]["depth_parameters"]["radial_distortion"]["m50"]
    try:
        device_context_from_export(bad)
    except ValueError as e:
        assert "m50" in str(e)
        return
    raise AssertionError("a truncated radial_distortion was accepted")


def test_missing_section_is_refused():
    bad = copy.deepcopy(EXPORT)
    del bad["value0"]["color2depth_transform"]
    try:
        device_context_from_export(bad)
    except ValueError as e:
        assert "color2depth_transform" in str(e)
        return
    raise AssertionError("a calibration without color2depth_transform was accepted")


def test_load_device_contexts_reads_a_directory(tmpdir=None):
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "calibration"))
        for cam in ("camera01", "camera02"):
            with open(os.path.join(root, "calibration", f"{cam}.json"), "w") as fh:
                json.dump(EXPORT, fh)
        got = load_device_contexts(root, ["camera01", "camera02"])
        assert sorted(got) == ["camera01", "camera02"]
        assert got["camera01"]["calibration"]["depthCameraParameters"]["width"] == 640

        try:
            load_device_contexts(root, ["camera09"])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing calibration file was accepted")

        assert load_device_contexts(root, ["camera09"], missing_ok=True) == {}


def test_real_export_calibration_if_present():
    """Runs against the actual test export when it is mounted; skips otherwise."""
    for root in ("/workspace/volumes/artekmed_test_data/data/orbbec_capture",
                 os.path.expanduser("~/develop/artekmed/artekmed_test_data/data/orbbec_capture")):
        if os.path.isdir(os.path.join(root, "calibration")):
            got = load_device_contexts(root, ["camera01"])
            depth = got["camera01"]["calibration"]["depthCameraParameters"]
            assert depth["width"] > 0 and depth["height"] > 0
            assert depth["fovX"] > 0 and depth["fovY"] > 0
            assert set(depth["distortionParams"]) == {"k1", "k2", "tx", "ty", "k3", "k4", "k5", "k6"}
            return
    print("     (skipped: no export mounted)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, repr(e))
        except Exception as e:
            bad += 1; print("ERROR", fn.__name__, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
