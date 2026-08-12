"""Host tests for the per-camera 180-degree flip helpers. numpy only.

Run directly: `python3 operators/tcn_artekmed/tcn_langsam/tests/test_flip.py`

The property that matters is exact reversibility: the mask produced from a rotated image is rotated
back before it leaves LangSAM, so every geometric consumer downstream sees the camera's native
orientation. A 180-degree rotation is its own inverse and needs no resampling, which is why this is
safe where an arbitrary angle would not be.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import camera_key, resolve_flipped_cameras, validate_flip_cameras

# Import the module file directly, not via the tcn_util package: its __init__ pulls in the cupy
# operators, and rotate.py itself imports nothing -- which is what keeps this a host test.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tcn_util"))
from rotate import rotate180 as flip180   # one implementation, shared with RotateImage180Op


def test_flip_is_its_own_inverse_for_a_map():
    a = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert np.array_equal(flip180(flip180(a)), a)


def test_flip_is_its_own_inverse_for_an_image():
    a = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    assert np.array_equal(flip180(flip180(a)), a)


def test_flip_moves_corners_diagonally():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    assert np.array_equal(flip180(a), np.array([[4, 3], [2, 1]]))


def test_flip_leaves_the_channel_axis_alone():
    """Only the spatial axes rotate -- flipping channels would swap colours, not orientation."""
    a = np.zeros((2, 2, 3), np.uint8)
    a[0, 0] = (10, 20, 30)
    f = flip180(a)
    assert tuple(f[1, 1]) == (10, 20, 30), tuple(f[1, 1])


def test_a_mask_round_trips_to_the_original_pixels():
    """The end-to-end property: label an image feature, then verify the mask lands back on it."""
    image = np.zeros((4, 6), np.uint8)
    image[0, 0] = 255                                    # a feature in the top-left
    rotated = flip180(image)
    assert rotated[3, 5] == 255                          # ...appears bottom-right to the model
    mask_in_rotated_space = (rotated == 255).astype(np.uint16) * 0x0101
    restored = flip180(mask_in_rotated_space)
    assert restored[0, 0] == 0x0101 and restored.sum() == 0x0101, restored


def test_camera_key_accepts_either_naming():
    assert camera_key("camera01") == "camera01"
    assert camera_key("camera01_colorimage") == "camera01"
    assert camera_key(" Camera01_Colorimage ") == "camera01"


def test_resolve_matches_both_forms_and_returns_original_names():
    cams = ["camera01_colorimage", "camera02_colorimage"]
    assert resolve_flipped_cameras(cams, ["camera01"]) == {"camera01_colorimage"}
    assert resolve_flipped_cameras(cams, ["camera02_colorimage"]) == {"camera02_colorimage"}
    assert resolve_flipped_cameras(cams, []) == set()


def test_validate_refuses_an_unknown_camera():
    """A typo must not silently leave a camera unrotated -- that looks like a model problem."""
    try:
        validate_flip_cameras(["camera01_colorimage"], ["camera99"])
    except ValueError as e:
        assert "camera99" in str(e)
        return
    raise AssertionError("an unknown camera name was accepted")


def test_resolve_ignores_cameras_owned_by_another_worker():
    """Per-worker resolution must NOT validate: a worker sees only its share of the cameras."""
    mine = ["camera01_colorimage", "camera02_colorimage"]
    got = resolve_flipped_cameras(mine, ["camera01", "camera03", "camera04"])
    assert got == {"camera01_colorimage"}, got


def test_validate_accepts_the_full_set_a_worker_would_reject():
    all_cams = [f"camera0{i}_colorimage" for i in (1, 2, 3, 4)]
    assert validate_flip_cameras(all_cams, ["camera01", "camera03", "camera04"]) == \
        {"camera01", "camera03", "camera04"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, e)
        except Exception as e:
            bad += 1; print("ERROR", fn.__name__, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
