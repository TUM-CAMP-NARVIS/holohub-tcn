"""Thin ctypes wrapper for the RenderDoc in-app API (RENDERDOC_API_1_6_0).

Usage — load librenderdoc.so *before* creating a Vulkan device, then bracket
your rendering with start_capture/end_capture:

    api = RenderDocAPI()
    api.set_capture_path("/tmp/my_capture")
    # ... create device, build scene ...
    api.start_capture()
    # ... render one frame ...
    api.end_capture()
    path = api.get_capture(0)  # -> "/tmp/my_capture_frame0.rdc"
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# RenderDoc API version enum value for 1.6.0
eRENDERDOC_API_Version_1_6_0 = 10600

# RENDERDOC_API_1_6_0 has 27 function-pointer slots (including unions that
# occupy a single pointer each).  We model the struct as a flat array of
# void-pointer-sized slots so we can index into it by ordinal.
_NUM_SLOTS = 27

# Slot indices (from renderdoc_app.h)
_SLOT_GET_API_VERSION = 0
_SLOT_SET_CAPTURE_FILE_PATH_TEMPLATE = 11
_SLOT_GET_CAPTURE_FILE_PATH_TEMPLATE = 12
_SLOT_GET_NUM_CAPTURES = 13
_SLOT_GET_CAPTURE = 14
_SLOT_START_FRAME_CAPTURE = 19
_SLOT_IS_FRAME_CAPTURING = 20
_SLOT_END_FRAME_CAPTURE = 21

# C function-pointer typedefs
DEVICE = ctypes.c_void_p
WINDOW = ctypes.c_void_p

_SetCaptureFilePathTemplate = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
_GetCaptureFilePathTemplate = ctypes.CFUNCTYPE(ctypes.c_char_p)
_GetNumCaptures = ctypes.CFUNCTYPE(ctypes.c_uint32)
_GetCapture = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_uint32,            # idx
    ctypes.c_char_p,            # filename (out, may be NULL)
    ctypes.POINTER(ctypes.c_uint32),  # pathlength (out, may be NULL)
    ctypes.POINTER(ctypes.c_uint64),  # timestamp (out, may be NULL)
)
_StartFrameCapture = ctypes.CFUNCTYPE(None, DEVICE, WINDOW)
_IsFrameCapturing = ctypes.CFUNCTYPE(ctypes.c_uint32)
_EndFrameCapture = ctypes.CFUNCTYPE(ctypes.c_uint32, DEVICE, WINDOW)
_GetAPIVersion = ctypes.CFUNCTYPE(None,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
)


def _find_librenderdoc() -> str:
    """Locate librenderdoc.so using RENDERDOC_PYTHON_PATH or common paths."""
    env = os.environ.get("RENDERDOC_PYTHON_PATH", "")
    if env:
        candidate = Path(env) / "librenderdoc.so"
        if candidate.exists():
            return str(candidate)
        # Maybe the env var points directly to the .so
        if Path(env).is_file() and env.endswith(".so"):
            return env

    for p in [
        Path.home() / ".local" / "renderdoc" / "librenderdoc.so",
        Path("/usr/lib/renderdoc/librenderdoc.so"),
        Path("/usr/local/lib/renderdoc/librenderdoc.so"),
    ]:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find librenderdoc.so. Set RENDERDOC_PYTHON_PATH or install RenderDoc."
    )


class RenderDocAPI:
    """Thin ctypes wrapper for the RenderDoc in-app API."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_librenderdoc()
        log.info("Loading librenderdoc from %s", path)
        self._lib = ctypes.CDLL(path)

        # RENDERDOC_GetAPI(version, &api) -> int
        get_api = self._lib.RENDERDOC_GetAPI
        get_api.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        get_api.restype = ctypes.c_int

        api_ptr = ctypes.c_void_p()
        ret = get_api(eRENDERDOC_API_Version_1_6_0, ctypes.byref(api_ptr))
        if ret != 1:
            raise RuntimeError(f"RENDERDOC_GetAPI returned {ret} (expected 1)")

        # Cast the returned pointer to an array of function pointers
        slot_array_t = ctypes.c_void_p * _NUM_SLOTS
        self._slots = ctypes.cast(api_ptr, ctypes.POINTER(slot_array_t)).contents

        # Verify we got a valid API by reading the version
        get_ver = ctypes.cast(self._slots[_SLOT_GET_API_VERSION], _GetAPIVersion)
        major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        get_ver(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        log.info("RenderDoc API version: %d.%d.%d", major.value, minor.value, patch.value)

    def set_capture_path(self, path_template: str) -> None:
        """Set the file path template for captures (without .rdc extension)."""
        fn = ctypes.cast(
            self._slots[_SLOT_SET_CAPTURE_FILE_PATH_TEMPLATE],
            _SetCaptureFilePathTemplate,
        )
        fn(path_template.encode("utf-8"))

    def get_capture_path_template(self) -> str:
        """Get the current capture file path template."""
        fn = ctypes.cast(
            self._slots[_SLOT_GET_CAPTURE_FILE_PATH_TEMPLATE],
            _GetCaptureFilePathTemplate,
        )
        result = fn()
        return result.decode("utf-8") if result else ""

    def start_capture(self) -> None:
        """Start capturing the current frame (call before rendering)."""
        fn = ctypes.cast(self._slots[_SLOT_START_FRAME_CAPTURE], _StartFrameCapture)
        fn(None, None)

    def is_capturing(self) -> bool:
        """Check if a frame capture is currently in progress."""
        fn = ctypes.cast(self._slots[_SLOT_IS_FRAME_CAPTURING], _IsFrameCapturing)
        return fn() != 0

    def end_capture(self) -> bool:
        """End the current frame capture. Returns True on success."""
        fn = ctypes.cast(self._slots[_SLOT_END_FRAME_CAPTURE], _EndFrameCapture)
        return fn(None, None) == 1

    def get_num_captures(self) -> int:
        """Return the number of captures made so far."""
        fn = ctypes.cast(self._slots[_SLOT_GET_NUM_CAPTURES], _GetNumCaptures)
        return fn()

    def get_capture(self, index: int = 0) -> str | None:
        """Return the file path of capture at the given index, or None."""
        fn = ctypes.cast(self._slots[_SLOT_GET_CAPTURE], _GetCapture)

        # First call to get path length
        path_len = ctypes.c_uint32()
        fn(index, None, ctypes.byref(path_len), None)

        if path_len.value == 0:
            return None

        # Second call to get the actual path
        buf = ctypes.create_string_buffer(path_len.value)
        timestamp = ctypes.c_uint64()
        fn(index, buf, ctypes.byref(path_len), ctypes.byref(timestamp))
        return buf.value.decode("utf-8")
