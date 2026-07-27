"""Capture analysis module using rdc-cli's daemon server.

Opens an .rdc capture file via a local rdc daemon and provides methods to
extract shader uniforms, vertex data, and post-VS output for validation.

Requires rdc-cli to be installed (``pip install -e /path/to/rdc-cli``).
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _send_request(
    host: str,
    port: int,
    payload: dict[str, Any],
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Send a JSON-RPC request and return the parsed response."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(data)
        f = sock.makefile("rb")
        line = f.readline()
        if not line:
            raise OSError("empty response from daemon")
        parsed: dict[str, Any] = json.loads(line.rstrip(b"\n").decode("utf-8"))
        return parsed


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CaptureAnalyzer:
    """Analyze a RenderDoc capture file using rdc-cli's daemon server.

    Starts a daemon subprocess for the capture, communicates via JSON-RPC,
    and provides high-level methods for extracting pipeline data.

    Usage::

        analyzer = CaptureAnalyzer("/path/to/capture.rdc")
        draws = analyzer.get_draw_calls()
        uniforms = analyzer.get_uniforms(draws[0]["eid"])
        analyzer.close()
    """

    def __init__(self, capture_path: str, timeout: float = 30.0):
        self._capture = str(capture_path)
        self._host = "127.0.0.1"
        self._port = _pick_port()
        self._token = secrets.token_hex(16)
        self._request_id = 0

        log.info("Starting rdc daemon for %s on port %d", self._capture, self._port)
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rdc.daemon_server",
                "--host", self._host,
                "--port", str(self._port),
                "--capture", self._capture,
                "--token", self._token,
                "--idle-timeout", "300",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for daemon to become ready
        deadline = time.monotonic() + timeout
        ready = False
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(
                    f"Daemon exited with code {self._proc.returncode}: {stderr}"
                )
            try:
                resp = self._call("ping", {})
                if resp.get("ok") is True:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)

        if not ready:
            self._proc.kill()
            raise RuntimeError(f"Daemon did not start within {timeout}s")

        log.info("Daemon ready for %s", self._capture)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
            "params": {"_token": self._token, **params},
        }
        resp = _send_request(self._host, self._port, payload)
        if "error" in resp:
            raise RuntimeError(
                f"RPC error in {method}: {resp['error'].get('message', resp['error'])}"
            )
        return resp.get("result", {})

    def get_draw_calls(self) -> list[dict[str, Any]]:
        """List all draw calls with EID, type, and triangle count."""
        result = self._call("draws", {})
        return result.get("draws", [])

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """List events, optionally filtered by type."""
        params: dict[str, Any] = {}
        if event_type:
            params["type"] = event_type
        result = self._call("events", params)
        return result.get("events", [])

    def goto(self, eid: int) -> None:
        """Move the replay to a specific event ID."""
        self._call("goto", {"eid": eid})

    def get_uniforms(
        self,
        eid: int,
        stage: str = "vs",
        descriptor_set: int = 0,
        binding: int = 0,
    ) -> dict[str, Any]:
        """Extract constant buffer variables at a draw call.

        Returns dict mapping variable name to its value (scalar, list, or
        nested dict for structs).
        """
        result = self._call("cbuffer_decode", {
            "eid": eid,
            "stage": stage,
            "set": descriptor_set,
            "binding": binding,
        })
        variables = result.get("variables", [])
        out: dict[str, Any] = {}
        for var in variables:
            name = var.get("name", "")
            value = var.get("value")
            var_type = var.get("type", "")
            # Reconstruct matrices from flat float lists
            if value is not None and isinstance(value, list):
                if "mat4" in var_type or "float4x4" in var_type:
                    out[name] = np.array(value, dtype=np.float32).reshape(4, 4)
                elif "mat3" in var_type or "float3x3" in var_type:
                    out[name] = np.array(value, dtype=np.float32).reshape(3, 3)
                else:
                    out[name] = np.array(value, dtype=np.float32)
            else:
                out[name] = value
        return out

    def get_vertex_input(self, eid: int, count: int | None = None) -> dict[str, np.ndarray]:
        """Extract vertex buffer data at a draw call.

        Returns dict mapping attribute semantic (e.g., "POSITION") to a numpy
        array of shape (num_vertices, num_components).
        """
        params: dict[str, Any] = {"eid": eid}
        if count is not None:
            params["count"] = count
        result = self._call("vbuffer_decode", params)
        columns = result.get("columns", [])
        vertices = result.get("vertices", [])

        if not columns or not vertices:
            return {}

        # Group columns by attribute name (before the dot)
        # e.g., ["POSITION.x", "POSITION.y", "POSITION.z", "NORMAL.x", ...]
        attrs: dict[str, list[int]] = {}
        for i, col in enumerate(columns):
            attr_name = col.split(".")[0] if "." in col else col
            attrs.setdefault(attr_name, []).append(i)

        data = np.array(vertices, dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        for attr_name, col_indices in attrs.items():
            out[attr_name] = data[:, col_indices]
        return out

    def get_post_vs(self, eid: int) -> dict[str, np.ndarray]:
        """Extract post-vertex-shader output at a draw call.

        Returns dict mapping output name to numpy array.
        Uses the mesh_data handler for full vertex decoding.
        """
        result = self._call("mesh_data", {"eid": eid, "stage": "vs-out"})
        vertices = result.get("vertices", [])
        comp_count = result.get("comp_count", 4)
        vertex_count = result.get("vertex_count", 0)

        if not vertices:
            return {}

        data = np.array(vertices, dtype=np.float32)
        # mesh_data returns flat rows; first 4 components are SV_Position
        out: dict[str, np.ndarray] = {}
        if data.shape[1] >= 4:
            out["SV_Position"] = data[:, :4]
        if data.shape[1] > 4:
            out["remaining"] = data[:, 4:]

        # Also store indices and topology
        indices = result.get("indices", [])
        if indices:
            out["_indices"] = np.array(indices, dtype=np.int32)

        return out

    def get_pipeline_state(self, eid: int) -> dict[str, Any]:
        """Get pipeline state at an event (viewport, topology, etc.)."""
        state: dict[str, Any] = {}

        # Viewport
        try:
            vp = self._call("pipe_viewport", {"eid": eid})
            state["viewport"] = vp
        except RuntimeError:
            pass

        # Topology
        try:
            topo = self._call("pipe_topology", {"eid": eid})
            state["topology"] = topo.get("topology", "")
        except RuntimeError:
            pass

        # Depth-stencil
        try:
            ds = self._call("pipe_depth_stencil", {"eid": eid})
            state["depth_stencil"] = ds
        except RuntimeError:
            pass

        # Rasterizer
        try:
            rast = self._call("pipe_rasterizer", {"eid": eid})
            state["rasterizer"] = rast
        except RuntimeError:
            pass

        return state

    def get_info(self) -> dict[str, Any]:
        """Get capture metadata (API, event count, etc.)."""
        return self._call("info", {})

    def close(self) -> None:
        """Shutdown the daemon."""
        try:
            self._call("shutdown", {})
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        log.info("Daemon shut down for %s", self._capture)

    def __enter__(self) -> CaptureAnalyzer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
