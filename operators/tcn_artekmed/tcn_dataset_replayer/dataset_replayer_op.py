"""TcnDatasetReplayerOp: replays an artekmed dataset export in place of TcnShmSubscriberOp.

See `operators/tcn_artekmed/tcn_dataset_replayer/README.md` and the design doc at
`applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-10-replay-harness-design.md`
(Part 1) for the full rationale. In short: the live SHM subscriber makes pipeline output
non-reproducible, which blocks every correctness gate (mask-IoU thresholds, byte-identity
checks). This operator preloads a fixed, on-disk dataset export and emits it deterministically
through the exact same two ports the subscriber uses, so it is a drop-in replacement at the
`compose()` call site.

Only `holoscan`/`cupy` are imported at module scope (both required just to subclass
`holoscan.core.Operator` and to build device tensors -- same as every other operator in
`operators/tcn_artekmed/tcn_util/`). `artekmed_dataset_reader` is imported lazily inside
`start()` because it pulls in `pyarrow`/`tifffile`/`PIL`, which are not installed on a plain
host; keeping that import lazy is what lets `_planning.py`'s arithmetic be tested outside the
container.
"""
import logging
from typing import Any, Dict, List, Optional

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.conditions import AsynchronousEventState
from holoscan.core import Operator, OperatorSpec

from ._planning import plan_frame_sequence

log = logging.getLogger("TcnDatasetReplayerOp")

_VALID_PLAYBACK = ("auto", "manual")
_VALID_CHANNEL_ORDER = ("bgr", "rgb")


def _pinned_copy(source: np.ndarray) -> np.ndarray:
    """Copy `source` into freshly allocated CUDA pinned host memory; return a numpy view onto it.

    Pinned (page-locked) host memory is what makes the per-`compute()` H2D copy a real DMA
    transfer instead of the slower pageable-memory path (design doc §1.2) -- it also matches
    the subscriber's own H2D pattern, so the harness exercises a realistic upload path. The
    returned ndarray's `.base` holds a reference to the underlying `PinnedMemory` object (via
    `np.frombuffer`), which keeps it alive for as long as the view is reachable.
    """
    mem = cp.cuda.alloc_pinned_memory(source.nbytes)
    view = np.frombuffer(mem, dtype=source.dtype, count=source.size).reshape(source.shape)
    view[...] = source
    return view


class TcnDatasetReplayerOp(Operator):
    """Emits preloaded frames from an artekmed dataset export, impersonating TcnShmSubscriberOp.

    **==Named Outputs==**

        color_outputs : dict of ``{f"{camera_id}_colorimage": <device uint8 (H, W, 3)>}``
        depth_outputs : dict of ``{f"{camera_id}_depthimage": <device uint16 (H, W, 1)>}``
            when ``emit_depth``, else an empty dict (so the port stays wired for a downstream
            operator that always reads it).

    **Channel order.** ``artekmed_dataset_reader`` decodes JPEGs as RGB, but the live SHM
    subscriber delivers BGR (the raw camera wire format) and every downstream consumer
    (``GdinoOp`` etc.) does ``[..., [2, 1, 0]]`` itself to recover RGB. This operator therefore
    defaults to ``channel_order="bgr"`` and reverses the reader's channel axis ONCE, at preload
    time, so its output is byte-identical to what the subscriber would have produced. **Do not
    "fix" this to emit RGB** -- that would make every consumer swap an already-correct image,
    quietly corrupting anything gated on this operator's output (design doc §1.3).

    **Preload, then upload.** All selected frames for all selected cameras are decoded once, at
    `start()`, into pinned host memory. Each `compute()` does one H2D copy per camera into a
    freshly allocated device array; no JPEG decoding happens on the hot path.

    Accepts (and ignores) `allocator` / `cuda_stream_pool` keyword arguments purely so that
    swapping this operator in for `TcnShmSubscriberOp` at a `compose()` call site is a one-line
    change -- this operator manages its own pinned host buffers and device allocations directly
    via cupy and does not need a Holoscan `Allocator` resource or a shared `CudaStreamPool`.
    """

    def __init__(
        self,
        fragment,
        *args,
        dataset_path: str,
        cameras: Optional[List[str]] = None,
        start_frame: int = 0,
        frame_count: Optional[int] = None,
        frame_step: int = 1,
        loop: bool = True,
        emit_depth: bool = False,
        channel_order: str = "bgr",
        playback: str = "auto",
        max_preload_frames: int = 64,
        async_condition: Any = None,
        allocator: Any = None,
        cuda_stream_pool: Any = None,
        **kwargs,
    ):
        self.dataset_path = str(dataset_path)
        self._cameras_param = list(cameras) if cameras is not None else None
        self.start_frame = int(start_frame)
        self.frame_count = None if frame_count is None else int(frame_count)
        self.frame_step = int(frame_step)
        self.loop = bool(loop)
        self.emit_depth = bool(emit_depth)

        if channel_order not in _VALID_CHANNEL_ORDER:
            raise ValueError(
                f"TcnDatasetReplayerOp: channel_order must be one of {_VALID_CHANNEL_ORDER}, "
                f"got {channel_order!r}"
            )
        self.channel_order = channel_order

        if playback not in _VALID_PLAYBACK:
            raise ValueError(
                f"TcnDatasetReplayerOp: playback must be one of {_VALID_PLAYBACK}, "
                f"got {playback!r}"
            )
        self.playback = playback
        self.max_preload_frames = int(max_preload_frames)

        # See class docstring: accepted only for drop-in construction-site compatibility.
        del allocator, cuda_stream_pool

        self._async_condition = async_condition

        # Populated in start().
        self._camera_ids: List[str] = []
        self._plan: List[int] = []
        self._color_pinned: Dict[int, Dict[str, np.ndarray]] = {}
        self._depth_pinned: Dict[int, Dict[str, np.ndarray]] = {}
        self._tick = 0
        self._frame_index: Optional[int] = None
        self._loop_count = 0
        self._exhausted_warned = False

        # Need to call the base class constructor last (tcn_util convention).
        super().__init__(fragment, *args, **kwargs)

    # --- read-only status, so a harness can label dumps with the true source frame number ---

    @property
    def frame_index(self) -> Optional[int]:
        """The source dataset frame NUMBER most recently emitted (None before the first tick)."""
        return self._frame_index

    @property
    def loop_count(self) -> int:
        """How many times the preloaded selection has fully wrapped (0 during the first pass)."""
        return self._loop_count

    def initialize(self):
        # Mirror TcnShmSubscriberOp::initialize(): register the AsynchronousCondition as a
        # scheduling condition ONLY for manual playback -- in "auto" mode compute() should fire
        # on every scheduler tick (bounded externally by a CountCondition), not be gated by an
        # async condition nobody is driving.
        if self.playback == "manual" and self._async_condition is not None:
            self.add_arg(self._async_condition)
        Operator.initialize(self)

    def setup(self, spec: OperatorSpec):
        spec.output("color_outputs")
        spec.output("depth_outputs")

    def start(self):
        # Imported lazily: pyarrow/tifffile/PIL are not available on a plain host, only inside
        # the container. Importing at module scope would make this whole module (and therefore
        # `_planning`, which the host tests import via this package) unimportable outside it.
        from artekmed_dataset_reader import open_dataset

        dataset = open_dataset(self.dataset_path)

        if self._cameras_param:
            unknown = [c for c in self._cameras_param if c not in dataset.camera_ids]
            if unknown:
                raise ValueError(
                    f"TcnDatasetReplayerOp: camera(s) {unknown} not found in dataset at "
                    f"{self.dataset_path!r}; available: {dataset.camera_ids}"
                )
            self._camera_ids = list(self._cameras_param)
        else:
            self._camera_ids = list(dataset.camera_ids)
        if not self._camera_ids:
            raise ValueError(f"TcnDatasetReplayerOp: dataset at {self.dataset_path!r} has no cameras")

        # `loop=False, ticks=None` here on purpose: this yields the fixed set of frames to
        # decode and preload ONCE. Looping across ticks (an unbounded, run-length-dependent
        # concept Holoscan doesn't tell us up front) is handled separately in compute() by
        # cycling this fixed list with a modulo -- see `self._tick`.
        self._plan = plan_frame_sequence(
            dataset.frame_numbers,
            start=self.start_frame,
            count=self.frame_count,
            step=self.frame_step,
            loop=False,
            ticks=None,
        )
        if len(self._plan) > self.max_preload_frames:
            raise ValueError(
                f"TcnDatasetReplayerOp: selection has {len(self._plan)} frame(s), exceeding "
                f"max_preload_frames={self.max_preload_frames}. Reduce frame_count/frame_step "
                f"or raise max_preload_frames."
            )

        total_bytes = 0
        self._color_pinned = {}
        self._depth_pinned = {}
        for frame_number in self._plan:
            frame_group = dataset.frame(frame_number)

            color_per_camera: Dict[str, np.ndarray] = {}
            for camera_id in self._camera_ids:
                rgb = frame_group.color_image(camera_id)  # RGB uint8 (H, W, 3)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(
                        f"TcnDatasetReplayerOp: expected a 3-channel color image for "
                        f"{camera_id!r} at frame {frame_number}, got shape {rgb.shape}"
                    )
                if self.channel_order == "bgr":
                    # THE correctness trap (design doc §1.3): the reader returns RGB, but the
                    # live subscriber delivers BGR straight off the wire, and downstream
                    # consumers (GdinoOp.compute et al.) do `[..., [2, 1, 0]]` themselves to
                    # undo that. Reversing once, here, at preload time, makes our *output*
                    # match the subscriber's; emitting RGB instead would make every consumer's
                    # swap turn a correct image into a wrong one -- plausible-looking, silently
                    # broken, and it would invalidate every gate built on top of this operator.
                    frame_data = np.ascontiguousarray(rgb[..., ::-1])
                else:
                    frame_data = rgb
                color_per_camera[camera_id] = _pinned_copy(frame_data)
                total_bytes += frame_data.nbytes
            self._color_pinned[frame_number] = color_per_camera

            if self.emit_depth:
                depth_per_camera: Dict[str, np.ndarray] = {}
                for camera_id in self._camera_ids:
                    depth = frame_group.depth_image(camera_id)  # uint16 (H, W)
                    if depth.ndim == 2:
                        depth = depth[..., np.newaxis]  # -> (H, W, 1), matches the subscriber
                    elif depth.ndim != 3 or depth.shape[2] != 1:
                        raise ValueError(
                            f"TcnDatasetReplayerOp: expected a single-channel depth image for "
                            f"{camera_id!r} at frame {frame_number}, got shape {depth.shape}"
                        )
                    depth_per_camera[camera_id] = _pinned_copy(np.ascontiguousarray(depth))
                    total_bytes += depth.nbytes
                self._depth_pinned[frame_number] = depth_per_camera

        log.info(
            "TcnDatasetReplayerOp: preloaded %d frame(s) x %d camera(s) (depth=%s), "
            "%.1f MiB pinned host memory",
            len(self._plan), len(self._camera_ids), self.emit_depth, total_bytes / (1024 * 1024),
        )

        self._tick = 0
        self._frame_index = None
        self._loop_count = 0
        self._exhausted_warned = False

        if self.playback == "manual" and self._async_condition is not None:
            self._async_condition.event_state = AsynchronousEventState.EVENT_WAITING

    def stop(self):
        if self.playback == "manual" and self._async_condition is not None:
            self._async_condition.event_state = AsynchronousEventState.EVENT_NEVER
        # Release pinned host memory.
        self._color_pinned = {}
        self._depth_pinned = {}

    def request_next(self):
        """Manual-playback step: arm the async condition so the scheduler runs one compute().

        Mirrors `TcnShmSubscriberOp`'s receiver thread notifying the scheduler once a frame is
        ready (`shm_subscriber_op.cpp::receiver_mainloop`): only flips EVENT_WAITING ->
        EVENT_DONE, so calling this while a compute() is still pending is a harmless no-op.
        """
        if self.playback != "manual":
            raise RuntimeError("TcnDatasetReplayerOp.request_next(): only valid when playback='manual'")
        if self._async_condition is None:
            raise RuntimeError(
                "TcnDatasetReplayerOp.request_next(): requires an async_condition to have been "
                "passed in at construction"
            )
        if self._async_condition.event_state == AsynchronousEventState.EVENT_WAITING:
            self._async_condition.event_state = AsynchronousEventState.EVENT_DONE

    def compute(self, op_input, op_output, context):
        n = len(self._plan)
        if n == 0:
            log.warning("TcnDatasetReplayerOp: compute() called with an empty frame plan")
            return

        seq = self._tick
        if not self.loop and seq >= n:
            if not self._exhausted_warned:
                log.warning(
                    "TcnDatasetReplayerOp: non-looping selection of %d frame(s) exhausted; "
                    "compute() will stop emitting. Bound the run to <= %d ticks (e.g. "
                    "CountCondition(app, count=%d)), or set loop=True.",
                    n, n, n,
                )
                self._exhausted_warned = True
            return

        index = seq % n
        self._loop_count = seq // n
        frame_number = self._plan[index]
        self._frame_index = frame_number
        self._tick += 1

        color_message = {}
        for camera_id in self._camera_ids:
            host_array = self._color_pinned[frame_number][camera_id]
            device_array = cp.empty(host_array.shape, dtype=host_array.dtype)
            device_array.set(host_array)
            color_message[f"{camera_id}_colorimage"] = hs.as_tensor(device_array)

        depth_message = {}
        if self.emit_depth:
            for camera_id in self._camera_ids:
                host_array = self._depth_pinned[frame_number][camera_id]
                device_array = cp.empty(host_array.shape, dtype=host_array.dtype)
                device_array.set(host_array)
                depth_message[f"{camera_id}_depthimage"] = hs.as_tensor(device_array)

        op_output.emit(color_message, "color_outputs")
        op_output.emit(depth_message, "depth_outputs")

        if self.playback == "manual" and self._async_condition is not None:
            self._async_condition.event_state = AsynchronousEventState.EVENT_WAITING
