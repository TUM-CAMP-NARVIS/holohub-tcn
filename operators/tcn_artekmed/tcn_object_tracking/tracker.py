"""Persistent object identity over time. Pure Python -- no holoscan, no cupy, no numpy.

Turns a per-frame list of `Detection`s into tracks whose ids stay the same for as long as the object
is observed. Deliberately has no appearance model and no motion model beyond "it was here last time":
the input runs at the mask rate (~5 fps) and carries no confidence values, so a Kalman filter would be
fitting noise. What it does have is an explicit lifecycle, so the *meaning* of an id is precise.

Host-testable in milliseconds, which is the point: identity bugs (swapped ids, churn, resurrection)
are invisible in a rendered scene and obvious in a unit test.
"""
import logging
from typing import Dict, List, Optional, Sequence

from .association import Detection, box_union, match_detections_to_tracks

log = logging.getLogger(__name__)


class Track:
    """One tracked object. `track_id` is stable for the object's whole life and never reused."""

    __slots__ = ("track_id", "class_id", "centroid", "box", "num_points", "cameras",
                 "hits", "misses", "age", "first_frame", "last_frame")

    def __init__(self, track_id: int, detection: Detection, frame: int):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.centroid = detection.centroid
        self.box = detection.box
        self.num_points = detection.num_points
        self.cameras = detection.cameras
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.first_frame = frame
        self.last_frame = frame

    @property
    def extent(self):
        return tuple(self.box[1][d] - self.box[0][d] for d in range(3))

    def update(self, detection: Detection, frame: int, box_smoothing: float) -> None:
        self.class_id = detection.class_id
        self.centroid = detection.centroid
        if box_smoothing <= 0.0:
            self.box = detection.box
        else:
            # Exponential smoothing per corner. Depth noise makes a raw box jitter by centimetres
            # frame to frame; smoothing makes the reported extent stable enough to be worth printing.
            a = box_smoothing
            self.box = (
                tuple(a * self.box[0][d] + (1.0 - a) * detection.box[0][d] for d in range(3)),
                tuple(a * self.box[1][d] + (1.0 - a) * detection.box[1][d] for d in range(3)),
            )
        self.num_points = detection.num_points
        self.cameras = detection.cameras
        self.hits += 1
        self.misses = 0
        self.last_frame = frame

    def mark_missed(self) -> None:
        self.misses += 1

    def as_dict(self, class_names: Optional[Dict[int, str]] = None) -> dict:
        name = (class_names or {}).get(self.class_id, f"class_{self.class_id}")
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": name,
            "centroid": self.centroid,
            "bbox_min": self.box[0],
            "bbox_max": self.box[1],
            "extent": self.extent,
            "num_points": self.num_points,
            "cameras": list(self.cameras),
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
        }

    def __repr__(self):
        return (f"Track(id={self.track_id} class={self.class_id} hits={self.hits} "
                f"misses={self.misses})")


class ObjectTracker:
    """Assigns and maintains persistent ids.

    Lifecycle, which is what an id actually means here:

    - a new detection starts a **tentative** track; it is not reported until it has `min_hits`
      observations, so a one-frame false positive never gets an id in the output
    - a track that goes unobserved is kept for `max_age` frames, then **retired**. Its box and
      centroid are held at their last observed values while it waits, so an object that reappears
      near where it vanished re-matches
    - ids are **never reused**, so a printed id refers to exactly one physical object for the
      lifetime of the process

    A returning object that was away longer than `max_age` therefore gets a NEW id. That is a
    deliberate refusal to claim continuity that has not been observed; position-based revival would
    bind the wrong object whenever two of a class swap places out of view.
    """

    def __init__(self, min_hits: int = 3, max_age: int = 8, iou_threshold: float = 0.1,
                 max_centroid_distance_m: float = 1.0, box_smoothing: float = 0.5):
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)
        self.max_centroid_distance_m = float(max_centroid_distance_m)
        self.box_smoothing = float(box_smoothing)
        self._tracks: List[Track] = []
        self._next_id = 1
        self._frame = 0
        self._class_signature: Optional[tuple] = None
        self.retired = 0

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    def confirmed(self) -> List[Track]:
        """Tracks that have earned an identity, in id order so output is stable frame to frame."""
        return sorted((t for t in self._tracks if t.hits >= self.min_hits),
                      key=lambda t: t.track_id)

    def reset(self, reason: str = "") -> None:
        """Drop all tracks. Ids continue upward, so a new track can never be confused with an old one."""
        if self._tracks:
            log.info(f"ObjectTracker: reset ({reason}), dropping {len(self._tracks)} track(s)")
        self._tracks = []

    def note_class_signature(self, signature: Sequence[str]) -> bool:
        """Reset when the class definition changes, returning whether it did.

        Class ids are 1-based positions in the prompt list, so changing the vocabulary renumbers every
        class. A track holding class id 2 would silently come to mean something else -- so identity
        cannot survive a prompt change, and pretending otherwise is worse than a reset.
        """
        sig = tuple(str(s).strip().lower() for s in signature)
        if self._class_signature is None:
            self._class_signature = sig
            return False
        if sig != self._class_signature:
            self._class_signature = sig
            self.reset("class definition changed")
            return True
        return False

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Advance one frame. Returns the confirmed tracks."""
        self._frame += 1
        for t in self._tracks:
            t.age += 1

        pairs, unmatched_d, unmatched_t = match_detections_to_tracks(
            detections, self._tracks,
            iou_threshold=self.iou_threshold,
            max_centroid_distance_m=self.max_centroid_distance_m)

        for di, ti in pairs:
            self._tracks[ti].update(detections[di], self._frame, self.box_smoothing)
        for ti in unmatched_t:
            self._tracks[ti].mark_missed()
        for di in unmatched_d:
            self._tracks.append(Track(self._next_id, detections[di], self._frame))
            self._next_id += 1

        keep = [t for t in self._tracks if t.misses <= self.max_age]
        self.retired += len(self._tracks) - len(keep)
        self._tracks = keep
        return self.confirmed()
