from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import cv2

from .config import CameraConfig, Settings
from .frame_queue import FairFrameQueue
from .images import validate_frame
from .models import FrameCandidate, utc_now
from .motion import MotionDetector

LOG = logging.getLogger(__name__)


class CameraWorker(threading.Thread):
    """Independent lightweight capture loop; AI work is handed to a fair shared queue."""

    def __init__(
        self,
        camera: CameraConfig,
        settings: Settings,
        output: FairFrameQueue,
        stop_event: threading.Event,
        on_status: Callable[[CameraConfig, str, str | None, int], None],
    ) -> None:
        super().__init__(name=f"capture-{camera.camera_id}", daemon=True)
        self.camera = camera
        self.settings = settings
        self.output = output
        self.stop_event = stop_event
        self.on_status = on_status
        self.motion = MotionDetector(settings.motion_min_area)
        self.dropped = 0

    def _enqueue(self, candidate: FrameCandidate) -> None:
        if self.output.put(candidate):
            self.dropped += 1
            self.on_status(self.camera, "online", "Inference queue pressure", 1)
            LOG.warning("Replaced oldest queued frame from %s; queue pressure recorded", self.camera.camera_id)

    def run(self) -> None:
        retry_seconds = 5
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.camera.source, cv2.CAP_V4L2)
            if not cap.isOpened():
                message = f"Cannot open {self.camera.source!r}; retrying in {retry_seconds}s"
                self.on_status(self.camera, "offline", message, 0)
                self.stop_event.wait(retry_seconds)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.on_status(self.camera, "online", None, 0)
            last_trigger = 0.0
            frame_delay = 1.0 / max(1.0, self.settings.monitor_fps)
            try:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Camera returned no frame")
                    frame = validate_frame(frame, normalize_noncontiguous=True)
                    now = time.monotonic()
                    if now - last_trigger >= self.settings.camera_trigger_cooldown_seconds and self.motion.has_activity(frame):
                        last_trigger = now
                        self._enqueue(FrameCandidate(self.camera.camera_id, now, utc_now(), frame.copy()))
                    remaining = frame_delay - (time.monotonic() - started)
                    if remaining > 0:
                        self.stop_event.wait(remaining)
            except Exception as exc:
                LOG.exception("Camera %s failed", self.camera.camera_id)
                self.on_status(self.camera, "offline", str(exc), 0)
                self.stop_event.wait(retry_seconds)
            finally:
                cap.release()
