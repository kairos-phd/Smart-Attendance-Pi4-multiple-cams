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
    def __init__(
        self,
        camera: CameraConfig,
        settings: Settings,
        output: FairFrameQueue,
        stop_event: threading.Event,
        on_status: Callable[[CameraConfig, str, str | None, int], None],
        recognition_enabled: threading.Event | None = None,
        on_preview_frame: Callable[[str, object], None] | None = None,
        preview_selected: Callable[[str], bool] | None = None,
        preview_fps: float | None = None,
        metrics=None,
    ) -> None:
        super().__init__(name=f"capture-{camera.camera_id}", daemon=True)
        self.camera = camera
        self.settings = settings
        self.output = output
        self.stop_event = stop_event
        self.on_status = on_status
        self.recognition_enabled = recognition_enabled or threading.Event()
        if recognition_enabled is None:
            self.recognition_enabled.set()
        self.on_preview_frame = on_preview_frame
        self.preview_selected = preview_selected
        self.preview_fps = preview_fps
        self.metrics = metrics
        self.motion = MotionDetector(settings.motion_min_area)
        self.dropped = 0
        self._last_monitor = float("-inf")
        self._last_trigger = float("-inf")

    def _enqueue(self, candidate: FrameCandidate) -> None:
        if self.metrics is not None:
            self.metrics.increment(f"camera.{self.camera.camera_id}.enqueued")
        replaced = self.output.put(candidate)
        if self.metrics is not None and hasattr(self.output, "qsize"):
            self.metrics.set_gauge("queue.depth", self.output.qsize())
        if replaced:
            self.dropped += 1
            if self.metrics is not None:
                self.metrics.increment(f"camera.{self.camera.camera_id}.queue_replaced")
            self.on_status(self.camera, "online", "Inference queue pressure", 1)
            LOG.warning("Replaced oldest queued frame from %s; queue pressure recorded", self.camera.camera_id)

    def _process_frame(self, frame, now: float) -> None:
        if self.metrics is not None:
            prefix = f"camera.{self.camera.camera_id}"
            self.metrics.increment(f"{prefix}.captured")
            self.metrics.set_gauge(f"{prefix}.width", frame.shape[1])
            self.metrics.set_gauge(f"{prefix}.height", frame.shape[0])
            self.metrics.set_gauge("recognition_enabled", self.recognition_enabled.is_set())
        if self.on_preview_frame is not None and self.preview_selected is not None and self.preview_selected(self.camera.camera_id):
            self.on_preview_frame(self.camera.camera_id, frame)
        prefix = f"camera.{self.camera.camera_id}"
        if not self.recognition_enabled.is_set():
            if self.metrics is not None:
                self.metrics.increment(f"{prefix}.recognition_disabled")
            return
        if now - self._last_monitor < 1.0 / self.settings.monitor_fps:
            if self.metrics is not None:
                self.metrics.increment(f"{prefix}.rate_limited")
            return
        self._last_monitor = now
        if now - self._last_trigger < self.settings.camera_trigger_cooldown_seconds:
            if self.metrics is not None:
                self.metrics.increment(f"{prefix}.trigger_cooldown")
            return
        if not self.motion.has_activity(frame):
            if self.metrics is not None:
                self.metrics.increment(f"{prefix}.no_motion")
            return
        self._last_trigger = now
        self._enqueue(FrameCandidate(self.camera.camera_id, now, utc_now(), frame.copy()))

    def _target_fps(self) -> float:
        if self.preview_fps is not None and self.preview_selected is not None and self.preview_selected(self.camera.camera_id):
            return self.preview_fps
        return self.settings.monitor_fps

    def _configure_capture(self, cap) -> None:
        if self.preview_fps is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self) -> None:
        retry_seconds = 5
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.camera.source, cv2.CAP_V4L2)
            if not cap.isOpened():
                message = f"Cannot open {self.camera.source!r}; retrying in {retry_seconds}s"
                self.on_status(self.camera, "offline", message, 0)
                self.stop_event.wait(retry_seconds)
                continue
            self._configure_capture(cap)
            self.on_status(self.camera, "online", None, 0)
            requested_fps = 0.0
            try:
                while not self.stop_event.is_set():
                    target_fps = self._target_fps()
                    if target_fps != requested_fps:
                        cap.set(cv2.CAP_PROP_FPS, target_fps)
                        LOG.info("Camera %s requested %.1f FPS", self.camera.camera_id, target_fps)
                        requested_fps = target_fps
                    started = time.monotonic()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Camera returned no frame")
                    frame = validate_frame(frame, normalize_noncontiguous=True)
                    now = time.monotonic()
                    self._process_frame(frame, now)
                    remaining = 1.0 / max(1.0, target_fps) - (time.monotonic() - started)
                    if remaining > 0:
                        self.stop_event.wait(remaining)
            except Exception as exc:
                LOG.exception("Camera %s failed", self.camera.camera_id)
                self.on_status(self.camera, "offline", str(exc), 0)
                self.stop_event.wait(retry_seconds)
            finally:
                cap.release()
