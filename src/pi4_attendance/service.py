from __future__ import annotations

import argparse
import hashlib
import logging
import os
import queue
import signal
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .api import create_app
from .camera import CameraWorker
from .cloud import CloudSummary
from .config import CameraConfig, Settings
from .face_engine import FaceEngine
from .embedding_catalog import EmbeddingCatalog
from .frame_queue import FairFrameQueue
from .images import crop_face
from .models import FrameCandidate, Observation, RecognitionDecision, utc_now
from .preview import PreviewWindow
from .recognition import RecognitionPolicy
from .storage import PiStore
from .telemetry import JsonlTelemetryWriter, ProcSampler, RollingMetrics, TelemetryMonitor

LOG = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    _samples: int = 0
    _queue_age_total_ms: float = 0.0
    _inference_total_ms: float = 0.0
    _queue_age_latest_ms: float = 0.0
    _inference_latest_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, queue_age_ms: float, inference_ms: float) -> None:
        with self._lock:
            self._samples += 1
            self._queue_age_latest_ms = max(0.0, float(queue_age_ms))
            self._inference_latest_ms = max(0.0, float(inference_ms))
            self._queue_age_total_ms += self._queue_age_latest_ms
            self._inference_total_ms += self._inference_latest_ms

    def snapshot(self) -> dict:
        with self._lock:
            samples = self._samples
            queue_average = self._queue_age_total_ms / samples if samples else 0.0
            inference_average = self._inference_total_ms / samples if samples else 0.0
            return {
                "queue_age_ms": {
                    "latest": round(self._queue_age_latest_ms, 3),
                    "aggregate": {"samples": samples, "average": round(queue_average, 3)},
                },
                "inference_ms": {
                    "latest": round(self._inference_latest_ms, 3),
                    "aggregate": {"samples": samples, "average": round(inference_average, 3)},
                },
            }


class AttendanceRuntime:
    def __init__(self, settings: Settings, preview: bool = False) -> None:
        self.settings = settings
        self.store = PiStore(settings.data_root)
        self.cloud = CloudSummary(settings)
        self.engine = FaceEngine(
            settings.model_root, settings.model_pack, settings.face_det_size, settings.match_threshold, settings.match_margin
        )
        self.recognition_policy = RecognitionPolicy(
            settings.match_threshold,
            settings.match_margin,
            settings.min_detection_score,
            settings.min_face_quality,
        )
        self.frames = FairFrameQueue([camera.camera_id for camera in settings.cameras], settings.frame_queue_size)
        self.stop_event = threading.Event()
        self._show_preview = preview
        self.preview = PreviewWindow(
            tuple(camera.camera_id for camera in settings.cameras),
            settings.preview_fps,
            self.enroll,
        )
        self.recognition_enabled = self.preview.recognition_enabled
        if not preview:
            self.recognition_enabled.set()
        self.inference_lock = threading.RLock()
        self._enrollment_lock = threading.Lock()
        self._catalog_lock = threading.Lock()
        self._workers: list[CameraWorker] = []
        self._api_server = None
        self._api_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._retention_thread: threading.Thread | None = None
        self._cloud_thread: threading.Thread | None = None
        self._cloud_wake = threading.Event()
        self._camera_cloud_lock = threading.Lock()
        self._pending_camera_status: dict[str, tuple[CameraConfig, str, str | None]] = {}
        self._online_camera_ids: set[str] = set()
        self._cloud_poll_seconds = 1.0
        self._started = time.monotonic()
        self._runtime_state = "starting"
        self.metrics = RuntimeMetrics()
        self.performance_metrics = RollingMetrics(window_seconds=10.0)
        self.performance_log_path: Path | None = None
        self._telemetry_writer: JsonlTelemetryWriter | None = None
        self._telemetry_monitor: TelemetryMonitor | None = None
        self._telemetry_run_id = ""
        self._telemetry_run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        self.performance_log_path = settings.data_root / "performance" / f"{self._telemetry_run_id}.jsonl"
        self._telemetry_writer = JsonlTelemetryWriter(self.performance_log_path)
        self._telemetry_monitor = TelemetryMonitor(
            ProcSampler(),
            self.performance_metrics,
            self._telemetry_writer,
            run_id=self._telemetry_run_id,
        )
        self._embedding_catalog = EmbeddingCatalog.from_users([], self.engine.model_version)

    def _refresh_embedding_catalog(self) -> None:
        with self._catalog_lock:
            catalog = EmbeddingCatalog.from_users(self.store.active_users(), self.engine.model_version)
            self._embedding_catalog = catalog

    def start(self) -> None:
        self.engine.start()
        self._refresh_embedding_catalog()
        self._workers = [
            CameraWorker(
                camera,
                self.settings,
                self.frames,
                self.stop_event,
                self._camera_status,
                recognition_enabled=self.recognition_enabled,
                on_preview_frame=self.preview.submit,
                preview_selected=self.preview.is_selected,
                preview_fps=self.settings.preview_fps if self._show_preview else None,
                metrics=self.performance_metrics,
            )
            for camera in self.settings.cameras
        ]
        self._inference_thread = threading.Thread(target=self._inference_loop, name="pi4-inference", daemon=True)
        self._retention_thread = threading.Thread(target=self._retention_loop, name="pi4-retention", daemon=True)
        if self.cloud.enabled:
            self._cloud_thread = threading.Thread(target=self._cloud_loop, name="pi4-cloud", daemon=True)
            self._cloud_thread.start()
        self._inference_thread.start()
        self._retention_thread.start()
        for worker in self._workers:
            worker.start()
        if self.settings.api_enabled:
            self._api_thread = threading.Thread(target=self._api_loop, name="pi4-api", daemon=True)
            self._api_thread.start()
        self._start_telemetry()
        self._runtime_state = "running"

    def _camera_status(self, camera: CameraConfig, status: str, error: str | None, dropped: int) -> None:
        self.store.update_camera(camera.camera_id, camera.name, status, error, dropped)
        with self._camera_cloud_lock:
            online_camera_ids = getattr(self, "_online_camera_ids", None)
            if online_camera_ids is None:
                online_camera_ids = self._online_camera_ids = set()
            if status == "online":
                online_camera_ids.add(camera.camera_id)
            else:
                online_camera_ids.discard(camera.camera_id)
            performance = getattr(self, "performance_metrics", None)
            if performance is not None:
                performance.set_gauge("active_cameras", len(online_camera_ids))
                performance.set_gauge(f"camera.{camera.camera_id}.online", status == "online")
                if status != "online":
                    performance.increment(f"camera.{camera.camera_id}.failures")
        if self.cloud.enabled:
            with self._camera_cloud_lock:
                self._pending_camera_status[camera.camera_id] = (camera, status, error)
            self._cloud_wake.set()

    def _start_telemetry(self) -> None:
        if self._telemetry_writer is None or self._telemetry_monitor is None:
            return
        self._telemetry_writer.submit(
            {
                "type": "metadata",
                "run_id": self._telemetry_run_id,
                "started_at": utc_now().isoformat(),
                "pi_id": self.settings.pi_id,
                "model_pack": self.settings.model_pack,
                "model_version": self.engine.model_version,
                "camera_count": len(self.settings.cameras),
                "cameras": [
                    {
                        "camera_id": camera.camera_id,
                        "name": camera.name,
                        "source": str(camera.source),
                    }
                    for camera in self.settings.cameras
                ],
                "requested_resolution": [self.settings.frame_width, self.settings.frame_height],
                "monitor_fps": self.settings.monitor_fps,
                "preview_fps": self.settings.preview_fps,
                "face_det_size": self.settings.face_det_size,
                "camera_trigger_cooldown_seconds": self.settings.camera_trigger_cooldown_seconds,
            }
        )
        self._telemetry_writer.start()
        self._telemetry_monitor.start()
        LOG.info("Performance log: %s", self.performance_log_path)

    def _stop_telemetry(self) -> None:
        monitor = getattr(self, "_telemetry_monitor", None)
        writer = getattr(self, "_telemetry_writer", None)
        if monitor is not None:
            monitor.stop()
        if writer is not None:
            writer.submit(
                {
                    "type": "final",
                    "run_id": getattr(self, "_telemetry_run_id", ""),
                    "finished_at": utc_now().isoformat(),
                    "runtime": self.performance_metrics.snapshot(),
                }
            )
            writer.stop()

    def _cloud_loop(self) -> None:
        while not self.stop_event.is_set():
            self._cloud_wake.clear()
            for decision in self.store.pending_attendance():
                if self.stop_event.is_set():
                    break
                try:
                    self.cloud.attendance(decision)
                except Exception:
                    LOG.warning("Could not synchronize attendance summary", exc_info=True)
                else:
                    self.store.mark_attendance_synced(decision.observation_id)
            if self.stop_event.is_set():
                break
            with self._camera_cloud_lock:
                statuses = list(self._pending_camera_status.values())
                self._pending_camera_status.clear()
            for camera, status, error in statuses:
                if self.stop_event.is_set():
                    break
                try:
                    self.cloud.camera_status(camera, status, error)
                except Exception:
                    LOG.warning("Could not synchronize camera summary", exc_info=True)
                    with self._camera_cloud_lock:
                        self._pending_camera_status.setdefault(camera.camera_id, (camera, status, error))
            self._cloud_wake.wait(self._cloud_poll_seconds)

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                candidate = self.frames.get(timeout=0.15)
            except queue.Empty:
                continue
            if not self.recognition_enabled.is_set():
                performance = getattr(self, "performance_metrics", None)
                if performance is not None:
                    performance.increment("inference.recognition_disabled")
                continue
            try:
                processing_started = time.monotonic()
                with self.inference_lock:
                    inference_started = time.monotonic()
                    observations = self.engine.recognize(
                        candidate.frame,
                        self._embedding_catalog,
                        candidate.camera_id,
                        candidate.captured_monotonic,
                        candidate.captured_at,
                    )
                inference_finished = time.monotonic()
                queue_age_ms = (processing_started - candidate.captured_monotonic) * 1000.0
                lock_wait_ms = (inference_started - processing_started) * 1000.0
                inference_ms = (inference_finished - inference_started) * 1000.0
                self.metrics.record(queue_age_ms, inference_ms)
                performance = getattr(self, "performance_metrics", None)
                if performance is not None:
                    performance.increment("inference.processed")
                    performance.observe("queue_age_ms", queue_age_ms)
                    performance.observe("inference_lock_wait_ms", lock_wait_ms)
                    performance.observe("inference_ms", inference_ms)
                    performance.set_gauge("queue.depth", self.frames.qsize())
                for observation in observations:
                    self.store.record_observation(observation)
                    self._handle_decision(
                        self.recognition_policy.decide(observation), observation, candidate.frame
                    )
                if performance is not None:
                    performance.observe(
                        "full_processing_ms",
                        (time.monotonic() - candidate.captured_monotonic) * 1000.0,
                    )
            except Exception:
                LOG.exception("Inference failed for %s", candidate.camera_id)

    def _handle_decision(
        self, decision: RecognitionDecision, observation: Observation, frame: np.ndarray
    ) -> None:
        inserted, attendance_outcome = self.store.record_decision_result(
            decision, self.settings.attendance_cooldown_seconds
        )
        performance = getattr(self, "performance_metrics", None)
        if performance is not None:
            performance.increment(f"recognition.{decision.status}")
        if decision.status == "accepted":
            if performance is not None:
                performance.increment(f"attendance.{attendance_outcome}")
            LOG.info(
                "Recognition accepted %s from %s (%.4f)%s",
                decision.name,
                decision.camera_id,
                decision.similarity,
                "" if inserted else " [cooldown suppressed]",
            )
            preview = getattr(self, "preview", None)
            if preview is not None and attendance_outcome in {"recorded", "cooldown"}:
                matched_user = self._embedding_catalog.user(decision.user_id)
                if matched_user is not None:
                    preview.show_recognition(
                        decision.camera_id,
                        name=decision.name or "Unknown",
                        student_id=matched_user.student_id,
                        similarity=decision.similarity,
                        timestamp=decision.observed_at.astimezone().strftime("%H:%M:%S"),
                        status="ATTENDANCE RECORDED" if inserted else "COOLDOWN",
                    )
            if inserted:
                self._cloud_wake.set()
        else:
            frames = [(observation, frame.copy())]
            relative_path = self._save_unknown(decision, frames) if self.settings.save_unknown_images else None
            self.store.record_unknown(uuid4().hex, decision, relative_path)
            LOG.info("Recognition rejected: %s", decision.reason)

    def _save_unknown(self, decision: RecognitionDecision, frames: list[tuple[Observation, np.ndarray]]) -> str | None:
        if not frames:
            return None
        matching = [item for item in frames if item[0].observation_id == decision.observation_id]
        if not matching:
            return None
        observation, frame = matching[0]
        try:
            crop = crop_face(frame, observation.bbox)
        except ValueError:
            return None
        if crop is None:
            return None
        day = decision.decided_at.strftime("%Y/%m/%d")
        directory = self.store.unknown_root / day
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{decision.observation_id}-{observation.camera_id}.jpg"
        temporary = self.store.tmp_root / f"{uuid4().hex}.jpg"
        if not cv2.imwrite(str(temporary), crop, [cv2.IMWRITE_JPEG_QUALITY, 86]):
            return None
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
        return path.relative_to(self.settings.data_root).as_posix()

    def _retention_loop(self) -> None:
        while not self.stop_event.wait(3600):
            try:
                removed = self.store.prune_unknown(self.settings.unknown_retention_days)
                if removed:
                    LOG.info("Retention removed %d expired unknown images", removed)
            except Exception:
                LOG.exception("Unknown-image retention failed")

    def _api_loop(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            create_app(self),
            host=self.settings.api_host,
            port=self.settings.api_port,
            log_level="info",
            proxy_headers=False,
        )
        self._api_server = uvicorn.Server(config)
        self._api_server.run()

    def health(self) -> dict:
        camera_rows = self.store.camera_rows()
        configured_ids = {camera.camera_id for camera in self.settings.cameras}
        current_rows = [row for row in camera_rows if row.get("camera_id") in configured_ids]
        inference_alive = self._inference_thread is not None and self._inference_thread.is_alive()
        if self._runtime_state == "stopped":
            status = "stopped"
        elif self._runtime_state == "starting" or len(current_rows) < len(configured_ids):
            status = "starting"
        elif not inference_alive or any(row.get("status") != "online" for row in current_rows):
            status = "degraded"
        else:
            status = "ok"
        telemetry_writer = getattr(self, "_telemetry_writer", None)
        telemetry_monitor = getattr(self, "_telemetry_monitor", None)
        return {
            "status": status,
            "pi_id": self.settings.pi_id,
            "camera_count": len(self.settings.cameras),
            "queue_depth": self.frames.qsize(),
            "model_version": self.engine.model_version,
            "inference_alive": inference_alive,
            "camera_status": current_rows,
            "dropped_frames": sum(int(row.get("dropped_frames", 0)) for row in current_rows),
            "runtime_metrics": self.metrics.snapshot(),
            "performance_metrics": self.performance_metrics.snapshot(),
            "system": None if telemetry_monitor is None else telemetry_monitor.latest_sample,
            "performance_log": None if self.performance_log_path is None else str(self.performance_log_path),
            "telemetry": {
                "monitor_alive": telemetry_monitor is not None and telemetry_monitor.is_alive(),
                "writer_alive": telemetry_writer is not None and telemetry_writer.is_alive(),
                "writer_error": None if telemetry_writer is None else telemetry_writer.error,
                "samples_dropped": 0 if telemetry_writer is None else telemetry_writer.dropped,
            },
            "uptime_seconds": round(time.monotonic() - self._started, 1),
        }

    def users(self) -> list[dict]:
        return self.store.user_rows()

    def cameras(self) -> list[dict]:
        return self.store.camera_rows()

    def attendance(self, limit: int) -> list[dict]:
        return self.store.recent_attendance(limit)

    def recognitions(self, limit: int) -> list[dict]:
        return self.store.recent_recognitions(limit)

    def control_status(self) -> dict:
        result = self.preview.status()
        camera_lock = getattr(self, "_camera_cloud_lock", None)
        if camera_lock is None:
            online_camera_ids = []
        else:
            with camera_lock:
                online_camera_ids = sorted(self._online_camera_ids)
        result.update(
            runtime_state=getattr(self, "_runtime_state", "starting"),
            online_camera_ids=online_camera_ids,
        )
        return result

    def select_camera(self, camera_id: str) -> dict:
        self.preview.select_camera(camera_id)
        return self.control_status()

    def set_recognition(self, enabled: bool) -> dict:
        self.preview.set_recognition(enabled)
        return self.control_status()

    def begin_enrollment(self, name: str, student_id: str, actor: str) -> dict:
        if not self.preview.begin_enrollment(name, student_id, actor):
            status = self.preview.status()
            if status["enrollment"] is not None or status["enrollment_busy"]:
                raise FileExistsError("Enrollment already in progress")
            raise ValueError(self.preview.enrollment_status or "Enrollment already in progress")
        return self.control_status()

    def capture_enrollment_frame(self) -> dict:
        if not self.preview.capture_enrollment_frame():
            raise ValueError(self.preview.enrollment_status or "Cannot capture enrollment frame")
        return self.control_status()

    def submit_enrollment(self) -> dict:
        if not self.preview.submit_enrollment():
            raise ValueError(self.preview.enrollment_status or "Cannot submit enrollment")
        return self.control_status()

    def cancel_enrollment(self) -> dict:
        if not self.preview.cancel_enrollment():
            raise ValueError(self.preview.enrollment_status or "Cannot cancel enrollment")
        return self.control_status()

    def enroll(self, name: str, student_id: str, images: list[np.ndarray], actor: str) -> dict:
        if not self._enrollment_lock.acquire(blocking=False):
            raise ValueError("Enrollment already in progress; retry later")
        try:
            if self.stop_event.is_set():
                raise ValueError("Runtime is stopping")
            return self._enroll(name, student_id, images, actor)
        finally:
            self._enrollment_lock.release()

    def _enroll(self, name: str, student_id: str, images: list[np.ndarray], actor: str) -> dict:
        embedding_q, scale = self.engine.enrollment_embedding(images)
        if self.stop_event.is_set():
            raise ValueError("Runtime is stopping")
        user_id = uuid4().hex
        user_directory = self.store.enrollment_root / user_id / "original"
        image_records = []
        temporary = None
        try:
            user_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            for index, image in enumerate(images, start=1):
                image_id = uuid4().hex
                path = user_directory / f"{index:02d}-{image_id}.jpg"
                temporary = self.store.tmp_root / f"{image_id}.jpg"
                if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError("Could not save enrollment image")
                temporary.chmod(0o600)
                os.replace(temporary, path)
                temporary = None
                path.chmod(0o600)
                image_records.append(
                    (
                        image_id,
                        path.relative_to(self.settings.data_root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        image.shape[1],
                        image.shape[0],
                    )
                )
            with self._catalog_lock:
                if self.stop_event.is_set():
                    raise ValueError("Runtime is stopping")
                catalog = self.store.enroll_user(
                    user_id,
                    name,
                    student_id,
                    embedding_q,
                    scale,
                    self.engine.model_version,
                    image_records,
                    actor,
                    lambda users: EmbeddingCatalog.from_users(users, self.engine.model_version),
                )
                self._embedding_catalog = catalog
        except Exception:
            shutil.rmtree(self.store.enrollment_root / user_id, ignore_errors=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"user_id": user_id, "name": name.strip(), "student_id": student_id.strip(), "status": "active"}

    def set_user_status(self, user_id: str, user_status: str, actor: str) -> bool:
        updated = self.store.set_user_status_audited(user_id, user_status, actor)
        if updated:
            self._refresh_embedding_catalog()
        return updated

    def stop(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True
        self.stop_event.set()
        self._cloud_wake.set()
        self._runtime_state = "stopped"

    def run(self) -> None:
        self.start()
        try:
            if not getattr(self, "_show_preview", hasattr(self, "preview")):
                while not self.stop_event.wait(1):
                    pass
            else:
                self.preview.run(self.stop_event)
        finally:
            self.stop()
            for worker in self._workers:
                worker.join(timeout=3)
            if self._inference_thread is not None:
                self._inference_thread.join(timeout=5)
            if self._retention_thread is not None:
                self._retention_thread.join(timeout=2)
            if self._cloud_thread is not None:
                self._cloud_thread.join(timeout=5)
            if self._api_thread is not None:
                self._api_thread.join(timeout=5)
            self._stop_telemetry()
            blocked_workers = [worker.name for worker in self._workers if worker.is_alive()]
            if blocked_workers:
                LOG.error("Camera workers did not stop; leaving SQLite open until process exit: %s", blocked_workers)
            else:
                with self._enrollment_lock:
                    self.store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SIC Raspberry Pi attendance runtime")
    parser.add_argument("--preview", action="store_true", help="show HDMI preview with recognition and enrollment controls")
    args = parser.parse_args()
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    runtime = AttendanceRuntime(settings, preview=args.preview)
    signal.signal(signal.SIGTERM, lambda *_: runtime.stop())
    signal.signal(signal.SIGINT, lambda *_: runtime.stop())
    runtime.run()


if __name__ == "__main__":
    main()
