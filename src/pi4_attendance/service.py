from __future__ import annotations

import hashlib
import logging
import os
import queue
import signal
import shutil
import threading
import time
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
from .frame_queue import FairFrameQueue
from .models import FrameCandidate, Observation, VoteDecision
from .storage import PiStore
from .voting import VoteCoordinator

LOG = logging.getLogger(__name__)


class AttendanceRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = PiStore(settings.data_root)
        self.cloud = CloudSummary(settings)
        self.engine = FaceEngine(
            settings.model_root, settings.model_pack, settings.face_det_size, settings.match_threshold, settings.match_margin
        )
        self.votes = VoteCoordinator(
            settings.vote_window_seconds,
            settings.vote_required_cameras,
            settings.match_threshold,
            settings.match_margin,
        )
        self.frames = FairFrameQueue([camera.camera_id for camera in settings.cameras], settings.frame_queue_size)
        self.stop_event = threading.Event()
        self.inference_lock = threading.RLock()
        self._workers: list[CameraWorker] = []
        self._session_frames: dict[str, list[tuple[Observation, np.ndarray]]] = {}
        self._api_server = None
        self._api_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._retention_thread: threading.Thread | None = None
        self._started = time.monotonic()

    def start(self) -> None:
        self.engine.start()
        self._workers = [
            CameraWorker(camera, self.settings, self.frames, self.stop_event, self._camera_status)
            for camera in self.settings.cameras
        ]
        self._inference_thread = threading.Thread(target=self._inference_loop, name="pi4-inference", daemon=True)
        self._retention_thread = threading.Thread(target=self._retention_loop, name="pi4-retention", daemon=True)
        self._inference_thread.start()
        self._retention_thread.start()
        for worker in self._workers:
            worker.start()
        if self.settings.api_enabled:
            self._api_thread = threading.Thread(target=self._api_loop, name="pi4-api", daemon=True)
            self._api_thread.start()

    def _camera_status(self, camera: CameraConfig, status: str, error: str | None, dropped: int) -> None:
        self.store.update_camera(camera.camera_id, camera.name, status, error, dropped)
        try:
            self.cloud.camera_status(camera, status, error)
        except Exception:
            LOG.warning("Could not synchronize camera summary", exc_info=True)

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                candidate = self.frames.get(timeout=0.15)
            except queue.Empty:
                self._finalize_votes(time.monotonic())
                continue
            try:
                users = self.store.active_users()
                with self.inference_lock:
                    observations = self.engine.recognize(
                        candidate.frame,
                        users,
                        candidate.camera_id,
                        candidate.captured_monotonic,
                        candidate.captured_at,
                    )
                for observation in observations:
                    session_id = self.votes.add(observation)
                    self.store.record_observation(observation, session_id)
                    self._session_frames.setdefault(session_id, []).append((observation, candidate.frame.copy()))
                self._finalize_votes(time.monotonic())
            except Exception:
                LOG.exception("Inference failed for %s", candidate.camera_id)

    def _finalize_votes(self, now: float, force: bool = False) -> None:
        for decision in self.votes.finalize_due(now, force):
            inserted = self.store.record_decision(decision, self.settings.attendance_cooldown_seconds)
            frames = self._session_frames.pop(decision.session_id, [])
            if decision.status == "confirmed":
                LOG.info(
                    "Vote confirmed %s from %s (%.4f)%s",
                    decision.name,
                    ",".join(decision.camera_ids),
                    decision.confidence,
                    "" if inserted else " [cooldown suppressed]",
                )
                if inserted:
                    try:
                        self.cloud.attendance(decision)
                    except Exception:
                        LOG.warning("Could not synchronize attendance summary", exc_info=True)
            else:
                relative_path = self._save_unknown(decision, frames) if self.settings.save_unknown_images else None
                self.store.record_unknown(uuid4().hex, decision, relative_path)
                LOG.info("Vote rejected: %s", decision.reason)

    def _save_unknown(self, decision: VoteDecision, frames: list[tuple[Observation, np.ndarray]]) -> str | None:
        if not frames:
            return None
        observation, frame = max(frames, key=lambda item: item[0].quality)
        day = decision.decided_at.strftime("%Y/%m/%d")
        directory = self.store.unknown_root / day
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{decision.session_id}-{observation.camera_id}.jpg"
        temporary = self.store.tmp_root / f"{uuid4().hex}.jpg"
        if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
            return None
        os.replace(temporary, path)
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
            create_app(self), host=self.settings.api_host, port=self.settings.api_port, log_level="info"
        )
        self._api_server = uvicorn.Server(config)
        self._api_server.run()

    def health(self) -> dict:
        return {
            "status": "ok",
            "pi_id": self.settings.pi_id,
            "camera_count": len(self.settings.cameras),
            "vote_required_cameras": self.settings.vote_required_cameras,
            "queue_depth": self.frames.qsize(),
            "model_version": self.engine.model_version,
            "uptime_seconds": round(time.monotonic() - self._started, 1),
        }

    def users(self) -> list[dict]:
        return self.store.user_rows()

    def cameras(self) -> list[dict]:
        return self.store.camera_rows()

    def attendance(self, limit: int) -> list[dict]:
        return self.store.recent_attendance(limit)

    def enroll(
        self, name: str, student_id: str, images: list[np.ndarray], raw_images: list[bytes], actor: str
    ) -> dict:
        with self.inference_lock:
            embedding_q, scale = self.engine.enrollment_embedding(images)
        user_id = uuid4().hex
        self.store.create_user(user_id, name, student_id, embedding_q, scale, self.engine.model_version)
        user_directory = self.store.enrollment_root / user_id / "original"
        user_directory.mkdir(parents=True, exist_ok=False)
        try:
            for index, (image, raw) in enumerate(zip(images, raw_images, strict=True), start=1):
                image_id = uuid4().hex
                path = user_directory / f"{index:02d}-{image_id}.jpg"
                temporary = self.store.tmp_root / f"{image_id}.jpg"
                if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError("Could not save enrollment image")
                os.replace(temporary, path)
                self.store.add_enrollment_image(
                    image_id,
                    user_id,
                    path.relative_to(self.settings.data_root).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    image.shape[1],
                    image.shape[0],
                )
            self.store.audit(actor, "enroll_user", user_id, student_id=student_id, image_count=len(images))
        except Exception:
            shutil.rmtree(self.store.enrollment_root / user_id, ignore_errors=True)
            self.store.delete_incomplete_user(user_id)
            raise
        return {"user_id": user_id, "name": name.strip(), "student_id": student_id.strip(), "status": "active"}

    def set_user_status(self, user_id: str, user_status: str, actor: str) -> bool:
        updated = self.store.set_user_status(user_id, user_status)
        if updated:
            self.store.audit(actor, f"set_user_{user_status}", user_id)
        return updated

    def stop(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True
        self.stop_event.set()

    def run(self) -> None:
        self.start()
        while not self.stop_event.wait(1):
            pass
        for worker in self._workers:
            worker.join(timeout=3)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5)
        if self._retention_thread is not None:
            self._retention_thread.join(timeout=2)
        if self._api_thread is not None:
            self._api_thread.join(timeout=5)
        self._finalize_votes(time.monotonic(), force=True)
        self.store.close()


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    runtime = AttendanceRuntime(settings)
    signal.signal(signal.SIGTERM, lambda *_: runtime.stop())
    signal.signal(signal.SIGINT, lambda *_: runtime.stop())
    runtime.run()


if __name__ == "__main__":
    main()
