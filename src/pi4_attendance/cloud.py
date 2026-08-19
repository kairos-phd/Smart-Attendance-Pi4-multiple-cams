from __future__ import annotations

import logging

import firebase_admin
from firebase_admin import credentials, firestore

from .config import CameraConfig, Settings
from .models import VoteDecision

LOG = logging.getLogger(__name__)


class CloudSummary:
    """Optional Firebase writer for non-biometric dashboard summaries only."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.cloud_summary_enabled
        self.pi_id = settings.pi_id
        self.db = None
        if not self.enabled:
            return
        assert settings.firebase_service_account is not None
        if not firebase_admin._apps:
            options = {"projectId": settings.firebase_project_id} if settings.firebase_project_id else None
            firebase_admin.initialize_app(credentials.Certificate(str(settings.firebase_service_account)), options)
        self.db = firestore.client()

    def camera_status(self, camera: CameraConfig, status: str, error: str | None) -> None:
        if self.db is None:
            return
        payload = {
            "camera_id": camera.camera_id,
            "name": camera.name,
            "raspberry_pi_id": self.pi_id,
            "status": status,
            "last_seen": firestore.SERVER_TIMESTAMP,
        }
        if error:
            payload["last_error"] = error
        self.db.collection("cameras").document(camera.camera_id).set(payload, merge=True)

    def attendance(self, decision: VoteDecision) -> None:
        if self.db is None or decision.status != "confirmed" or not decision.user_id:
            return
        self.db.collection("attendance").document(decision.session_id).set(
            {
                "attendance_id": decision.session_id,
                "user_id": decision.user_id,
                "name": decision.name,
                "camera_id": "+".join(decision.camera_ids),
                "camera_ids": list(decision.camera_ids),
                "timestamp": decision.decided_at,
                "date": decision.decided_at.date().isoformat(),
                "confidence_scaled": int(round(decision.confidence * 10_000)),
                "status": "recognized",
                "vote_status": "confirmed",
                "raspberry_pi_id": self.pi_id,
                "created_at": firestore.SERVER_TIMESTAMP,
            }
        )
