from __future__ import annotations

import logging

import firebase_admin
from firebase_admin import credentials, firestore

from .config import CameraConfig, Settings
from .models import RecognitionDecision

LOG = logging.getLogger(__name__)


class CloudSummary:
    """Optional Firebase writer for metadata-only dashboard summaries."""

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
            "last_error": error,
        }
        self.db.collection("cameras").document(camera.camera_id).set(
            payload, merge=True, retry=None, timeout=5
        )

    def attendance(self, decision: RecognitionDecision) -> None:
        if self.db is None or decision.status != "accepted" or not decision.user_id:
            return
        self.db.collection("attendance").document(decision.observation_id).set(
            {
                "schema_version": 2,
                "attendance_id": decision.observation_id,
                "observation_id": decision.observation_id,
                "user_id": decision.user_id,
                "name": decision.name,
                "camera_id": decision.camera_id,
                "camera_ids": [decision.camera_id],
                "observed_at": decision.observed_at,
                "timestamp": decision.observed_at,
                "date": decision.observed_at.date().isoformat(),
                "similarity": decision.similarity,
                "confidence": decision.similarity,
                "confidence_scaled": int(round(decision.similarity * 10_000)),
                "margin": decision.margin,
                "detection_score": decision.detection_score,
                "quality": decision.quality,
                "bbox": list(decision.bbox),
                "model_version": decision.model_version,
                "reason": decision.reason,
                "status": "recognized",
                "vote_status": "confirmed",
                "raspberry_pi_id": self.pi_id,
                "created_at": firestore.SERVER_TIMESTAMP,
            },
            retry=None,
            timeout=5,
        )
