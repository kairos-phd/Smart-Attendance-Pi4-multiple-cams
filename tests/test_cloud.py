from datetime import datetime, timezone

from pi4_attendance.cloud import CloudSummary
from pi4_attendance.models import RecognitionDecision
from types import SimpleNamespace


class FakeDocument:
    def __init__(self) -> None:
        self.payload = None
        self.options = None

    def set(self, payload, **options) -> None:
        self.payload = payload
        self.options = options


class FakeCollection:
    def __init__(self, document: FakeDocument) -> None:
        self._document = document
        self.document_id = None

    def document(self, document_id: str) -> FakeDocument:
        self.document_id = document_id
        return self._document


class FakeDatabase:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def collection(self, name: str) -> FakeCollection:
        assert name == "attendance"
        return self._collection


def test_attendance_summary_keeps_aliases_and_full_recognition_metadata() -> None:
    document = FakeDocument()
    collection = FakeCollection(document)
    cloud = object.__new__(CloudSummary)
    cloud.db = FakeDatabase(collection)
    cloud.pi_id = "pi-1"
    observed_at = datetime(2026, 8, 27, 8, 30, tzinfo=timezone.utc)
    decision = RecognitionDecision(
        "obs-1", "accepted", "u1", "Test User", "camera-1", observed_at,
        0.81, 0.12, 0.93, 0.74, (1, 2, 30, 40), "model-v1", "thresholds_passed",
    )

    cloud.attendance(decision)

    payload = document.payload
    assert collection.document_id == "obs-1"
    assert payload["schema_version"] == 2
    assert payload["observation_id"] == "obs-1"
    assert payload["observed_at"] == observed_at
    assert payload["timestamp"] == observed_at
    assert payload["camera_id"] == "camera-1"
    assert payload["camera_ids"] == ["camera-1"]
    assert payload["similarity"] == payload["confidence"] == 0.81
    assert payload["margin"] == 0.12
    assert payload["detection_score"] == 0.93
    assert payload["quality"] == 0.74
    assert payload["bbox"] == [1, 2, 30, 40]
    assert payload["model_version"] == "model-v1"
    assert payload["reason"] == "thresholds_passed"
    assert payload["vote_status"] == "confirmed"
    assert document.options == {"retry": None, "timeout": 5}


def test_camera_recovery_clears_stale_cloud_error() -> None:
    document = FakeDocument()
    collection = FakeCollection(document)
    cloud = object.__new__(CloudSummary)
    cloud.db = SimpleNamespace(collection=lambda name: collection)
    cloud.pi_id = "pi-1"
    camera = SimpleNamespace(camera_id="camera-1", name="Front")

    cloud.camera_status(camera, "offline", "camera unavailable")
    assert document.payload["last_error"] == "camera unavailable"

    cloud.camera_status(camera, "online", None)
    assert "last_error" in document.payload
    assert document.payload["last_error"] is None
