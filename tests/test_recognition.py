from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pi4_attendance.face_engine import FaceEngine
from pi4_attendance.models import Observation, RegisteredUser
from pi4_attendance.recognition import RecognitionPolicy


def observation(**overrides) -> Observation:
    values = {
        "camera_id": "camera-1",
        "observed_monotonic": 10.0,
        "observed_at": datetime(2026, 8, 27, tzinfo=timezone.utc),
        "candidate_user_id": "user-1",
        "candidate_name": "Test User",
        "similarity": 0.70,
        "margin": 0.10,
        "detection_score": 0.90,
        "bbox": (1, 2, 30, 40),
        "quality": 0.80,
        "model_version": "model-v1",
        "observation_id": "obs-1",
    }
    values.update(overrides)
    return Observation(**values)


def test_one_qualified_observation_is_accepted_with_complete_metadata() -> None:
    decision = RecognitionPolicy(0.45, 0.06, 0.60, 0.35).decide(observation())

    assert decision.status == "accepted"
    assert decision.user_id == "user-1"
    assert decision.observation_id == "obs-1"
    assert decision.camera_id == "camera-1"
    assert decision.similarity == 0.70
    assert decision.margin == 0.10
    assert decision.detection_score == 0.90
    assert decision.quality == 0.80
    assert decision.bbox == (1, 2, 30, 40)
    assert decision.model_version == "model-v1"


def test_each_failed_threshold_has_a_stable_rejection_reason() -> None:
    policy = RecognitionPolicy(0.45, 0.06, 0.60, 0.35)

    assert policy.decide(observation(candidate_user_id=None, candidate_name=None)).reason == "no_candidate"
    assert policy.decide(observation(similarity=0.44)).reason == "similarity_below_threshold"
    assert policy.decide(observation(margin=0.05)).reason == "margin_below_threshold"
    assert policy.decide(observation(detection_score=0.59)).reason == "detection_score_below_threshold"
    assert policy.decide(observation(quality=0.34)).reason == "quality_below_threshold"


@pytest.mark.parametrize("field", ("similarity", "margin", "detection_score", "quality"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_recognition_metrics_are_rejected(field: str, value: float) -> None:
    decision = RecognitionPolicy(0.45, 0.06, 0.60, 0.35).decide(observation(**{field: value}))

    assert decision.status == "rejected"
    assert decision.user_id is None
    assert decision.name is None
    assert decision.reason == "non_finite_metric"


def test_finite_threshold_boundaries_are_inclusive() -> None:
    decision = RecognitionPolicy(0.45, 0.06, 0.60, 0.35).decide(
        observation(similarity=0.45, margin=0.06, detection_score=0.60, quality=0.35)
    )

    assert decision.status == "accepted"


def test_camera_id_is_provenance_not_a_quorum_requirement() -> None:
    policy = RecognitionPolicy(0.45, 0.06, 0.60, 0.35)

    first = policy.decide(observation(camera_id="only-camera", observation_id="obs-1"))
    second = policy.decide(observation(camera_id="only-camera", observation_id="obs-2"))

    assert first.status == "accepted"
    assert second.status == "accepted"


def test_face_engine_preserves_best_candidate_for_policy_rejection_reason() -> None:
    engine = FaceEngine(Path("models"), "model-v1", 320, 0.45, 0.06)
    face = SimpleNamespace(
        bbox=np.array([0, 0, 20, 20]), det_score=0.9,
        normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    engine._app = SimpleNamespace(get=lambda _image: [face])
    user = RegisteredUser("u1", "Test User", "S001", np.array([0, 127], dtype=np.int8), 1 / 127, "model-v1")

    [result] = engine.recognize(
        np.zeros((32, 32, 3), dtype=np.uint8), [user], "camera-1", 1.0,
        datetime(2026, 8, 27, tzinfo=timezone.utc),
    )

    assert result.candidate_user_id == "u1"
    assert RecognitionPolicy(0.45, 0.06, 0.60, 0.35).decide(result).reason == "similarity_below_threshold"


def test_face_engine_rejects_invalid_frame_before_detection() -> None:
    engine = FaceEngine(Path("models"), "model-v1", 320, 0.45, 0.06)
    called = False

    def detect(_image):
        nonlocal called
        called = True
        return []

    engine._app = SimpleNamespace(get=detect)

    try:
        engine.recognize(
            np.zeros((8, 8, 3), dtype=np.uint8), [], "camera-1", 1.0,
            datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("tiny frame reached detection")
    assert not called


def test_enrollment_rejects_noncontiguous_frame_before_detection() -> None:
    engine = FaceEngine(Path("models"), "model-v1", 320, 0.45, 0.06)
    called = False

    def detect(_image):
        nonlocal called
        called = True
        return []

    engine._app = SimpleNamespace(get=detect)
    image = np.zeros((32, 64, 3), dtype=np.uint8)[:, ::2]

    try:
        engine.enrollment_embedding([image, image, image])
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous enrollment frame reached detection")
    assert not called


def test_enrollment_rejects_invalid_detector_bbox_clearly() -> None:
    engine = FaceEngine(Path("models"), "model-v1", 320, 0.45, 0.06)
    face = SimpleNamespace(
        bbox=np.array([20, 10, 10, 30]),
        det_score=0.9,
        normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    engine._app = SimpleNamespace(get=lambda _image: [face])
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="bounding box"):
        engine.enrollment_embedding([image, image, image])


def test_recognition_skips_invalid_detector_bbox() -> None:
    engine = FaceEngine(Path("models"), "model-v1", 320, 0.45, 0.06)
    face = SimpleNamespace(
        bbox=np.array([0, 0, float("nan"), 20]),
        det_score=0.9,
        normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    engine._app = SimpleNamespace(get=lambda _image: [face])

    results = engine.recognize(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [],
        "camera-1",
        1.0,
        datetime(2026, 8, 27, tzinfo=timezone.utc),
    )

    assert results == []
