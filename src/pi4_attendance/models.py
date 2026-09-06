from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RegisteredUser:
    user_id: str
    name: str
    student_id: str
    embedding_q: np.ndarray
    embedding_scale: float
    model_version: str


@dataclass(frozen=True)
class FrameCandidate:
    camera_id: str
    captured_monotonic: float
    captured_at: datetime
    frame: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Observation:
    camera_id: str
    observed_monotonic: float
    observed_at: datetime
    candidate_user_id: str | None
    candidate_name: str | None
    similarity: float
    margin: float
    detection_score: float
    bbox: tuple[int, int, int, int]
    quality: float
    model_version: str
    observation_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class RecognitionDecision:
    observation_id: str
    status: str
    user_id: str | None
    name: str | None
    camera_id: str
    observed_at: datetime
    similarity: float
    margin: float
    detection_score: float
    quality: float
    bbox: tuple[int, int, int, int]
    model_version: str
    reason: str
    decided_at: datetime = field(default_factory=utc_now)

