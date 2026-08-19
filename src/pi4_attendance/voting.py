from __future__ import annotations

import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import uuid4

from .models import Observation, VoteDecision, utc_now


@dataclass
class _Session:
    session_id: str
    opened_monotonic: float
    deadline_monotonic: float
    observations: list[Observation] = field(default_factory=list)


class VoteCoordinator:
    """Groups a one-person passage window and requires agreement from distinct cameras."""

    def __init__(self, window_seconds: float, required_cameras: int, threshold: float, margin: float) -> None:
        self.window_seconds = window_seconds
        self.required_cameras = required_cameras
        self.threshold = threshold
        self.margin = margin
        self._active: _Session | None = None
        self._ready: list[_Session] = []
        self._lock = threading.Lock()

    def add(self, observation: Observation) -> str:
        with self._lock:
            if self._active is None or observation.observed_monotonic > self._active.deadline_monotonic:
                if self._active is not None:
                    self._ready.append(self._active)
                self._active = _Session(
                    uuid4().hex,
                    observation.observed_monotonic,
                    observation.observed_monotonic + self.window_seconds,
                )
            self._active.observations.append(observation)
            return self._active.session_id

    def finalize_due(self, now_monotonic: float, force: bool = False) -> list[VoteDecision]:
        with self._lock:
            ready = self._ready
            self._ready = []
            if self._active is not None and (force or now_monotonic >= self._active.deadline_monotonic):
                ready.append(self._active)
                self._active = None
        return [self._decide(session) for session in ready]

    def _decide(self, session: _Session) -> VoteDecision:
        qualified = [
            item
            for item in session.observations
            if item.candidate_user_id and item.similarity >= self.threshold and item.margin >= self.margin
        ]
        by_user: dict[str, dict[str, Observation]] = defaultdict(dict)
        for item in qualified:
            current = by_user[item.candidate_user_id].get(item.camera_id)
            if current is None or (item.similarity * item.quality) > (current.similarity * current.quality):
                by_user[item.candidate_user_id][item.camera_id] = item
        ranked = sorted(by_user.items(), key=lambda pair: (len(pair[1]), sum(x.similarity for x in pair[1].values())), reverse=True)
        if not ranked or len(ranked[0][1]) < self.required_cameras:
            return VoteDecision(
                session.session_id,
                "rejected",
                None,
                None,
                0.0,
                tuple(sorted({item.camera_id for item in session.observations})),
                "insufficient_cross_camera_agreement",
            )
        winner_id, winner_by_camera = ranked[0]
        if len(ranked) > 1 and len(ranked[1][1]) >= self.required_cameras:
            return VoteDecision(
                session.session_id,
                "conflict",
                None,
                None,
                0.0,
                tuple(sorted({item.camera_id for item in session.observations})),
                "multiple_identities_reached_quorum",
            )
        observations = list(winner_by_camera.values())
        confidence = statistics.median(item.similarity * (0.75 + 0.25 * item.quality) for item in observations)
        return VoteDecision(
            session.session_id,
            "confirmed",
            winner_id,
            observations[0].candidate_name,
            float(confidence),
            tuple(sorted(winner_by_camera)),
            f"{len(winner_by_camera)}_camera_consensus",
            utc_now(),
        )
