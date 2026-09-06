from __future__ import annotations

import math

from .models import Observation, RecognitionDecision


class RecognitionPolicy:
    def __init__(
        self,
        similarity_threshold: float,
        margin_threshold: float,
        detection_threshold: float,
        quality_threshold: float,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.detection_threshold = detection_threshold
        self.quality_threshold = quality_threshold

    def decide(self, observation: Observation) -> RecognitionDecision:
        reason = "thresholds_passed"
        metrics = (
            observation.similarity,
            observation.margin,
            observation.detection_score,
            observation.quality,
        )
        if not all(math.isfinite(value) for value in metrics):
            reason = "non_finite_metric"
        elif not observation.candidate_user_id or not observation.candidate_name:
            reason = "no_candidate"
        elif observation.similarity < self.similarity_threshold:
            reason = "similarity_below_threshold"
        elif observation.margin < self.margin_threshold:
            reason = "margin_below_threshold"
        elif observation.detection_score < self.detection_threshold:
            reason = "detection_score_below_threshold"
        elif observation.quality < self.quality_threshold:
            reason = "quality_below_threshold"
        accepted = reason == "thresholds_passed"
        return RecognitionDecision(
            observation_id=observation.observation_id,
            status="accepted" if accepted else "rejected",
            user_id=observation.candidate_user_id if accepted else None,
            name=observation.candidate_name if accepted else None,
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            similarity=observation.similarity,
            margin=observation.margin,
            detection_score=observation.detection_score,
            quality=observation.quality,
            bbox=observation.bbox,
            model_version=observation.model_version,
            reason=reason,
        )
