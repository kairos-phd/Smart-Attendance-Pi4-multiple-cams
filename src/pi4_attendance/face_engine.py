from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .models import Observation, RegisteredUser
from .quantization import cosine_int8, normalize_embedding, quantize_embedding

LOG = logging.getLogger(__name__)


def _face_quality(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    area_score = min(1.0, (crop.shape[0] * crop.shape[1]) / max(1.0, width * height * 0.08))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0)
    brightness = float(np.mean(gray))
    exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    return float(0.45 * area_score + 0.35 * sharpness + 0.20 * exposure)


class FaceEngine:
    """InsightFace runtime with locally stored INT8 embeddings and calibrated matching."""

    def __init__(self, model_root: Path, model_pack: str, det_size: int, threshold: float, margin: float) -> None:
        self.model_root = model_root
        self.model_pack = model_pack
        self.det_size = det_size
        self.threshold = threshold
        self.margin = margin
        self._app = None
        self.model_version = model_pack

    def start(self) -> None:
        from insightface.app import FaceAnalysis

        manifest_path = self.model_root / "models" / self.model_pack / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_version = str(manifest.get("model_version", self.model_pack))
        self._app = FaceAnalysis(name=self.model_pack, root=str(self.model_root), providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        LOG.info("Loaded model pack %s (%s)", self.model_pack, self.model_version)

    def _faces(self, image: np.ndarray):
        if self._app is None:
            raise RuntimeError("Face engine is not started")
        return self._app.get(image)

    def enrollment_embedding(self, images: list[np.ndarray]) -> tuple[np.ndarray, float]:
        embeddings: list[np.ndarray] = []
        for image in images:
            faces = self._faces(image)
            if len(faces) != 1:
                raise ValueError("Each enrollment image must contain exactly one clear foreground face")
            face = faces[0]
            if float(face.det_score) < 0.60:
                raise ValueError("Face detection confidence is too low")
            bbox = tuple(int(value) for value in face.bbox)
            if _face_quality(image, bbox) < 0.35:
                raise ValueError("An enrollment image is too small, blurry, dark, or overexposed")
            embeddings.append(normalize_embedding(np.asarray(face.normed_embedding, dtype=np.float32)))
        if not 3 <= len(embeddings) <= 8:
            raise ValueError("Enrollment requires 3 to 8 images")
        reference = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
        consistency = min(float(np.dot(reference, item)) for item in embeddings)
        if consistency < 0.55:
            raise ValueError("Enrollment images do not appear to show the same person")
        return quantize_embedding(reference)

    def recognize(
        self,
        image: np.ndarray,
        users: list[RegisteredUser],
        camera_id: str,
        observed_monotonic: float,
        observed_at: datetime,
    ) -> list[Observation]:
        results: list[Observation] = []
        for face in self._faces(image):
            bbox = tuple(int(value) for value in face.bbox)
            query_q, query_scale = quantize_embedding(np.asarray(face.normed_embedding, dtype=np.float32))
            ranked = sorted(
                (
                    (cosine_int8(query_q, query_scale, user.embedding_q, user.embedding_scale), user)
                    for user in users
                    if user.model_version == self.model_version
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            best_score, best_user = ranked[0] if ranked else (0.0, None)
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            match_margin = best_score - second_score
            accepted = best_user if best_score >= self.threshold and match_margin >= self.margin else None
            results.append(
                Observation(
                    camera_id=camera_id,
                    observed_monotonic=observed_monotonic,
                    observed_at=observed_at,
                    candidate_user_id=accepted.user_id if accepted else None,
                    candidate_name=accepted.name if accepted else None,
                    similarity=float(best_score),
                    margin=float(match_margin),
                    detection_score=float(face.det_score),
                    bbox=bbox,
                    quality=_face_quality(image, bbox),
                )
            )
        return results
