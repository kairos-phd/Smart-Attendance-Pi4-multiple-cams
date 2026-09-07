from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from pi4_attendance.embedding_catalog import EmbeddingCatalog
from pi4_attendance.face_engine import FaceEngine
from pi4_attendance.models import RegisteredUser


def user(user_id: str, vector: list[int], scale: float = 0.01, model: str = "model-v1") -> RegisteredUser:
    return RegisteredUser(user_id, user_id.upper(), f"S-{user_id}", np.array(vector, dtype=np.int8), scale, model)


def test_catalog_vectorized_top_two_matches_scalar_cosine() -> None:
    users = [user("u1", [100, 0, 0], 0.01), user("u2", [80, 60, 0], 0.02), user("old", [0, 0, 127], model="old")]
    catalog = EmbeddingCatalog.from_users(users, "model-v1")

    best, score, second = catalog.top_two(np.array([127, 0, 0], dtype=np.int8), 0.5)

    assert best.user_id == "u1"
    assert score == 1.0
    assert np.isclose(second, 0.8)


def test_catalog_is_stable_for_empty_and_single_user() -> None:
    empty = EmbeddingCatalog.from_users([], "model-v1")
    assert empty.top_two(np.array([1, 0], dtype=np.int8), 1.0) == (None, 0.0, 0.0)
    single = EmbeddingCatalog.from_users([user("u1", [1, 0])], "model-v1")
    best, score, second = single.top_two(np.array([1, 0], dtype=np.int8), 2.0)
    assert best.user_id == "u1"
    assert score == 1.0
    assert second == 0.0


def test_catalog_looks_up_user_metadata_without_scanning() -> None:
    expected = user("u1", [1, 0])
    catalog = EmbeddingCatalog.from_users([expected], "model-v1")

    assert catalog.user("u1") is expected
    assert catalog.user("missing") is None


def test_catalog_rejects_mixed_dimensions_and_invalid_scale() -> None:
    with pytest.raises(ValueError):
        EmbeddingCatalog.from_users([user("u1", [1, 2]), user("u2", [1, 2, 3])], "model-v1")
    with pytest.raises(ValueError):
        EmbeddingCatalog.from_users([user("u1", [1, 2], np.nan)], "model-v1")
    with pytest.raises(ValueError, match="non-zero"):
        EmbeddingCatalog.from_users([user("u1", [0, 0])], "model-v1")


def test_catalog_rejects_invalid_query_scale() -> None:
    catalog = EmbeddingCatalog.from_users([user("u1", [1, 0])], "model-v1")
    with pytest.raises(ValueError):
        catalog.top_two(np.array([1, 0], dtype=np.int8), 0.0)


def test_face_engine_recognize_uses_catalog() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    engine.model_version = "model-v1"
    engine._faces = lambda image: [SimpleNamespace(bbox=(0, 0, 16, 16), det_score=0.9, normed_embedding=np.array([1.0, 0.0]))]
    catalog = EmbeddingCatalog.from_users([user("u1", [127, 0], 0.25)], "model-v1")

    [observation] = engine.recognize(np.full((16, 16, 3), 127, dtype=np.uint8), catalog, "cam", 1.0, datetime.now(timezone.utc))

    assert observation.candidate_user_id == "u1"
    assert observation.similarity == 1.0
