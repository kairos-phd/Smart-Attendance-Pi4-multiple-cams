from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .models import RegisteredUser


@dataclass(frozen=True)
class EmbeddingCatalog:
    users: tuple[RegisteredUser, ...]
    vectors: np.ndarray
    model_version: str
    users_by_id: Mapping[str, RegisteredUser]

    @classmethod
    def from_users(cls, users: list[RegisteredUser], model_version: str) -> "EmbeddingCatalog":
        compatible = tuple(user for user in users if user.model_version == model_version)
        users_by_id = MappingProxyType({user.user_id: user for user in compatible})
        if not compatible:
            return cls((), np.empty((0, 0), dtype=np.float32), model_version, users_by_id)
        dimension = compatible[0].embedding_q.size
        if dimension <= 0 or any(user.embedding_q.dtype != np.int8 or user.embedding_q.ndim != 1 or user.embedding_q.size != dimension for user in compatible):
            raise ValueError("Compatible embeddings must be non-empty INT8 vectors with one dimension")
        if any(not np.isfinite(user.embedding_scale) or user.embedding_scale <= 0 for user in compatible):
            raise ValueError("Embedding scales must be positive and finite")
        matrix = np.stack(
            [user.embedding_q.astype(np.float32) * float(user.embedding_scale) for user in compatible]
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
        if not np.all(valid):
            raise ValueError("Compatible embeddings must be finite non-zero vectors")
        if matrix.size:
            matrix = np.ascontiguousarray(matrix / norms)
            matrix.flags.writeable = False
        return cls(compatible, matrix, model_version, users_by_id)

    def user(self, user_id: str | None) -> RegisteredUser | None:
        return None if user_id is None else self.users_by_id.get(user_id)

    def top_two(self, query_q: np.ndarray, query_scale: float) -> tuple[RegisteredUser | None, float, float]:
        if not np.isfinite(query_scale) or query_scale <= 0:
            raise ValueError("Query scale must be positive and finite")
        if not self.users:
            return None, 0.0, 0.0
        raw_query = np.asarray(query_q)
        if raw_query.dtype != np.int8 or raw_query.ndim != 1:
            raise ValueError("Query must be a one-dimensional INT8 embedding")
        query = raw_query.astype(np.float32) * float(query_scale)
        if query.size != self.vectors.shape[1]:
            return None, 0.0, 0.0
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= 0:
            return None, 0.0, 0.0
        scores = self.vectors @ (query / norm)
        if scores.size == 1:
            return self.users[0], float(scores[0]), 0.0
        indices = np.argpartition(scores, -2)[-2:]
        ordered = indices[np.argsort(scores[indices])[::-1]]
        return self.users[int(ordered[0])], float(scores[ordered[0]]), float(scores[ordered[1]])
