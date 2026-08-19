from __future__ import annotations

import numpy as np


INT8_MAX = 127


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has zero or invalid norm")
    return array / norm


def quantize_embedding(value: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-vector INT8 quantization for normalized face embeddings."""
    normalized = normalize_embedding(value)
    max_abs = float(np.max(np.abs(normalized)))
    scale = max(max_abs / INT8_MAX, 1e-8)
    quantized = np.clip(np.rint(normalized / scale), -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scale


def dequantize_embedding(value: np.ndarray, scale: float) -> np.ndarray:
    return normalize_embedding(np.asarray(value, dtype=np.int8).astype(np.float32) * float(scale))


def cosine_int8(query_q: np.ndarray, query_scale: float, stored_q: np.ndarray, stored_scale: float) -> float:
    """Cosine similarity with INT32 accumulation; scales cancel after normalization."""
    left = np.asarray(query_q, dtype=np.int8).astype(np.int32)
    right = np.asarray(stored_q, dtype=np.int8).astype(np.int32)
    if left.shape != right.shape:
        raise ValueError("Embedding dimensions differ")
    dot = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return dot / denominator if denominator else 0.0

