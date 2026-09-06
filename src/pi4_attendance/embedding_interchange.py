from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

import numpy as np

from .storage import PiStore


ARCHIVE_SCHEMA_VERSION = 1
REQUIRED = {"schema_version", "user_id", "student_id", "name", "status", "embeddings", "scales", "dimension", "model_version"}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_USERS = 10_000
MAX_EMBEDDING_DIMENSION = 4096
MAX_TEXT_LENGTH = 256


def export_npz(store: PiStore, path: Path, model_version: str) -> int:
    rows = [row for row in store.embedding_rows() if row["model_version"] == model_version]
    if not rows:
        raise ValueError("No compatible embeddings to export")
    dimension = rows[0]["embedding_q"].size if rows else 0
    if any(row["embedding_q"].size != dimension for row in rows):
        raise ValueError("Stored embeddings have inconsistent dimensions")
    if any(not np.any(row["embedding_q"]) for row in rows):
        raise ValueError("Stored embeddings must be non-zero")
    embeddings = np.stack([row["embedding_q"] for row in rows])
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                schema_version=np.array(ARCHIVE_SCHEMA_VERSION, dtype=np.int64),
                user_id=np.array([row["user_id"] for row in rows], dtype=str),
                student_id=np.array([row["student_id"] for row in rows], dtype=str),
                name=np.array([row["name"] for row in rows], dtype=str),
                status=np.array([row["status"] for row in rows], dtype=str),
                embeddings=embeddings,
                scales=np.array([row["embedding_scale"] for row in rows], dtype=np.float32),
                dimension=np.array(dimension, dtype=np.int64),
                model_version=np.array(model_version, dtype=str),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        _read_npz(temporary_path, model_version)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(rows)


def _text_vector(value: np.ndarray, name: str, count: int) -> list[str]:
    if value.dtype.kind not in "US" or value.ndim != 1 or len(value) != count:
        raise ValueError(f"Invalid {name}")
    result = [str(item).strip() for item in value]
    if any(not item or len(item) > MAX_TEXT_LENGTH for item in result):
        raise ValueError(f"Invalid {name}")
    return result


def _integer_scalar(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"Invalid {name}")
    return int(array.item())


def _text_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "US":
        raise ValueError(f"Invalid {name}")
    result = str(array.item()).strip()
    if not result or len(result) > MAX_TEXT_LENGTH:
        raise ValueError(f"Invalid {name}")
    return result


def _read_npz(path: Path, expected_model_version: str) -> list[tuple[str, str, str, np.ndarray, float, str, str]]:
    try:
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds size limit")
        with ZipFile(path) as zip_file:
            if sum(item.file_size for item in zip_file.infolist()) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("Archive exceeds expanded size limit")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED:
                raise ValueError("Archive fields do not match schema")
            if _integer_scalar(archive["schema_version"], "schema version") != ARCHIVE_SCHEMA_VERSION:
                raise ValueError("Unsupported archive schema")
            model_version = _text_scalar(archive["model_version"], "model version")
            if model_version != expected_model_version:
                raise ValueError("Archive model version is incompatible")
            embeddings = np.asarray(archive["embeddings"])
            scales = np.asarray(archive["scales"])
            dimension = _integer_scalar(archive["dimension"], "embedding dimension")
            if not 0 < dimension <= MAX_EMBEDDING_DIMENSION:
                raise ValueError("Invalid embedding dimension")
            if embeddings.dtype != np.int8 or embeddings.ndim != 2 or embeddings.shape[1] != dimension:
                raise ValueError("Invalid INT8 embedding matrix")
            count = embeddings.shape[0]
            if count == 0:
                raise ValueError("Archive must contain at least one user")
            if count > MAX_ARCHIVE_USERS:
                raise ValueError("Archive exceeds user limit")
            if scales.ndim != 1 or len(scales) != count or not np.issubdtype(scales.dtype, np.floating):
                raise ValueError("Invalid scales")
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("Scales must be positive and finite")
            if np.any(np.all(embeddings == 0, axis=1)):
                raise ValueError("Embeddings must be non-zero")
            user_ids = _text_vector(np.asarray(archive["user_id"]), "user IDs", count)
            student_ids = _text_vector(np.asarray(archive["student_id"]), "student IDs", count)
            names = _text_vector(np.asarray(archive["name"]), "names", count)
            statuses = _text_vector(np.asarray(archive["status"]), "statuses", count)
    except (OSError, BadZipFile, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError("Invalid NumPy archive") from exc
    if len(set(user_ids)) != count or len(set(student_ids)) != count:
        raise ValueError("Duplicate user or student ID")
    if any(status not in {"active", "inactive"} for status in statuses):
        raise ValueError("Invalid user status")
    rows = [
        (user_ids[i], names[i], student_ids[i], embeddings[i].copy(), float(scales[i]), model_version, statuses[i])
        for i in range(count)
    ]
    return rows


def import_npz(store: PiStore, path: Path, expected_model_version: str) -> int:
    rows = _read_npz(path, expected_model_version)
    store.import_users(rows)
    return len(rows)
