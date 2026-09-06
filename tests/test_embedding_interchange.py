import os
from pathlib import Path

import numpy as np
import pytest

import pi4_attendance.embedding_interchange as interchange
from pi4_attendance.embedding_interchange import export_npz, import_npz
from pi4_attendance.storage import PiStore


def create(store: PiStore, user_id="u1", student_id="S1", model="model-v1") -> None:
    store.create_user(user_id, "User One", student_id, np.array([1, -2, 3], dtype=np.int8), 0.01, model)


def test_npz_round_trip_preserves_users_and_status(tmp_path: Path) -> None:
    source = PiStore(tmp_path / "source")
    create(source)
    source.set_user_status("u1", "inactive")
    archive = tmp_path / "users.npz"
    export_npz(source, archive, "model-v1")
    target = PiStore(tmp_path / "target")

    assert import_npz(target, archive, "model-v1") == 1
    assert target.user_rows() == [{**target.user_rows()[0], "user_id": "u1", "student_id": "S1", "name": "User One", "status": "inactive", "model_version": "model-v1"}]


def test_export_rejects_empty_catalog_without_writing(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "empty.npz"
    with pytest.raises(ValueError):
        export_npz(store, path, "model-v1")
    assert not path.exists()


def test_export_write_failure_preserves_existing_archive_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PiStore(tmp_path / "data")
    create(store)
    path = tmp_path / "users.npz"
    archive(path)
    previous = path.read_bytes()

    def fail_write(target, **values) -> None:
        if hasattr(target, "write"):
            target.write(b"partial archive")
        else:
            Path(target).write_bytes(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail_write)

    with pytest.raises(OSError, match="disk full"):
        export_npz(store, path, "model-v1")

    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_export_validation_failure_preserves_existing_archive_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PiStore(tmp_path / "data")
    create(store)
    path = tmp_path / "users.npz"
    archive(path)
    previous = path.read_bytes()

    def fail_validation(*args, **kwargs):
        raise ValueError("invalid generated archive")

    monkeypatch.setattr(np, "load", fail_validation)

    with pytest.raises(ValueError, match="invalid generated archive"):
        export_npz(store, path, "model-v1")

    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_export_replace_failure_preserves_existing_archive_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = PiStore(tmp_path / "data")
    create(store)
    path = tmp_path / "users.npz"
    archive(path)
    previous = path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        export_npz(store, path, "model-v1")

    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_import_rejects_empty_archive_without_writes(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "empty.npz"
    archive(
        path,
        user_id=np.array([], dtype="U1"),
        student_id=np.array([], dtype="U1"),
        name=np.array([], dtype="U1"),
        status=np.array([], dtype="U1"),
        embeddings=np.empty((0, 3), dtype=np.int8),
        scales=np.array([], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="at least one user"):
        import_npz(store, path, "model-v1")
    assert store.user_rows() == []


def archive(path: Path, **overrides) -> None:
    values = {
        "schema_version": np.array(1, dtype=np.int64),
        "user_id": np.array(["u1"], dtype="U2"),
        "student_id": np.array(["S1"], dtype="U2"),
        "name": np.array(["User"], dtype="U4"),
        "status": np.array(["active"], dtype="U8"),
        "embeddings": np.array([[1, 2, 3]], dtype=np.int8),
        "scales": np.array([0.01], dtype=np.float32),
        "dimension": np.array(3, dtype=np.int64),
        "model_version": np.array("model-v1", dtype="U8"),
    }
    values.update(overrides)
    np.savez(path, **values)


@pytest.mark.parametrize("change", [
    {"scales": np.array([np.nan])},
    {"scales": np.array([0.0])},
    {"dimension": np.array(4)},
    {"user_id": np.array(["u1", "u1"]), "student_id": np.array(["S1", "S2"]), "name": np.array(["A", "B"]), "status": np.array(["active", "active"]), "embeddings": np.array([[1,2,3],[4,5,6]], dtype=np.int8), "scales": np.array([.1,.1])},
    {"model_version": np.array("other")},
    {"embeddings": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
    {"embeddings": np.array([[0, 0, 0]], dtype=np.int8)},
])
def test_import_rejects_malformed_or_incompatible_archive_without_partial_writes(tmp_path: Path, change) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "bad.npz"
    archive(path, **change)
    with pytest.raises(ValueError):
        import_npz(store, path, "model-v1")
    assert store.user_rows() == []


def test_import_rejects_object_arrays_without_writes(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "object.npz"
    archive(path, name=np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError):
        import_npz(store, path, "model-v1")
    assert store.user_rows() == []


@pytest.mark.parametrize("dimension", [np.array([3], dtype=np.int64), np.array(3.0), np.array(3.5)])
def test_import_requires_scalar_integer_dimension(tmp_path: Path, dimension: np.ndarray) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "bad-dimension.npz"
    archive(path, dimension=dimension)

    with pytest.raises(ValueError, match="dimension"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_unreasonable_dimension(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "huge-dimension.npz"
    archive(
        path,
        embeddings=np.ones((1, 4097), dtype=np.int8),
        dimension=np.array(4097, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="dimension"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1, 2, 3], dtype=np.int8),
        np.array([[[1, 2, 3]]], dtype=np.int8),
        np.array([np.array([1, 2]), np.array([3])], dtype=object),
    ],
)
def test_import_rejects_malformed_embedding_shapes(tmp_path: Path, embeddings: np.ndarray) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "malformed-shape.npz"
    archive(path, embeddings=embeddings)

    with pytest.raises(ValueError):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_dimension_incompatible_with_existing_model_records(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    create(store)
    path = tmp_path / "different-dimension.npz"
    archive(
        path,
        user_id=np.array(["u2"]),
        student_id=np.array(["S2"]),
        embeddings=np.array([[1, 2, 3, 4]], dtype=np.int8),
        dimension=np.array(4, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="existing records"):
        import_npz(store, path, "model-v1")

    assert [row["user_id"] for row in store.user_rows()] == ["u1"]


def test_import_rejects_unreasonably_long_text_without_writes(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "long-text.npz"
    archive(path, name=np.array(["x" * 257]))

    with pytest.raises(ValueError, match="names"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_archive_over_compressed_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "too-large.npz"
    archive(path)
    monkeypatch.setattr(interchange, "MAX_ARCHIVE_BYTES", path.stat().st_size - 1, raising=False)

    with pytest.raises(ValueError, match="size limit"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_archive_over_uncompressed_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "expanded-too-large.npz"
    archive(path)
    monkeypatch.setattr(interchange, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1, raising=False)

    with pytest.raises(ValueError, match="expanded size limit"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_unreasonable_user_count_without_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "too-many-users.npz"
    archive(
        path,
        user_id=np.array(["u1", "u2"]),
        student_id=np.array(["S1", "S2"]),
        name=np.array(["One", "Two"]),
        status=np.array(["active", "active"]),
        embeddings=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
        scales=np.array([0.01, 0.01], dtype=np.float32),
    )
    monkeypatch.setattr(interchange, "MAX_ARCHIVE_USERS", 1, raising=False)

    with pytest.raises(ValueError, match="user limit"):
        import_npz(store, path, "model-v1")

    assert store.user_rows() == []


def test_import_rejects_missing_schema_field_without_writes(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    path = tmp_path / "missing.npz"
    np.savez(path, embeddings=np.array([[1, 2, 3]], dtype=np.int8))
    with pytest.raises(ValueError):
        import_npz(store, path, "model-v1")
    assert store.user_rows() == []


def test_import_is_one_transaction_when_database_conflict_occurs(tmp_path: Path) -> None:
    store = PiStore(tmp_path / "data")
    create(store, "existing", "S2")
    path = tmp_path / "conflict.npz"
    archive(path, user_id=np.array(["u1", "u2"]), student_id=np.array(["S1", "S2"]), name=np.array(["A", "B"]), status=np.array(["active", "active"]), embeddings=np.array([[1,2,3],[4,5,6]], dtype=np.int8), scales=np.array([.1,.1]))
    with pytest.raises(Exception):
        import_npz(store, path, "model-v1")
    assert [row["user_id"] for row in store.user_rows()] == ["existing"]
