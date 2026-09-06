from pathlib import Path
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from pi4_attendance.embedding_catalog import EmbeddingCatalog
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.storage import PiStore


def enrollment_runtime(tmp_path: Path) -> AttendanceRuntime:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(data_root=tmp_path)
    runtime.store = PiStore(tmp_path)
    runtime.engine = SimpleNamespace(
        model_version="model-v1",
        enrollment_embedding=lambda images: (np.array([1, 0], dtype=np.int8), 0.1),
    )
    runtime.inference_lock = threading.RLock()
    runtime._enrollment_lock = threading.Lock()
    runtime.stop_event = threading.Event()
    runtime._catalog_lock = threading.Lock()
    runtime._embedding_catalog = EmbeddingCatalog.from_users([], "model-v1")
    return runtime


def database_counts(store: PiStore) -> tuple[int, int, int]:
    return tuple(
        store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("users", "enrollment_images", "audit_log")
    )


def assert_no_partial_enrollment(runtime: AttendanceRuntime) -> None:
    assert database_counts(runtime.store) == (0, 0, 0)
    assert list(runtime.store.enrollment_root.iterdir()) == []
    assert list(runtime.store.tmp_root.iterdir()) == []
    assert runtime._embedding_catalog.users == ()


def test_store_rolls_back_user_images_and_audit_when_catalog_build_fails(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    image_records = [("image-1", "enrollments/u1/original/01.jpg", "abc", 4, 4)]

    with pytest.raises(RuntimeError, match="catalog failure"):
        store.enroll_user(
            "u1",
            "User",
            "S1",
            np.array([1, 0], dtype=np.int8),
            0.1,
            "model-v1",
            image_records,
            "admin",
            lambda users: (_ for _ in ()).throw(RuntimeError("catalog failure")),
        )

    assert database_counts(store) == (0, 0, 0)
    store.close()


def test_successful_enrollment_commits_metadata_audit_and_catalog_together(tmp_path: Path) -> None:
    runtime = enrollment_runtime(tmp_path)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)

    result = runtime.enroll(" User ", " S1 ", [image], "admin")

    user_id = result["user_id"]
    assert result == {"user_id": user_id, "name": "User", "student_id": "S1", "status": "active"}
    assert database_counts(runtime.store) == (1, 1, 1)
    assert runtime._embedding_catalog.users[0].user_id == user_id
    [saved] = (runtime.store.enrollment_root / user_id / "original").iterdir()
    assert cv2.imread(str(saved)).shape[:2] == (4, 5)
    runtime.store.close()


def test_invalid_embedding_leaves_no_enrollment_state(tmp_path: Path) -> None:
    runtime = enrollment_runtime(tmp_path)
    runtime.engine.enrollment_embedding = lambda images: (np.zeros(2, dtype=np.int8), 0.1)

    with pytest.raises(ValueError, match="non-zero"):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")

    assert_no_partial_enrollment(runtime)
    runtime.store.close()


def test_image_write_failure_removes_partial_temp_file(tmp_path: Path, monkeypatch) -> None:
    runtime = enrollment_runtime(tmp_path)

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        return False

    monkeypatch.setattr(cv2, "imwrite", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="Could not save enrollment image"):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")

    assert_no_partial_enrollment(runtime)
    runtime.store.close()


def test_database_user_creation_failure_removes_staged_files(tmp_path: Path) -> None:
    runtime = enrollment_runtime(tmp_path)
    runtime.store.create_user("existing", "Existing", "S1", np.array([1, 0], dtype=np.int8), 0.1, "model-v1")

    with pytest.raises(Exception, match="UNIQUE constraint"):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")

    assert database_counts(runtime.store) == (1, 0, 0)
    assert list(runtime.store.enrollment_root.iterdir()) == []
    assert list(runtime.store.tmp_root.iterdir()) == []
    assert [user.user_id for user in runtime.store.active_users()] == ["existing"]
    runtime.store.close()


def test_audit_failure_rolls_back_user_and_image_metadata(tmp_path: Path) -> None:
    runtime = enrollment_runtime(tmp_path)

    with pytest.raises(Exception):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], object())

    assert_no_partial_enrollment(runtime)
    runtime.store.close()


def test_catalog_failure_rolls_back_database_and_files(tmp_path: Path, monkeypatch) -> None:
    runtime = enrollment_runtime(tmp_path)
    original = EmbeddingCatalog.from_users

    def fail_nonempty_catalog(users, model_version):
        if users:
            raise RuntimeError("catalog failure")
        return original(users, model_version)

    monkeypatch.setattr(EmbeddingCatalog, "from_users", fail_nonempty_catalog)
    with pytest.raises(RuntimeError, match="catalog failure"):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")

    assert_no_partial_enrollment(runtime)
    runtime.store.close()
