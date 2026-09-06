from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import threading

import cv2
import numpy as np

from pi4_attendance.models import Observation, RecognitionDecision
from pi4_attendance.models import FrameCandidate
from pi4_attendance.service import AttendanceRuntime, RuntimeMetrics
from pi4_attendance.embedding_catalog import EmbeddingCatalog
from pi4_attendance.storage import PiStore
import pytest


def test_runtime_metrics_report_latest_and_aggregate_latencies() -> None:
    metrics = RuntimeMetrics()
    metrics.record(12.0, 30.0)
    metrics.record(18.0, 50.0)

    assert metrics.snapshot() == {
        "queue_age_ms": {"latest": 18.0, "aggregate": {"samples": 2, "average": 15.0}},
        "inference_ms": {"latest": 50.0, "aggregate": {"samples": 2, "average": 40.0}},
    }


def runtime_for_health(state="running", camera_rows=None, inference_alive=True):
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(pi_id="pi-1", cameras=(SimpleNamespace(camera_id="camera-1"),))
    runtime.frames = SimpleNamespace(qsize=lambda: 2)
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime._started = 1.0
    runtime._runtime_state = state
    runtime._inference_thread = SimpleNamespace(is_alive=lambda: inference_alive)
    rows = [] if camera_rows is None else [dict({"camera_id": "camera-1"}, **row) for row in camera_rows]
    runtime.store = SimpleNamespace(camera_rows=lambda: rows)
    runtime.metrics = RuntimeMetrics()
    runtime.metrics.record(10.0, 20.0)
    return runtime


def test_health_exposes_metrics_and_persisted_dropped_frame_counter() -> None:
    runtime = runtime_for_health(camera_rows=[{"status": "online", "dropped_frames": 7}])

    health = runtime.health()

    assert health["status"] == "ok"
    assert health["dropped_frames"] == 7
    assert health["runtime_metrics"]["queue_age_ms"]["latest"] == 10.0
    assert health["runtime_metrics"]["inference_ms"]["aggregate"]["average"] == 20.0


def test_health_ignores_dropped_frames_from_stale_historical_camera() -> None:
    runtime = runtime_for_health(
        camera_rows=[
            {"status": "online", "dropped_frames": 7},
            {"camera_id": "retired-camera", "status": "offline", "dropped_frames": 99},
        ]
    )

    health = runtime.health()

    assert health["status"] == "ok"
    assert health["dropped_frames"] == 7


def test_health_reports_starting_degraded_and_stopped_honestly() -> None:
    assert runtime_for_health(state="starting").health()["status"] == "starting"
    assert runtime_for_health(camera_rows=[]).health()["status"] == "starting"
    assert runtime_for_health(camera_rows=[{"status": "offline", "dropped_frames": 2}]).health()["status"] == "degraded"
    assert runtime_for_health(camera_rows=[{"status": "online", "dropped_frames": 2}], inference_alive=False).health()["status"] == "degraded"
    assert runtime_for_health(state="stopped").health()["status"] == "stopped"


def test_unknown_retention_writes_only_crop_for_matching_observation(tmp_path: Path, monkeypatch) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(data_root=tmp_path)
    runtime.store = SimpleNamespace(unknown_root=tmp_path / "unknown", tmp_root=tmp_path / "tmp")
    runtime.store.tmp_root.mkdir()
    observed_at = datetime(2026, 8, 27, tzinfo=timezone.utc)
    observation = Observation(
        "camera-1", 1.0, observed_at, None, None, 0.0, 0.0, 0.9,
        (10, 10, 30, 30), 0.8, "model-v1", "obs-correct",
    )
    other = Observation(
        "camera-2", 1.0, observed_at, None, None, 0.0, 0.0, 0.9,
        (0, 0, 50, 50), 1.0, "model-v1", "obs-other",
    )
    decision = RecognitionDecision(
        "obs-correct", "rejected", None, None, "camera-1", observed_at,
        0.0, 0.0, 0.9, 0.8, (10, 10, 30, 30), "model-v1", "no_candidate",
    )
    frame = np.full((60, 60, 3), 127, dtype=np.uint8)

    relative = runtime._save_unknown(decision, [(other, frame), (observation, frame)])

    saved = cv2.imread(str(tmp_path / relative))
    assert saved.shape[:2] == (42, 42)
    assert saved.shape[:2] != frame.shape[:2]


def test_unknown_retention_skips_invalid_crop(tmp_path: Path) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(data_root=tmp_path)
    runtime.store = SimpleNamespace(unknown_root=tmp_path / "unknown", tmp_root=tmp_path / "tmp")
    runtime.store.tmp_root.mkdir()
    observed_at = datetime(2026, 8, 27, tzinfo=timezone.utc)
    observation = Observation(
        "camera-1", 1.0, observed_at, None, None, 0.0, 0.0, 0.9,
        (100, 100, 110, 110), 0.8, "model-v1", "obs-1",
    )
    decision = RecognitionDecision(
        "obs-1", "rejected", None, None, "camera-1", observed_at,
        0.0, 0.0, 0.9, 0.8, observation.bbox, "model-v1", "no_candidate",
    )

    assert runtime._save_unknown(decision, [(observation, np.zeros((40, 40, 3), dtype=np.uint8))]) is None
    assert not runtime.store.unknown_root.exists()


def test_runtime_refreshes_cache_only_after_successful_mutations() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime._catalog_lock = threading.Lock()
    calls = []
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime.store = SimpleNamespace(
        active_users=lambda: calls.append("load") or [],
        set_user_status_audited=lambda user_id, status, actor: status == "inactive",
    )
    runtime._embedding_catalog = EmbeddingCatalog.from_users([], "model-v1")
    assert runtime.set_user_status("u1", "inactive", "admin") is True
    assert calls == ["load"]
    assert runtime.set_user_status("u1", "active", "admin") is False
    assert calls == ["load"]


def test_runtime_catalog_reads_sqlite_once_not_once_per_recognition() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime._catalog_lock = threading.Lock()
    calls = []
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime.store = SimpleNamespace(active_users=lambda: calls.append("load") or [])

    runtime._refresh_embedding_catalog()
    first = runtime._embedding_catalog
    second = runtime._embedding_catalog

    assert first is second
    assert calls == ["load"]


def test_two_recognition_frames_use_cached_catalog_without_database_reads() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    candidates = [FrameCandidate("cam", 1.0, datetime.now(timezone.utc), frame), FrameCandidate("cam", 2.0, datetime.now(timezone.utc), frame)]
    runtime.stop_event = SimpleNamespace(is_set=lambda: not candidates)
    runtime.frames = SimpleNamespace(get=lambda timeout: candidates.pop(0))
    runtime._embedding_catalog = EmbeddingCatalog.from_users([], "model-v1")
    seen = []
    runtime.engine = SimpleNamespace(recognize=lambda image, catalog, *args: seen.append(catalog) or [])
    runtime.inference_lock = __import__("threading").RLock()
    runtime.recognition_enabled = threading.Event()
    runtime.recognition_enabled.set()
    runtime.metrics = RuntimeMetrics()
    runtime.store = SimpleNamespace(active_users=lambda: (_ for _ in ()).throw(AssertionError("per-frame DB read")))
    runtime._inference_loop()
    assert seen == [runtime._embedding_catalog, runtime._embedding_catalog]


def test_inference_discards_queued_frame_after_preview_disables_recognition() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    candidates = [FrameCandidate("cam", 1.0, datetime.now(timezone.utc), frame)]
    runtime.stop_event = SimpleNamespace(is_set=lambda: not candidates)
    runtime.frames = SimpleNamespace(get=lambda timeout: candidates.pop(0))
    runtime.recognition_enabled = threading.Event()
    runtime.engine = SimpleNamespace(recognize=lambda *args: (_ for _ in ()).throw(AssertionError("recognition ran")))

    runtime._inference_loop()


def test_preview_failure_still_stops_threads_and_closes_store() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    calls = []
    runtime.start = lambda: calls.append("start")
    runtime.stop = lambda: calls.append("stop")
    runtime.stop_event = threading.Event()
    runtime.preview = SimpleNamespace(run=lambda stop: (_ for _ in ()).throw(RuntimeError("preview failed")))
    runtime._workers = [SimpleNamespace(join=lambda timeout: calls.append("worker"), is_alive=lambda: False)]
    runtime._inference_thread = runtime._retention_thread = runtime._cloud_thread = runtime._api_thread = None
    runtime._enrollment_lock = threading.Lock()
    runtime.store = SimpleNamespace(close=lambda: calls.append("close"))

    with pytest.raises(RuntimeError, match="preview failed"):
        runtime.run()

    assert calls == ["start", "stop", "worker", "close"]


def test_shutdown_does_not_close_store_under_blocked_camera_worker() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    closed = []
    runtime.start = lambda: None
    runtime.stop = lambda: runtime.stop_event.set()
    runtime.stop_event = threading.Event()
    runtime.preview = SimpleNamespace(run=lambda stop: stop.set())
    runtime._workers = [SimpleNamespace(name="blocked-camera", join=lambda timeout: None, is_alive=lambda: True)]
    runtime._inference_thread = runtime._retention_thread = runtime._cloud_thread = runtime._api_thread = None
    runtime._enrollment_lock = threading.Lock()
    runtime.store = SimpleNamespace(close=lambda: closed.append(True))

    runtime.run()

    assert closed == []


def test_failed_atomic_status_audit_keeps_database_and_cache_consistent(tmp_path: Path) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime._catalog_lock = threading.Lock()
    runtime.store = PiStore(tmp_path)
    runtime.store.create_user("u1", "User", "S1", np.array([1, 0], dtype=np.int8), 0.1, "model-v1")
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime._refresh_embedding_catalog()
    original = runtime._embedding_catalog
    with pytest.raises(Exception):
        runtime.set_user_status("u1", "inactive", object())
    assert runtime.store.user_rows()[0]["status"] == "active"
    assert runtime._embedding_catalog is original
    runtime.store.close()


def test_stale_catalog_refresh_cannot_restore_an_inactive_user(tmp_path: Path) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.store = PiStore(tmp_path)
    runtime.store.create_user("u1", "User", "S1", np.array([1, 0], dtype=np.int8), 0.1, "model-v1")
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime._catalog_lock = threading.Lock()
    runtime._embedding_catalog = EmbeddingCatalog.from_users([], "model-v1")
    original_active_users = runtime.store.active_users
    first_read = threading.Event()
    second_read = threading.Event()
    release_first = threading.Event()
    reads = 0
    reads_lock = threading.Lock()

    def ordered_active_users():
        nonlocal reads
        users = original_active_users()
        with reads_lock:
            reads += 1
            current_read = reads
        if current_read == 1:
            first_read.set()
            assert release_first.wait(1)
        else:
            second_read.set()
        return users

    runtime.store.active_users = ordered_active_users
    stale_refresh = threading.Thread(target=runtime._refresh_embedding_catalog)
    deactivate = threading.Thread(target=runtime.set_user_status, args=("u1", "inactive", "admin"))
    stale_refresh.start()
    assert first_read.wait(1)
    deactivate.start()
    second_read.wait(0.2)
    release_first.set()
    stale_refresh.join(1)
    deactivate.join(1)
    status = runtime.store.user_rows()[0]["status"]
    match = runtime._embedding_catalog.top_two(np.array([1, 0], dtype=np.int8), 0.1)
    runtime.store.close()

    assert not stale_refresh.is_alive()
    assert not deactivate.is_alive()
    assert status == "inactive"
    assert match == (None, 0.0, 0.0)


def test_failed_enrollment_removes_user_loaded_by_concurrent_refresh(tmp_path: Path, monkeypatch) -> None:
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
    image_write_started = threading.Event()
    release_image_write = threading.Event()
    errors = []

    def fail_image_write(*args, **kwargs):
        image_write_started.set()
        assert release_image_write.wait(1)
        return False

    monkeypatch.setattr(cv2, "imwrite", fail_image_write)

    def enroll() -> None:
        try:
            runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")
        except RuntimeError as exc:
            errors.append(exc)

    enrollment = threading.Thread(target=enroll)
    enrollment.start()
    assert image_write_started.wait(1)
    refresh = threading.Thread(target=runtime._refresh_embedding_catalog)
    refresh.start()
    refresh.join(1)
    assert not refresh.is_alive()
    release_image_write.set()
    enrollment.join(1)
    users = runtime.store.user_rows()
    match = runtime._embedding_catalog.top_two(np.array([1, 0], dtype=np.int8), 0.1)
    runtime.store.close()

    assert not enrollment.is_alive()
    assert len(errors) == 1
    assert users == []
    assert match == (None, 0.0, 0.0)


def test_enrollment_directory_failure_rolls_back_created_user(tmp_path: Path, monkeypatch) -> None:
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
    original_mkdir = Path.mkdir

    def fail_user_directory(path, *args, **kwargs):
        if path.name == "original" and path.parent.parent == runtime.store.enrollment_root:
            raise OSError("injected enrollment directory failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_user_directory)

    with pytest.raises(OSError, match="injected enrollment directory failure"):
        runtime.enroll("User", "S1", [np.zeros((4, 4, 3), dtype=np.uint8)], "admin")

    users = runtime.store.user_rows()
    match = runtime._embedding_catalog.top_two(np.array([1, 0], dtype=np.int8), 0.1)
    runtime.store.close()
    assert users == []
    assert match == (None, 0.0, 0.0)
