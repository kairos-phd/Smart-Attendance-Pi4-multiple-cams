import asyncio
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import threading
import time

import cv2
import httpx
import numpy as np
import uvicorn

from pi4_attendance.models import Observation, RecognitionDecision, RegisteredUser
from pi4_attendance.models import FrameCandidate
from pi4_attendance.service import AttendanceRuntime, RuntimeMetrics
from pi4_attendance.preview import PreviewWindow
from pi4_attendance.embedding_catalog import EmbeddingCatalog
from pi4_attendance.storage import PiStore
from pi4_attendance.telemetry import RollingMetrics
import pytest


def test_runtime_recognitions_reads_accepted_events_during_cooldown(tmp_path: Path) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.store = PiStore(tmp_path)
    try:
        runtime.store.create_user("u1", "Dang", "S001", np.array([1], dtype=np.int8), 0.01, "model-v1")
        for index in range(2):
            runtime.store.record_decision(RecognitionDecision(
                f"o{index}", "accepted", "u1", "Dang", f"camera-{index + 1}", datetime.now(timezone.utc),
                0.8, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
            ), 300)
        rows = runtime.recognitions(2)
        assert [row["camera_ids"] for row in rows] == [["camera-2"], ["camera-1"]]
        assert len(runtime.recognitions(1)) == 1
        assert len(runtime.attendance(2)) == 1
    finally:
        runtime.store.close()


def test_headless_service_import_does_not_require_gui_modules() -> None:
    script = """
import sys

class BlockGuiImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tkinter" or fullname.startswith("tkinter.") or fullname == "PIL" or fullname.startswith("PIL."):
            raise ModuleNotFoundError(f"blocked GUI module: {fullname}")
        return None

sys.meta_path.insert(0, BlockGuiImports())
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.preview import PreviewWindow

state = PreviewWindow(("camera-1",), 30.0)
state.set_recognition(True)
assert state.status()["recognition_enabled"] is True
assert AttendanceRuntime.control_status
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_api_server_ignores_forwarded_client_addresses(monkeypatch) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(
        api_host="127.0.0.1",
        api_port=8000,
        api_cors_origins=[],
        api_auth_mode="shared_token",
        admin_shared_token="secret",
    )
    runtime.preview = PreviewWindow(("camera-1",), 30.0)
    runtime._camera_cloud_lock = threading.Lock()
    runtime._online_camera_ids = {"camera-1"}
    runtime._runtime_state = "running"
    statuses = []

    class TestServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            self.config.load()

            async def exercise():
                cases = (
                    (("198.51.100.10", 41000), "127.0.0.1"),
                    (("127.0.0.1", 41000), "198.51.100.10"),
                )
                for client_address, forwarded_address in cases:
                    transport = httpx.ASGITransport(app=self.config.loaded_app, client=client_address)
                    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
                        response = await client.get(
                            "/api/control", headers={"X-Forwarded-For": forwarded_address}
                        )
                        statuses.append(response.status_code)

            asyncio.run(exercise())

    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    monkeypatch.setattr(uvicorn, "Server", TestServer)

    runtime._api_loop()

    assert statuses == [403, 200]


def test_runtime_metrics_report_latest_and_aggregate_latencies() -> None:
    metrics = RuntimeMetrics()
    metrics.record(12.0, 30.0)
    metrics.record(18.0, 50.0)

    assert metrics.snapshot() == {
        "queue_age_ms": {"latest": 18.0, "aggregate": {"samples": 2, "average": 15.0}},
        "inference_ms": {"latest": 50.0, "aggregate": {"samples": 2, "average": 40.0}},
    }


@pytest.mark.parametrize(("show_preview", "recognition_enabled"), ((False, True), (True, False)))
def test_every_runtime_prepares_shared_control_and_performance_log(
    tmp_path: Path, show_preview: bool, recognition_enabled: bool
) -> None:
    settings = SimpleNamespace(
        data_root=tmp_path,
        model_root=tmp_path,
        model_pack="pi4_fp32",
        pi_id="pi-1",
        cloud_summary_enabled=False,
        face_det_size=320,
        match_threshold=0.45,
        match_margin=0.06,
        min_detection_score=0.60,
        min_face_quality=0.35,
        frame_queue_size=3,
        preview_fps=30.0,
        cameras=(SimpleNamespace(camera_id="camera-1", name="Camera 1", source=0),),
        frame_width=640,
        frame_height=480,
        monitor_fps=4.0,
        camera_trigger_cooldown_seconds=0.8,
    )

    runtime = AttendanceRuntime(settings, preview=show_preview)

    assert runtime.preview is not None
    assert runtime.recognition_enabled.is_set() is recognition_enabled
    assert runtime.performance_log_path.parent == tmp_path / "performance"
    assert runtime.performance_log_path.suffix == ".jsonl"
    assert runtime._telemetry_monitor is not None
    assert runtime._telemetry_writer is not None
    runtime.store.close()


def test_telemetry_lifecycle_writes_run_metadata_and_stops_monitor_before_writer(tmp_path: Path) -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime._telemetry_run_id = "run-1"
    runtime.performance_log_path = tmp_path / "performance" / "run-1.jsonl"
    runtime.settings = SimpleNamespace(
        pi_id="pi-1",
        model_pack="pi4_fp32",
        cameras=(SimpleNamespace(camera_id="camera-1", name="Camera 1", source="/dev/video0"),),
        frame_width=640,
        frame_height=480,
        monitor_fps=4.0,
        preview_fps=30.0,
        face_det_size=320,
        camera_trigger_cooldown_seconds=0.8,
    )
    runtime.engine = SimpleNamespace(model_version="model-v1")
    runtime.performance_metrics = RollingMetrics()
    runtime.performance_metrics.increment("inference.processed")
    calls = []
    submitted = []
    runtime._telemetry_writer = SimpleNamespace(
        submit=lambda row: submitted.append(row) or True,
        start=lambda: calls.append("writer-start"),
        stop=lambda: calls.append("writer-stop"),
    )
    runtime._telemetry_monitor = SimpleNamespace(
        start=lambda: calls.append("monitor-start"),
        stop=lambda: calls.append("monitor-stop"),
    )

    runtime._start_telemetry()
    runtime._stop_telemetry()

    assert submitted[0]["type"] == "metadata"
    assert submitted[0]["run_id"] == "run-1"
    assert submitted[0]["camera_count"] == 1
    assert submitted[0]["cameras"][0]["source"] == "/dev/video0"
    assert submitted[-1]["type"] == "final"
    assert submitted[-1]["runtime"]["counters"]["inference.processed"] == 1
    assert calls == ["writer-start", "monitor-start", "monitor-stop", "writer-stop"]


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
    runtime.performance_metrics = RollingMetrics()
    runtime.performance_metrics.increment("inference.processed")
    runtime.performance_log_path = Path("performance/run.jsonl")
    runtime._telemetry_writer = SimpleNamespace(is_alive=lambda: True, error=None, dropped=0)
    runtime._telemetry_monitor = SimpleNamespace(
        is_alive=lambda: True,
        latest_sample={"cpu_percent": 12.5, "process_rss_mb": 128.0},
    )
    return runtime


def test_health_exposes_metrics_and_persisted_dropped_frame_counter() -> None:
    runtime = runtime_for_health(camera_rows=[{"status": "online", "dropped_frames": 7}])

    health = runtime.health()

    assert health["status"] == "ok"
    assert health["dropped_frames"] == 7
    assert health["runtime_metrics"]["queue_age_ms"]["latest"] == 10.0
    assert health["runtime_metrics"]["inference_ms"]["aggregate"]["average"] == 20.0
    assert health["performance_metrics"]["counters"]["inference.processed"] == 1
    assert health["performance_log"] == str(Path("performance/run.jsonl"))
    assert health["system"] == {"cpu_percent": 12.5, "process_rss_mb": 128.0}
    assert health["telemetry"] == {
        "monitor_alive": True,
        "writer_alive": True,
        "writer_error": None,
        "samples_dropped": 0,
    }


def test_runtime_control_wrappers_use_shared_state_and_preserve_actor() -> None:
    enrolled = []
    started = threading.Event()
    release = threading.Event()

    def enroll(name, student_id, images, actor):
        started.set()
        assert release.wait(1)
        enrolled.append((name, student_id, images, actor))
        return {}

    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.preview = PreviewWindow(("camera-1", "camera-2"), 30.0, enroll)
    runtime.recognition_enabled = runtime.preview.recognition_enabled

    assert runtime.set_recognition(True)["recognition_enabled"] is True
    assert runtime.select_camera("camera-2")["selected_camera_id"] == "camera-2"
    assert runtime.begin_enrollment("Dang", "S001", "local-api")["enrollment"]["name"] == "Dang"
    for value in range(3):
        runtime.preview.submit("camera-2", np.full((2, 2, 3), value, dtype=np.uint8))
        assert runtime.capture_enrollment_frame()["enrollment"]["captured"] == value + 1
    assert runtime.submit_enrollment()["enrollment_busy"] is True
    assert started.wait(1)
    release.set()
    deadline = time.monotonic() + 1
    while runtime.preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert enrolled[0][3] == "local-api"


def test_accepted_decision_updates_preview_with_catalog_student_id() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    observed_at = datetime(2026, 9, 7, 3, 42, 18, tzinfo=timezone.utc)
    decision = RecognitionDecision(
        "obs-1", "accepted", "u1", "Nguyễn Văn A", "camera-2", observed_at,
        0.823, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
    )
    runtime.settings = SimpleNamespace(attendance_cooldown_seconds=300)
    runtime.store = SimpleNamespace(record_decision_result=lambda *_: (True, "recorded"))
    runtime._cloud_wake = SimpleNamespace(set=lambda: None)
    runtime._embedding_catalog = EmbeddingCatalog.from_users(
        [RegisteredUser("u1", "Nguyễn Văn A", "S001", np.array([1], dtype=np.int8), 0.1, "model-v1")],
        "model-v1",
    )
    shown = []
    runtime.preview = SimpleNamespace(show_recognition=lambda camera_id, **data: shown.append((camera_id, data)))
    runtime.performance_metrics = RollingMetrics()

    runtime._handle_decision(decision, SimpleNamespace(), np.empty((0, 0)))

    assert shown == [
        (
            "camera-2",
            {
                "name": "Nguyễn Văn A",
                "student_id": "S001",
                "similarity": 0.823,
                "timestamp": observed_at.astimezone().strftime("%H:%M:%S"),
                "status": "ATTENDANCE RECORDED",
            },
        )
    ]
    counters = runtime.performance_metrics.snapshot()["counters"]
    assert counters["recognition.accepted"] == 1
    assert counters["attendance.recorded"] == 1


def test_inactive_user_race_does_not_show_a_false_cooldown_identity() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    observed_at = datetime.now(timezone.utc)
    decision = RecognitionDecision(
        "obs-1", "accepted", "u1", "User", "camera-1", observed_at,
        0.823, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
    )
    runtime.settings = SimpleNamespace(attendance_cooldown_seconds=300)
    runtime.store = SimpleNamespace(
        record_decision=lambda *_: False,
        record_decision_result=lambda *_: (False, "inactive_user"),
    )
    runtime._cloud_wake = SimpleNamespace(set=lambda: None)
    runtime._embedding_catalog = EmbeddingCatalog.from_users(
        [RegisteredUser("u1", "User", "S001", np.array([1], dtype=np.int8), 0.1, "model-v1")],
        "model-v1",
    )
    shown = []
    runtime.preview = SimpleNamespace(show_recognition=lambda *args, **kwargs: shown.append((args, kwargs)))
    runtime.performance_metrics = RollingMetrics()

    runtime._handle_decision(decision, SimpleNamespace(), np.empty((0, 0)))

    assert shown == []
    assert runtime.performance_metrics.snapshot()["counters"]["attendance.inactive_user"] == 1


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


def test_camera_status_tracks_current_active_camera_count_and_failures() -> None:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.store = SimpleNamespace(update_camera=lambda *args: None)
    runtime.cloud = SimpleNamespace(enabled=False)
    runtime._camera_cloud_lock = threading.Lock()
    runtime._online_camera_ids = set()
    runtime.performance_metrics = RollingMetrics()
    camera_1 = SimpleNamespace(camera_id="camera-1", name="Camera 1")
    camera_2 = SimpleNamespace(camera_id="camera-2", name="Camera 2")

    runtime._camera_status(camera_1, "online", None, 0)
    runtime._camera_status(camera_2, "online", None, 0)
    runtime._camera_status(camera_1, "offline", "read failed", 0)

    snapshot = runtime.performance_metrics.snapshot()
    assert snapshot["gauges"]["active_cameras"] == 1
    assert snapshot["gauges"]["camera.camera-1.online"] == 0
    assert snapshot["counters"]["camera.camera-1.failures"] == 1


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
    runtime.frames = SimpleNamespace(get=lambda timeout: candidates.pop(0), qsize=lambda: len(candidates))
    runtime._embedding_catalog = EmbeddingCatalog.from_users([], "model-v1")
    seen = []
    runtime.engine = SimpleNamespace(recognize=lambda image, catalog, *args: seen.append(catalog) or [])
    runtime.inference_lock = __import__("threading").RLock()
    runtime.recognition_enabled = threading.Event()
    runtime.recognition_enabled.set()
    runtime.metrics = RuntimeMetrics()
    runtime.performance_metrics = RollingMetrics()
    runtime.store = SimpleNamespace(active_users=lambda: (_ for _ in ()).throw(AssertionError("per-frame DB read")))
    runtime._inference_loop()
    assert seen == [runtime._embedding_catalog, runtime._embedding_catalog]
    snapshot = runtime.performance_metrics.snapshot()
    assert snapshot["counters"]["inference.processed"] == 2
    assert snapshot["distributions"]["inference_ms"]["samples"] == 2
    assert snapshot["distributions"]["inference_lock_wait_ms"]["samples"] == 2
    assert snapshot["distributions"]["queue_age_ms"]["samples"] == 2
    assert snapshot["distributions"]["full_processing_ms"]["samples"] == 2


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
