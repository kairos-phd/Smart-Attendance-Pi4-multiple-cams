from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np

from pi4_attendance.models import RecognitionDecision
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.storage import PiStore


def decision(observation_id: str = "obs-1") -> RecognitionDecision:
    observed_at = datetime(2026, 8, 27, 8, 30, tzinfo=timezone.utc)
    return RecognitionDecision(
        observation_id,
        "accepted",
        "u1",
        "Test User",
        "camera-1",
        observed_at,
        0.81,
        0.12,
        0.93,
        0.74,
        (1, 2, 30, 40),
        "model-v1",
        "thresholds_passed",
    )


def wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


class FakeCloud:
    enabled = True

    def __init__(self, failures: int = 0, block: threading.Event | None = None) -> None:
        self.failures = failures
        self.block = block
        self.attendance_ids: list[str] = []
        self.camera_updates: list[tuple[str, str]] = []
        self.started = threading.Event()

    def attendance(self, item: RecognitionDecision) -> None:
        self.attendance_ids.append(item.observation_id)
        self.started.set()
        if self.block is not None:
            self.block.wait(1)
        if len(self.attendance_ids) <= self.failures:
            raise OSError("cloud unavailable")

    def camera_status(self, camera, status: str, error: str | None) -> None:
        self.camera_updates.append((camera.camera_id, status))
        self.started.set()
        if self.block is not None:
            self.block.wait(1)


def make_runtime(store: PiStore, cloud: FakeCloud) -> AttendanceRuntime:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(attendance_cooldown_seconds=300, save_unknown_images=False)
    runtime.store = store
    runtime.cloud = cloud
    runtime.stop_event = threading.Event()
    runtime._cloud_wake = threading.Event()
    runtime._camera_cloud_lock = threading.Lock()
    runtime._pending_camera_status = {}
    runtime._cloud_poll_seconds = 0.01
    runtime._cloud_thread = None
    runtime._api_server = None
    runtime._runtime_state = "running"
    return runtime


def start_worker(runtime: AttendanceRuntime) -> threading.Thread:
    worker = threading.Thread(target=runtime._cloud_loop)
    runtime._cloud_thread = worker
    worker.start()
    return worker


def add_user(store: PiStore) -> None:
    store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")


def test_successful_upload_marks_local_attendance_synced(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    assert store.record_decision(decision(), 300)
    runtime = make_runtime(store, FakeCloud())
    worker = start_worker(runtime)

    wait_until(lambda: store.recent_attendance()[0]["sync_state"] == "synced")

    runtime.stop()
    worker.join(1)
    assert runtime.cloud.attendance_ids == ["obs-1"]
    assert not worker.is_alive()
    store.close()


def test_failed_upload_stays_pending_and_retries_until_success(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    assert store.record_decision(decision(), 300)
    runtime = make_runtime(store, FakeCloud(failures=2))
    worker = start_worker(runtime)

    wait_until(lambda: store.recent_attendance()[0]["sync_state"] == "synced")

    runtime.stop()
    worker.join(1)
    assert runtime.cloud.attendance_ids == ["obs-1", "obs-1", "obs-1"]
    store.close()


def test_repeated_failures_leave_attendance_pending(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    assert store.record_decision(decision(), 300)
    runtime = make_runtime(store, FakeCloud(failures=100))
    worker = start_worker(runtime)

    wait_until(lambda: len(runtime.cloud.attendance_ids) >= 3)

    assert store.recent_attendance()[0]["sync_state"] == "pending"
    runtime.stop()
    worker.join(1)
    assert not worker.is_alive()
    store.close()


def test_pending_attendance_survives_store_restart(tmp_path: Path) -> None:
    first_store = PiStore(tmp_path)
    add_user(first_store)
    assert first_store.record_decision(decision(), 300)
    first_store.close()

    restarted_store = PiStore(tmp_path)
    assert restarted_store.recent_attendance()[0]["sync_state"] == "pending"
    runtime = make_runtime(restarted_store, FakeCloud())
    worker = start_worker(runtime)
    wait_until(lambda: restarted_store.recent_attendance()[0]["sync_state"] == "synced")

    runtime.stop()
    worker.join(1)
    assert runtime.cloud.attendance_ids == ["obs-1"]
    restarted_store.close()


def test_uncertain_failure_retries_with_same_id(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    assert store.record_decision(decision("stable-id"), 300)
    runtime = make_runtime(store, FakeCloud(failures=1))
    worker = start_worker(runtime)
    wait_until(lambda: store.recent_attendance()[0]["sync_state"] == "synced")

    runtime.stop()
    worker.join(1)
    assert runtime.cloud.attendance_ids == ["stable-id", "stable-id"]
    store.close()


def test_slow_cloud_does_not_block_decision_or_camera_status(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    release_cloud = threading.Event()
    runtime = make_runtime(store, FakeCloud(block=release_cloud))
    worker = start_worker(runtime)

    started = time.monotonic()
    runtime._handle_decision(decision(), SimpleNamespace(), np.empty((0, 0)))
    decision_elapsed = time.monotonic() - started
    assert runtime.cloud.started.wait(1)

    camera = SimpleNamespace(camera_id="camera-1", name="Front")
    started = time.monotonic()
    runtime._camera_status(camera, "online", None, 0)
    camera_elapsed = time.monotonic() - started

    assert decision_elapsed < 0.1
    assert camera_elapsed < 0.1
    assert store.recent_attendance()[0]["sync_state"] == "pending"

    release_cloud.set()
    wait_until(lambda: store.recent_attendance()[0]["sync_state"] == "synced")
    runtime.stop()
    worker.join(1)
    store.close()


def test_camera_status_is_coalesced_before_background_upload(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    runtime = make_runtime(store, FakeCloud())
    camera = SimpleNamespace(camera_id="camera-1", name="Front")
    runtime._camera_status(camera, "offline", "lost", 0)
    runtime._camera_status(camera, "online", None, 0)

    worker = start_worker(runtime)
    wait_until(lambda: runtime.cloud.camera_updates == [("camera-1", "online")])

    runtime.stop()
    worker.join(1)
    assert not worker.is_alive()
    store.close()


def test_idle_cloud_worker_stops_promptly(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    runtime = make_runtime(store, FakeCloud())
    runtime._cloud_poll_seconds = 60
    worker = start_worker(runtime)
    time.sleep(0.01)

    runtime.stop()
    worker.join(0.2)

    assert not worker.is_alive()
    store.close()


def test_stop_finishes_current_upload_without_starting_another(tmp_path: Path) -> None:
    store = PiStore(tmp_path)
    add_user(store)
    first = decision("obs-1")
    later = first.decided_at + timedelta(seconds=301)
    assert store.record_decision(first, 300)
    assert store.record_decision(replace(first, observation_id="obs-2", observed_at=later, decided_at=later), 300)
    release_cloud = threading.Event()
    runtime = make_runtime(store, FakeCloud(block=release_cloud))
    worker = start_worker(runtime)
    assert runtime.cloud.started.wait(1)

    runtime.stop()
    release_cloud.set()
    worker.join(0.5)

    assert not worker.is_alive()
    assert runtime.cloud.attendance_ids == ["obs-1"]
    assert [row["sync_state"] for row in store.recent_attendance()] == ["pending", "synced"]
    store.close()
