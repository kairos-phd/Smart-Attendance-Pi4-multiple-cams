import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from pi4_attendance.api import create_app
from pi4_attendance.face_engine import FaceEngine
from test_enrollment_atomic import enrollment_runtime


def test_health_responds_while_enrollment_is_waiting():
    entered, release = threading.Event(), threading.Event()
    runtime = SimpleNamespace(
        settings=SimpleNamespace(api_cors_origins=[], api_auth_mode="shared_token", admin_shared_token="test"),
        health=lambda: {"status": "ok"},
    )

    def enroll(*args):
        entered.set()
        release.wait(2)
        return {"status": "active"}

    runtime.enroll = enroll
    _, png = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/api/users", data={"name": "User", "student_id": "S1"},
                files=[("face_images", ("image.png", png.tobytes(), "image/png"))] * 3,
                headers={"Authorization": "Bearer test"}))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                health = await asyncio.wait_for(client.get("/health"), 1)
                assert health.status_code == 200
                assert not request.done(), "Enrollment blocked the event loop until completion"
            finally:
                release.set()
                await request

    asyncio.run(exercise())


def test_camera_model_access_between_enrollment_images(tmp_path, monkeypatch):
    runtime = enrollment_runtime(tmp_path)
    engine = FaceEngine(tmp_path, "model-v1", 320, 0.45, 0.06)
    runtime.engine = engine
    first_image, release = threading.Event(), threading.Event()
    face = SimpleNamespace(det_score=0.99, bbox=(0, 0, 8, 8), normed_embedding=np.array([1, 0], dtype=np.float32))
    engine._app = SimpleNamespace(get=lambda image: [face])

    def quality(image, bbox):
        if threading.current_thread().name.startswith("enrollment") and not first_image.is_set():
            first_image.set()
            assert release.wait(2)
        return 1.0

    monkeypatch.setattr("pi4_attendance.face_engine._face_quality", quality)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with ThreadPoolExecutor(1, thread_name_prefix="enrollment") as pool:
        request = pool.submit(runtime.enroll, "User", "S1", [image] * 3, "admin")
        try:
            assert first_image.wait(2)
            acquired = runtime.inference_lock.acquire(timeout=0.2)
            assert acquired, "Whole enrollment batch holds the inference lock"
            try:
                assert len(engine.recognize(image, runtime._embedding_catalog, "cam1", 0, datetime.now(timezone.utc))) == 1
            finally:
                runtime.inference_lock.release()
        finally:
            release.set()
        request.result(2)
    runtime.store.close()


def test_overlapping_enrollment_is_rejected_without_model_work(tmp_path):
    runtime = enrollment_runtime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def embedding(images):
        calls.append(1)
        entered.set()
        assert release.wait(2)
        return np.array([1, 0], dtype=np.int8), 0.1

    runtime.engine.enrollment_embedding = embedding
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with ThreadPoolExecutor(1) as pool:
        request = pool.submit(runtime.enroll, "User", "S1", [image], "admin")
        try:
            assert entered.wait(2)
            with pytest.raises(ValueError, match="progress"):
                runtime.enroll("Other", "S2", [image], "admin")
            assert calls == [1]
        finally:
            release.set()
        request.result(2)
    runtime.store.close()


def test_shutdown_waits_for_active_enrollment_before_closing_store(tmp_path):
    runtime = enrollment_runtime(tmp_path)
    runtime._api_server = None
    runtime._cloud_wake = threading.Event()
    runtime._workers = []
    runtime._inference_thread = runtime._retention_thread = runtime._cloud_thread = runtime._api_thread = None
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_close = runtime.store.close
    runtime.start = lambda: None
    runtime.store.close = lambda: (closed.set(), original_close())

    def embedding(images):
        started.set()
        assert release.wait(2)
        return np.array([1, 0], dtype=np.int8), 0.1

    runtime.engine.enrollment_embedding = embedding
    with ThreadPoolExecutor(2) as pool:
        enrollment = pool.submit(runtime.enroll, "User", "S1", [np.zeros((8, 8, 3), dtype=np.uint8)], "admin")
        assert started.wait(2)
        runtime.stop()
        shutdown = pool.submit(runtime.run)
        try:
            assert not closed.wait(0.2), "Store closed while enrollment was active"
        finally:
            release.set()
        with pytest.raises(ValueError, match="stopping"):
            enrollment.result(2)
        shutdown.result(2)
    assert closed.is_set()
    assert list(runtime.store.enrollment_root.iterdir()) == []


def test_model_calls_are_serialized(tmp_path):
    engine = FaceEngine(tmp_path, "model-v1", 320, 0.45, 0.06)
    entered, release, overlap = threading.Event(), threading.Event(), threading.Event()

    def get(image):
        if entered.is_set():
            overlap.set()
        entered.set()
        assert release.wait(2)
        return []

    engine._app = SimpleNamespace(get=get)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(engine._faces, None)
        assert entered.wait(2)
        second = pool.submit(engine._faces, None)
        try:
            assert not overlap.wait(0.2)
        finally:
            release.set()
        assert first.result(2) == second.result(2) == []
