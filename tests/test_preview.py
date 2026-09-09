import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tkinter as tk
from PIL import ImageTk

from pi4_attendance.preview import PreviewWindow


def test_preview_keys_toggle_recognition_select_camera_and_quit() -> None:
    preview = PreviewWindow(("camera-1", "camera-2", "camera-3"), 30.0)

    assert preview.selected_camera_id == "camera-1"
    assert not preview.recognition_enabled.is_set()
    assert preview.handle_key(ord("a"))
    assert preview.recognition_enabled.is_set()
    assert preview.handle_key(ord("A"))
    assert not preview.recognition_enabled.is_set()
    assert preview.handle_key(ord("2"))
    assert preview.selected_camera_id == "camera-2"
    assert not preview.is_selected("camera-1")
    assert preview.is_selected("camera-2")
    assert not preview.handle_key(ord("q"))
    assert not preview.handle_key(27)


def test_preview_keeps_latest_copied_frame_and_sequence_per_camera() -> None:
    preview = PreviewWindow(("camera-1", "camera-2"), 30.0)
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    preview.submit("camera-2", second)
    assert preview.latest_frame() is None
    camera_2_frame, camera_2_sequence = preview.latest_sample("camera-2")
    second[:] = 8
    assert np.array_equal(camera_2_frame, np.ones((2, 2, 3), dtype=np.uint8))
    camera_2_frame[:] = 7
    assert np.array_equal(preview.latest_sample("camera-2")[0], np.ones((2, 2, 3), dtype=np.uint8))
    assert camera_2_sequence == 1
    preview.submit("camera-1", first)
    first[:] = 9

    assert np.array_equal(preview.latest_frame(), np.zeros((2, 2, 3), dtype=np.uint8))
    _, first_sequence = preview.latest_sample()
    _, same_sequence = preview.latest_sample()
    preview.submit("camera-1", second)
    _, second_sequence = preview.latest_sample()

    assert first_sequence == same_sequence
    assert second_sequence == first_sequence + 1
    assert preview.latest_sample("camera-2")[1] == camera_2_sequence


def test_preview_selects_known_camera_and_sets_recognition_exactly() -> None:
    preview = PreviewWindow(("camera-1", "camera-2"), 30.0)

    assert preview.set_recognition(True) is True
    assert preview.set_recognition(True) is True
    assert preview.set_recognition(False) is False
    assert preview.select_camera("camera-2") == "camera-2"
    with pytest.raises(ValueError, match="Unknown camera"):
        preview.select_camera("camera-3")

    assert preview.status() == {
        "camera_ids": ["camera-1", "camera-2"],
        "selected_camera_id": "camera-2",
        "recognition_enabled": False,
        "enrollment": None,
        "enrollment_busy": False,
        "enrollment_status": "",
    }


def test_preview_recognition_is_scoped_to_camera_and_expires(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("pi4_attendance.preview.time.monotonic", lambda: now[0])
    preview = PreviewWindow(("camera-1", "camera-2"), 30.0)

    preview.show_recognition(
        "camera-2",
        name="Nguyễn Văn A",
        student_id="S001",
        similarity=0.823,
        timestamp="10:42:18",
        status="ATTENDANCE RECORDED",
    )

    assert preview.latest_recognition is None
    preview.handle_key(ord("2"))
    result = preview.latest_recognition
    assert result is not None
    assert (result.name, result.student_id, result.similarity) == ("Nguyễn Văn A", "S001", 0.823)
    assert (result.timestamp, result.status) == ("10:42:18", "ATTENDANCE RECORDED")

    now[0] = 103.01
    assert preview.latest_recognition is None


def test_preview_run_closes_window_when_stop_is_requested(monkeypatch) -> None:
    preview = PreviewWindow(("camera-1",), 30.0)
    stop = threading.Event()
    calls: list[str] = []

    class Root:
        def title(self, value):
            calls.append("open")

        def bind(self, *args):
            pass

        def protocol(self, *args):
            pass

        def update_idletasks(self):
            pass

        def update(self):
            stop.set()

        def destroy(self):
            calls.append("close")

    labels = []

    class Label:
        def __init__(self, root, **options):
            self.options = options
            self.configured = {}
            labels.append(self)

        def pack(self, **options):
            pass

        def configure(self, **options):
            self.configured.update(options)

    monkeypatch.setattr(tk, "Tk", Root)
    monkeypatch.setattr(tk, "Label", Label)

    preview.run(stop)

    assert calls == ["open", "close"]
    assert labels[1].options["bg"] == "#111111"
    assert labels[1].options["fg"] == "#ffffff"


def test_switching_to_camera_without_a_frame_clears_stale_image_and_identity(monkeypatch) -> None:
    preview = PreviewWindow(("camera-1", "camera-2"), 30.0)
    preview.submit("camera-1", np.zeros((2, 2, 3), dtype=np.uint8))
    preview.show_recognition(
        "camera-1",
        name="Nguyễn Văn A",
        student_id="S001",
        similarity=0.823,
        timestamp="10:42:18",
        status="ATTENDANCE RECORDED",
    )
    stop = threading.Event()
    updates = 0

    class Root:
        def title(self, value):
            pass

        def bind(self, *args):
            pass

        def protocol(self, *args):
            pass

        def update_idletasks(self):
            pass

        def update(self):
            nonlocal updates
            updates += 1
            if updates == 1:
                preview.handle_key(ord("2"))
            else:
                stop.set()

        def destroy(self):
            pass

    labels = []

    class Label:
        def __init__(self, root, **options):
            self.configured = {}
            labels.append(self)

        def pack(self, **options):
            pass

        def configure(self, **options):
            self.configured.update(options)

    monkeypatch.setattr(tk, "Tk", Root)
    monkeypatch.setattr(tk, "Label", Label)
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda image: object())

    preview.run(stop)

    assert labels[0].configured["image"] == ""
    assert "Camera: camera-2" in labels[1].configured["text"]
    assert "Nguyễn Văn A" not in labels[1].configured["text"]


def test_preview_without_desktop_has_clear_error(monkeypatch) -> None:
    preview = PreviewWindow(("camera-1",), 30.0)
    monkeypatch.setattr(tk, "Tk", lambda: (_ for _ in ()).throw(tk.TclError("no display")))

    with pytest.raises(RuntimeError, match="Pi desktop"):
        preview.run(threading.Event())


def test_preview_captures_three_to_eight_enrollment_frames() -> None:
    preview = PreviewWindow(("camera-1",), 30.0, lambda *_: {})

    preview.submit("camera-1", np.full((2, 2, 3), 99, dtype=np.uint8))
    preview.begin_enrollment(" Dang ", " S001 ")
    assert not preview.capture_enrollment_frame()
    for value in range(8):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        assert preview.capture_enrollment_frame()
        assert not preview.capture_enrollment_frame()
    preview.submit("camera-1", np.full((2, 2, 3), 9, dtype=np.uint8))
    assert not preview.capture_enrollment_frame()
    assert preview.enrollment_status == "Maximum 8 images"

    assert preview.enrollment_progress == ("Dang", "S001", 8)
    assert np.array_equal(preview.enrollment_frames[0], np.zeros((2, 2, 3), dtype=np.uint8))
    assert not preview.begin_enrollment("Other", "S002")
    assert preview.enrollment_progress == ("Dang", "S001", 8)


def test_preview_validates_enrollment_identity() -> None:
    assert not PreviewWindow(("camera-1",), 30.0).begin_enrollment("", "S001")
    assert not PreviewWindow(("camera-1",), 30.0).begin_enrollment("x" * 121, "S002")
    assert not PreviewWindow(("camera-1",), 30.0).begin_enrollment("User", "x" * 81)


def test_preview_submits_enrollment_in_background_with_actor() -> None:
    started = threading.Event()
    release = threading.Event()
    received = []

    def enroll(name, student_id, images, actor):
        started.set()
        assert release.wait(1)
        received.append((name, student_id, images, actor))
        return {"status": "active"}

    preview = PreviewWindow(("camera-1",), 30.0, enroll)
    preview.begin_enrollment("Dang", "S001", "local-console")
    for value in range(3):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        assert preview.capture_enrollment_frame()

    assert preview.submit_enrollment()
    assert started.wait(1)
    assert preview.enrollment_busy
    assert not preview.submit_enrollment()
    release.set()
    deadline = time.monotonic() + 1
    while preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert received[0][:2] == ("Dang", "S001")
    assert len(received[0][2]) == 3
    assert received[0][3] == "local-console"
    assert preview.enrollment_progress is None
    assert preview.enrollment_status == "Enrollment complete"


def test_preview_requires_three_frames_and_can_cancel_enrollment() -> None:
    preview = PreviewWindow(("camera-1",), 30.0, lambda *_: {})
    preview.begin_enrollment("Dang", "S001")
    preview.submit("camera-1", np.zeros((2, 2, 3), dtype=np.uint8))
    assert preview.capture_enrollment_frame()

    assert not preview.submit_enrollment()
    assert "3 images" in preview.enrollment_status
    assert preview.cancel_enrollment()
    assert preview.enrollment_progress is None


def test_preview_enrollment_failure_can_be_cancelled() -> None:
    attempts = 0

    def fail(*_):
        nonlocal attempts
        attempts += 1
        raise ValueError("Face detection confidence is too low")

    preview = PreviewWindow(("camera-1",), 30.0, fail)
    preview.begin_enrollment("Dang", "S001")
    for value in range(3):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        preview.capture_enrollment_frame()

    assert preview.submit_enrollment()
    deadline = time.monotonic() + 1
    while preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert attempts == 1
    assert preview.enrollment_progress == ("Dang", "S001", 3)
    assert preview.enrollment_status == "Enrollment failed: Face detection confidence is too low"
    assert preview.cancel_enrollment()


def test_preview_hides_internal_enrollment_errors() -> None:
    def fail(*_):
        raise RuntimeError("database secret detail")

    preview = PreviewWindow(("camera-1",), 30.0, fail)
    preview.begin_enrollment("Dang", "S001")
    for value in range(3):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        preview.capture_enrollment_frame()

    preview.submit_enrollment()
    deadline = time.monotonic() + 1
    while preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert preview.enrollment_status == "Enrollment failed; check the terminal log"


def test_preview_reports_duplicate_student_id() -> None:
    def fail(*_):
        raise RuntimeError("UNIQUE constraint failed: users.student_id")

    preview = PreviewWindow(("camera-1",), 30.0, fail)
    preview.begin_enrollment("Dang", "S001")
    for value in range(3):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        preview.capture_enrollment_frame()

    preview.submit_enrollment()
    deadline = time.monotonic() + 1
    while preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert preview.enrollment_status == "Enrollment failed: Student ID is already registered"


def test_preview_ui_keys_start_capture_submit_and_cancel(monkeypatch) -> None:
    enrolled = threading.Event()
    preview = PreviewWindow(("camera-1",), 30.0, lambda *_: enrolled.set())
    responses = iter(("Dang", "S001", "Other", "S002"))
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *_, **__: next(responses))
    root = object()
    stop = threading.Event()

    preview._handle_ui_key(root, stop, SimpleNamespace(keysym="e", char="e"))
    for value in range(3):
        preview.submit("camera-1", np.full((2, 2, 3), value, dtype=np.uint8))
        preview._handle_ui_key(root, stop, SimpleNamespace(keysym="c", char="c"))
    preview._handle_ui_key(root, stop, SimpleNamespace(keysym="Return", char="\r"))
    assert enrolled.wait(1)

    deadline = time.monotonic() + 1
    while preview.enrollment_busy and time.monotonic() < deadline:
        time.sleep(0.01)
    preview._handle_ui_key(root, stop, SimpleNamespace(keysym="e", char="e"))
    preview._handle_ui_key(root, stop, SimpleNamespace(keysym="x", char="x"))
    assert preview.enrollment_progress is None
    assert not stop.is_set()
