import threading

import numpy as np
import pytest
import tkinter as tk

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


def test_preview_keeps_only_latest_selected_camera_frame() -> None:
    preview = PreviewWindow(("camera-1", "camera-2"), 30.0)
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    preview.submit("camera-2", second)
    assert preview.latest_frame() is None
    preview.submit("camera-1", first)
    first[:] = 9

    assert np.array_equal(preview.latest_frame(), np.zeros((2, 2, 3), dtype=np.uint8))
    _, first_sequence = preview.latest_sample()
    _, same_sequence = preview.latest_sample()
    preview.submit("camera-1", second)
    _, second_sequence = preview.latest_sample()

    assert first_sequence == same_sequence
    assert second_sequence == first_sequence + 1


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

    class Label:
        def __init__(self, root):
            pass

        def pack(self):
            pass

    monkeypatch.setattr("pi4_attendance.preview.tk.Tk", Root)
    monkeypatch.setattr("pi4_attendance.preview.tk.Label", Label)

    preview.run(stop)

    assert calls == ["open", "close"]


def test_preview_without_desktop_has_clear_error(monkeypatch) -> None:
    preview = PreviewWindow(("camera-1",), 30.0)
    monkeypatch.setattr("pi4_attendance.preview.tk.Tk", lambda: (_ for _ in ()).throw(tk.TclError("no display")))

    with pytest.raises(RuntimeError, match="Pi desktop"):
        preview.run(threading.Event())
