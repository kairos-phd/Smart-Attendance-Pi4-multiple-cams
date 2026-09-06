from __future__ import annotations

import threading
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class PreviewWindow:
    def __init__(self, camera_ids: tuple[str, ...], fps: float) -> None:
        self.camera_ids = camera_ids
        self.fps = fps
        self.recognition_enabled = threading.Event()
        self._selected = 0
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def selected_camera_id(self) -> str:
        with self._lock:
            return self.camera_ids[self._selected]

    def is_selected(self, camera_id: str) -> bool:
        return camera_id == self.selected_camera_id

    def submit(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            if camera_id == self.camera_ids[self._selected]:
                self._frame = frame.copy()
                self._sequence += 1

    def latest_frame(self) -> np.ndarray | None:
        return self.latest_sample()[0]

    def latest_sample(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), self._sequence)

    def handle_key(self, key: int) -> bool:
        if key in (ord("a"), ord("A")):
            if self.recognition_enabled.is_set():
                self.recognition_enabled.clear()
            else:
                self.recognition_enabled.set()
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.camera_ids):
                with self._lock:
                    self._selected = index
                    self._frame = None
        return key not in (ord("q"), ord("Q"), 27)

    def run(self, stop_event: threading.Event) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("HDMI preview must run from a Terminal in the active Pi desktop session") from exc
        root.title("SIC Attendance Preview")
        label = tk.Label(root)
        label.pack()
        root.bind("<Key>", lambda event: stop_event.set() if not self.handle_key(27 if event.keysym == "Escape" else ord(event.char) if event.char else -1) else None)
        root.protocol("WM_DELETE_WINDOW", stop_event.set)
        frames = 0
        measured_fps = 0.0
        measured_at = time.monotonic()
        last_sequence = -1
        try:
            while not stop_event.is_set():
                frame, sequence = self.latest_sample()
                now = time.monotonic()
                if sequence != last_sequence:
                    frames += 1
                    last_sequence = sequence
                elapsed = now - measured_at
                if elapsed >= 1.0:
                    measured_fps = frames / elapsed
                    frames = 0
                    measured_at = now
                if frame is not None:
                    state = "ON" if self.recognition_enabled.is_set() else "OFF"
                    cv2.putText(frame, f"Camera: {self.selected_camera_id}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Camera: {measured_fps:.1f} FPS", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Recognition: {state}  [A]", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    label.configure(image=image)
                    label.image = image
                root.update_idletasks()
                root.update()
                stop_event.wait(1.0 / self.fps)
        except tk.TclError:
            stop_event.set()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass
