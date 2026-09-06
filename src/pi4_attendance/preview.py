from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

LOG = logging.getLogger(__name__)


class PreviewWindow:
    def __init__(
        self,
        camera_ids: tuple[str, ...],
        fps: float,
        on_enroll: Callable[[str, str, list[np.ndarray]], dict] | None = None,
    ) -> None:
        self.camera_ids = camera_ids
        self.fps = fps
        self._on_enroll = on_enroll
        self.recognition_enabled = threading.Event()
        self._selected = 0
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._enrollment_name: str | None = None
        self._enrollment_student_id: str | None = None
        self._enrollment_frames: list[np.ndarray] = []
        self._last_enrollment_sequence = -1
        self._enrollment_busy = False
        self._enrollment_status = ""
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

    @property
    def enrollment_progress(self) -> tuple[str, str, int] | None:
        with self._lock:
            if self._enrollment_name is None or self._enrollment_student_id is None:
                return None
            return self._enrollment_name, self._enrollment_student_id, len(self._enrollment_frames)

    @property
    def enrollment_frames(self) -> list[np.ndarray]:
        with self._lock:
            return [frame.copy() for frame in self._enrollment_frames]

    @property
    def enrollment_busy(self) -> bool:
        with self._lock:
            return self._enrollment_busy

    @property
    def enrollment_status(self) -> str:
        with self._lock:
            return self._enrollment_status

    def begin_enrollment(self, name: str, student_id: str) -> bool:
        name = name.strip()
        student_id = student_id.strip()
        if not name or not student_id or len(name) > 120 or len(student_id) > 80:
            with self._lock:
                self._enrollment_status = "Name and student ID are required (max 120/80)"
            return False
        with self._lock:
            if self._enrollment_busy or self._enrollment_name is not None:
                return False
            self._enrollment_name = name
            self._enrollment_student_id = student_id
            self._enrollment_frames = []
            self._last_enrollment_sequence = self._sequence
            self._enrollment_status = "Capture 3 to 8 images"
        return True

    def capture_enrollment_frame(self) -> bool:
        with self._lock:
            if self._enrollment_name is None or self._enrollment_busy or self._frame is None:
                return False
            if self._sequence == self._last_enrollment_sequence:
                self._enrollment_status = "Wait for a new camera frame"
                return False
            if len(self._enrollment_frames) >= 8:
                self._enrollment_status = "Maximum 8 images"
                return False
            self._enrollment_frames.append(self._frame.copy())
            self._last_enrollment_sequence = self._sequence
            self._enrollment_status = f"Captured {len(self._enrollment_frames)}/8"
        return True

    def submit_enrollment(self) -> bool:
        with self._lock:
            if self._on_enroll is None or self._enrollment_name is None or self._enrollment_student_id is None:
                return False
            if len(self._enrollment_frames) < 3:
                self._enrollment_status = "Capture at least 3 images"
                return False
            if self._enrollment_busy:
                return False
            name = self._enrollment_name
            student_id = self._enrollment_student_id
            frames = [frame.copy() for frame in self._enrollment_frames]
            self._enrollment_busy = True
            self._enrollment_status = "Enrolling..."
        threading.Thread(
            target=self._enroll,
            args=(name, student_id, frames),
            name="pi4-preview-enrollment",
            daemon=True,
        ).start()
        return True

    def cancel_enrollment(self) -> bool:
        with self._lock:
            if self._enrollment_name is None or self._enrollment_busy:
                return False
            self._enrollment_name = None
            self._enrollment_student_id = None
            self._enrollment_frames = []
            self._last_enrollment_sequence = -1
            self._enrollment_status = "Enrollment cancelled"
        return True

    def _enroll(self, name: str, student_id: str, frames: list[np.ndarray]) -> None:
        try:
            assert self._on_enroll is not None
            self._on_enroll(name, student_id, frames)
        except ValueError as exc:
            with self._lock:
                self._enrollment_busy = False
                self._enrollment_status = f"Enrollment failed: {exc}"
            return
        except Exception as exc:
            if "UNIQUE constraint" in str(exc) and "student_id" in str(exc):
                status = "Enrollment failed: Student ID is already registered"
            else:
                LOG.exception("Preview enrollment failed")
                status = "Enrollment failed; check the terminal log"
            with self._lock:
                self._enrollment_busy = False
                self._enrollment_status = status
            return
        with self._lock:
            self._enrollment_name = None
            self._enrollment_student_id = None
            self._enrollment_frames = []
            self._last_enrollment_sequence = -1
            self._enrollment_busy = False
            self._enrollment_status = "Enrollment complete"

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

    def _handle_ui_key(self, root: tk.Tk, stop_event: threading.Event, event) -> None:
        if event.keysym == "Return":
            self.submit_enrollment()
            return
        key = ord(event.char) if event.char else 27 if event.keysym == "Escape" else -1
        if key in (ord("e"), ord("E")):
            if self.enrollment_progress is not None or self.enrollment_busy:
                return
            name = simpledialog.askstring("Enroll user", "Name", parent=root)
            if name is None:
                return
            student_id = simpledialog.askstring("Enroll user", "Student ID", parent=root)
            if student_id is not None:
                self.begin_enrollment(name, student_id)
        elif key in (ord("c"), ord("C")):
            self.capture_enrollment_frame()
        elif key in (ord("x"), ord("X")):
            self.cancel_enrollment()
        elif not self.handle_key(key):
            stop_event.set()

    def run(self, stop_event: threading.Event) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("HDMI preview must run from a Terminal in the active Pi desktop session") from exc
        root.title("SIC Attendance Preview")
        label = tk.Label(root)
        label.pack()
        root.bind("<Key>", lambda event: self._handle_ui_key(root, stop_event, event))
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
                    progress = self.enrollment_progress
                    if progress is None:
                        enrollment = "Enroll: [E] start"
                    else:
                        enrollment = f"Enroll {progress[0]}: {progress[2]}/8 [C] [Enter] [X]"
                    cv2.putText(frame, enrollment, (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    cv2.putText(frame, self.enrollment_status, (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
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
