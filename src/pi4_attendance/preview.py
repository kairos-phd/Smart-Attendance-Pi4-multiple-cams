from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecognitionDisplay:
    name: str
    student_id: str
    similarity: float
    timestamp: str
    status: str
    expires_at: float


class PreviewWindow:
    def __init__(
        self,
        camera_ids: tuple[str, ...],
        fps: float,
        on_enroll: Callable[[str, str, list[np.ndarray], str], dict] | None = None,
    ) -> None:
        self.camera_ids = camera_ids
        self.fps = fps
        self._on_enroll = on_enroll
        self.recognition_enabled = threading.Event()
        self._selected = 0
        self._samples: dict[str, tuple[np.ndarray | None, int]] = {
            camera_id: (None, 0) for camera_id in camera_ids
        }
        self._display_samples: dict[str, tuple[np.ndarray | None, int]] = {
            camera_id: (None, 0) for camera_id in camera_ids
        }
        self._enrollment_name: str | None = None
        self._enrollment_student_id: str | None = None
        self._enrollment_actor: str | None = None
        self._enrollment_frames: list[np.ndarray] = []
        self._last_enrollment_sequences: dict[str, int] = {}
        self._enrollment_busy = False
        self._enrollment_status = ""
        self._recognition_by_camera: dict[str, RecognitionDisplay] = {}
        self._lock = threading.Lock()

    @property
    def selected_camera_id(self) -> str:
        with self._lock:
            return self.camera_ids[self._selected]

    def is_selected(self, camera_id: str) -> bool:
        return camera_id == self.selected_camera_id

    def select_camera(self, camera_id: str) -> str:
        try:
            index = self.camera_ids.index(camera_id)
        except ValueError as exc:
            raise ValueError(f"Unknown camera: {camera_id}") from exc
        with self._lock:
            self._selected = index
        return camera_id

    def set_recognition(self, enabled: bool) -> bool:
        if enabled:
            self.recognition_enabled.set()
        else:
            self.recognition_enabled.clear()
        return self.recognition_enabled.is_set()

    def submit(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            if camera_id not in self._samples:
                raise ValueError(f"Unknown camera: {camera_id}")
            _, sequence = self._samples[camera_id]
            copied = frame.copy()
            self._samples[camera_id] = (copied, sequence + 1)
            _, display_sequence = self._display_samples[camera_id]
            self._display_samples[camera_id] = (copied, display_sequence + 1)

    def submit_overlay(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            if camera_id not in self._display_samples:
                raise ValueError(f"Unknown camera: {camera_id}")
            _, sequence = self._display_samples[camera_id]
            self._display_samples[camera_id] = (frame.copy(), sequence + 1)

    def latest_display_sample(self, camera_id: str) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if camera_id not in self._display_samples:
                raise ValueError(f"Unknown camera: {camera_id}")
            frame, sequence = self._display_samples[camera_id]
            return (None if frame is None else frame.copy(), sequence)

    def latest_frame(self) -> np.ndarray | None:
        return self.latest_sample()[0]

    def latest_sample(self, camera_id: str | None = None) -> tuple[np.ndarray | None, int]:
        with self._lock:
            camera_id = self.camera_ids[self._selected] if camera_id is None else camera_id
            if camera_id not in self._samples:
                raise ValueError(f"Unknown camera: {camera_id}")
            frame, sequence = self._samples[camera_id]
            return (None if frame is None else frame.copy(), sequence)

    def status(self) -> dict:
        with self._lock:
            enrollment = None
            if self._enrollment_name is not None and self._enrollment_student_id is not None:
                enrollment = {
                    "name": self._enrollment_name,
                    "student_id": self._enrollment_student_id,
                    "captured": len(self._enrollment_frames),
                    "actor": self._enrollment_actor,
                }
            return {
                "camera_ids": list(self.camera_ids),
                "selected_camera_id": self.camera_ids[self._selected],
                "recognition_enabled": self.recognition_enabled.is_set(),
                "enrollment": enrollment,
                "enrollment_busy": self._enrollment_busy,
                "enrollment_status": self._enrollment_status,
            }

    def show_recognition(
        self,
        camera_id: str,
        *,
        name: str,
        student_id: str,
        similarity: float,
        timestamp: str,
        status: str,
    ) -> None:
        result = RecognitionDisplay(
            name,
            student_id,
            similarity,
            timestamp,
            status,
            time.monotonic() + 3.0,
        )
        with self._lock:
            self._recognition_by_camera[camera_id] = result

    @property
    def latest_recognition(self) -> RecognitionDisplay | None:
        now = time.monotonic()
        with self._lock:
            result = self._recognition_by_camera.get(self.camera_ids[self._selected])
            return result if result is not None and result.expires_at > now else None

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

    def begin_enrollment(self, name: str, student_id: str, actor: str = "hdmi-preview") -> bool:
        name = name.strip()
        student_id = student_id.strip()
        actor = actor.strip()
        if not name or not student_id or not actor or len(name) > 120 or len(student_id) > 80:
            with self._lock:
                self._enrollment_status = "Name and student ID are required (max 120/80)"
            return False
        with self._lock:
            if self._enrollment_busy or self._enrollment_name is not None:
                return False
            self._enrollment_name = name
            self._enrollment_student_id = student_id
            self._enrollment_actor = actor
            self._enrollment_frames = []
            self._last_enrollment_sequences = {
                camera_id: sequence for camera_id, (_, sequence) in self._samples.items()
            }
            self._enrollment_status = "Capture 3 to 8 images"
        return True

    def capture_enrollment_frame(self) -> bool:
        with self._lock:
            camera_id = self.camera_ids[self._selected]
            frame, sequence = self._samples[camera_id]
            if self._enrollment_name is None or self._enrollment_busy or frame is None:
                return False
            if sequence == self._last_enrollment_sequences.get(camera_id):
                self._enrollment_status = "Wait for a new camera frame"
                return False
            if len(self._enrollment_frames) >= 8:
                self._enrollment_status = "Maximum 8 images"
                return False
            self._enrollment_frames.append(frame.copy())
            self._last_enrollment_sequences[camera_id] = sequence
            self._enrollment_status = f"Captured {len(self._enrollment_frames)}/8"
        return True

    def submit_enrollment(self) -> bool:
        with self._lock:
            if (
                self._on_enroll is None
                or self._enrollment_name is None
                or self._enrollment_student_id is None
                or self._enrollment_actor is None
            ):
                return False
            if len(self._enrollment_frames) < 3:
                self._enrollment_status = "Capture at least 3 images"
                return False
            if self._enrollment_busy:
                return False
            name = self._enrollment_name
            student_id = self._enrollment_student_id
            actor = self._enrollment_actor
            frames = [frame.copy() for frame in self._enrollment_frames]
            self._enrollment_busy = True
            self._enrollment_status = "Enrolling..."
        threading.Thread(
            target=self._enroll,
            args=(name, student_id, frames, actor),
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
            self._enrollment_actor = None
            self._enrollment_frames = []
            self._last_enrollment_sequences = {}
            self._enrollment_status = "Enrollment cancelled"
        return True

    def _enroll(self, name: str, student_id: str, frames: list[np.ndarray], actor: str) -> None:
        try:
            assert self._on_enroll is not None
            self._on_enroll(name, student_id, frames, actor)
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
            self._enrollment_actor = None
            self._enrollment_frames = []
            self._last_enrollment_sequences = {}
            self._enrollment_busy = False
            self._enrollment_status = "Enrollment complete"

    def handle_key(self, key: int) -> bool:
        if key in (ord("a"), ord("A")):
            self.set_recognition(not self.recognition_enabled.is_set())
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.camera_ids):
                self.select_camera(self.camera_ids[index])
        return key not in (ord("q"), ord("Q"), 27)

    def _handle_ui_key(self, root: object, stop_event: threading.Event, event) -> None:
        if event.keysym == "Return":
            self.submit_enrollment()
            return
        key = ord(event.char) if event.char else 27 if event.keysym == "Escape" else -1
        if key in (ord("e"), ord("E")):
            if self.enrollment_progress is not None or self.enrollment_busy:
                return
            from tkinter import simpledialog

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
        import tkinter as tk
        from PIL import Image, ImageTk

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("HDMI preview must run from a Terminal in the active Pi desktop session") from exc
        root.title("SIC Attendance Preview")
        label = tk.Label(root)
        label.pack()
        status_label = tk.Label(
            root,
            bg="#111111",
            fg="#ffffff",
            font=("TkDefaultFont", 14, "bold"),
            justify="left",
            anchor="w",
            padx=12,
            pady=8,
        )
        status_label.pack(fill=tk.X)
        root.bind("<Key>", lambda event: self._handle_ui_key(root, stop_event, event))
        root.protocol("WM_DELETE_WINDOW", stop_event.set)
        frames = 0
        measured_fps = 0.0
        measured_at = time.monotonic()
        last_sequence = -1
        try:
            while not stop_event.is_set():
                frame, sequence = self.latest_display_sample(self.selected_camera_id)
                now = time.monotonic()
                if sequence != last_sequence:
                    frames += 1
                    last_sequence = sequence
                elapsed = now - measured_at
                if elapsed >= 1.0:
                    measured_fps = frames / elapsed
                    frames = 0
                    measured_at = now
                state = "ON" if self.recognition_enabled.is_set() else "OFF"
                progress = self.enrollment_progress
                if progress is None:
                    enrollment = "Enroll: [E] start"
                else:
                    enrollment = f"Enroll {progress[0]}: {progress[2]}/8 [C] [Enter] [X]"
                lines = [
                    f"Camera: {self.selected_camera_id}  |  {measured_fps:.1f} FPS",
                    f"Recognition: {state} [A]  |  {enrollment}",
                ]
                if self.enrollment_status:
                    lines.append(self.enrollment_status)
                result = self.latest_recognition
                if result is not None:
                    lines.extend(
                        (
                            f"{result.name}  |  ID: {result.student_id}",
                            f"Similarity: {result.similarity:.3f}  |  {result.timestamp}  |  {result.status}",
                        )
                    )
                status_label.configure(text="\n".join(lines))
                if frame is not None:
                    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    label.configure(image=image)
                    label.image = image
                else:
                    label.configure(image="")
                    label.image = None
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
