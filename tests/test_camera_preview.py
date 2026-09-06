import threading
from types import SimpleNamespace

import numpy as np
import cv2

from pi4_attendance.camera import CameraWorker
from pi4_attendance.config import CameraConfig


class FrameSink:
    def __init__(self) -> None:
        self.frames = []

    def put(self, frame) -> bool:
        self.frames.append(frame)
        return False


def test_preview_frames_continue_while_recognition_is_disabled_and_sampling_stays_bounded() -> None:
    recognition_enabled = threading.Event()
    output = FrameSink()
    previews = []
    settings = SimpleNamespace(motion_min_area=0, monitor_fps=4.0, camera_trigger_cooldown_seconds=0.8)
    worker = CameraWorker(
        CameraConfig("camera-1", "Camera 1", 0),
        settings,
        output,
        threading.Event(),
        lambda *args: None,
        recognition_enabled=recognition_enabled,
        on_preview_frame=lambda camera_id, frame: previews.append((camera_id, frame.copy())),
        preview_selected=lambda camera_id: True,
        preview_fps=30.0,
    )
    worker.motion = SimpleNamespace(has_activity=lambda frame: True)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    worker._process_frame(frame, 0.0)
    recognition_enabled.set()
    worker._process_frame(frame, 0.01)
    worker._process_frame(frame, 0.10)
    worker._process_frame(frame, 0.26)
    worker._process_frame(frame, 0.82)

    assert len(previews) == 5
    assert [candidate.captured_monotonic for candidate in output.frames] == [0.01, 0.82]


def test_non_preview_worker_keeps_recognition_enabled_by_default() -> None:
    worker = CameraWorker(
        CameraConfig("camera-1", "Camera 1", 0),
        SimpleNamespace(motion_min_area=0),
        FrameSink(),
        threading.Event(),
        lambda *args: None,
    )

    assert worker.recognition_enabled.is_set()


def test_only_selected_camera_requests_preview_fps() -> None:
    selected = "camera-1"
    settings = SimpleNamespace(motion_min_area=0, monitor_fps=4.0)
    worker = CameraWorker(
        CameraConfig("camera-1", "Camera 1", 0),
        settings,
        FrameSink(),
        threading.Event(),
        lambda *args: None,
        preview_selected=lambda camera_id: camera_id == selected,
        preview_fps=30.0,
    )

    assert worker._target_fps() == 30.0
    selected = "camera-2"
    assert worker._target_fps() == 4.0


def test_preview_configures_mjpg_before_resolution() -> None:
    properties = []
    capture = SimpleNamespace(set=lambda prop, value: properties.append(prop) or True)
    worker = CameraWorker(
        CameraConfig("camera-1", "Camera 1", 0),
        SimpleNamespace(motion_min_area=0, frame_width=640, frame_height=480),
        FrameSink(),
        threading.Event(),
        lambda *args: None,
        preview_fps=30.0,
    )

    worker._configure_capture(capture)

    assert properties == [cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_BUFFERSIZE]
