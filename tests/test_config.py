from dataclasses import replace
import math
from pathlib import Path

from pi4_attendance.config import CameraConfig, Settings


def settings() -> Settings:
    return Settings(
        data_root=Path("data"), model_root=Path("models"), model_pack="pi4_fp32", pi_id="pi-1",
        cameras=(CameraConfig("camera-1", "Camera 1", 0),), frame_width=640, frame_height=480,
        monitor_fps=4.0, motion_min_area=1800, camera_trigger_cooldown_seconds=0.8,
        frame_queue_size=3, face_det_size=320, match_threshold=0.45, match_margin=0.06,
        min_detection_score=0.60, min_face_quality=0.35, attendance_cooldown_seconds=300,
        save_unknown_images=True, unknown_retention_days=14, api_enabled=False, api_host="127.0.0.1",
        api_port=8080, api_cors_origins=(), api_auth_mode="shared_token", firebase_service_account=None,
        admin_shared_token=None, cloud_summary_enabled=False, firebase_project_id=None, log_level="INFO",
    )


def test_one_unique_camera_is_valid() -> None:
    settings().validate()


def test_duplicate_or_empty_camera_ids_are_rejected() -> None:
    base = settings()
    for cameras in (
        (),
        (CameraConfig("", "Empty", 0),),
        (CameraConfig("   ", "Whitespace", 0),),
        (CameraConfig("camera-1", "One", 0), CameraConfig("camera-1", "Two", 1)),
    ):
        try:
            replace(base, cameras=cameras).validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid camera configuration was accepted")


def test_runtime_dimensions_and_rates_must_be_positive() -> None:
    base = settings()
    invalid = (
        {"frame_width": 0},
        {"frame_height": -1},
        {"monitor_fps": 0.0},
        {"camera_trigger_cooldown_seconds": 0.0},
        {"face_det_size": 0},
    )
    for overrides in invalid:
        try:
            replace(base, **overrides).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid runtime configuration was accepted: {overrides}")


def test_thresholds_must_be_finite_and_within_unit_interval() -> None:
    base = settings()
    for field in ("match_threshold", "match_margin", "min_detection_score", "min_face_quality"):
        for value in (-0.01, 1.01, math.nan, math.inf, -math.inf):
            try:
                replace(base, **{field: value}).validate()
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid {field} was accepted: {value}")


def test_limits_and_capacity_are_sane() -> None:
    base = replace(
        settings(),
        cameras=(CameraConfig("c1", "One", 0), CameraConfig("c2", "Two", 1)),
        frame_queue_size=2,
    )
    base.validate()
    invalid = (
        {"motion_min_area": -1},
        {"attendance_cooldown_seconds": -1},
        {"unknown_retention_days": -1},
        {"api_port": 0},
        {"api_port": 65536},
        {"frame_queue_size": 1},
    )
    for overrides in invalid:
        try:
            replace(base, **overrides).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid limit was accepted: {overrides}")


def test_shared_token_auth_rejects_sample_and_weak_tokens() -> None:
    base = replace(settings(), api_enabled=True)
    for token in (
        None,
        "short",
        "REPLACE_WITH_A_LONG_RANDOM_TOKEN",
        "change-me-to-a-secure-random-token-123456",
        "example_admin_token_that_is_long_enough_123",
        "a" * 40,
    ):
        try:
            replace(base, admin_shared_token=token).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe shared token was accepted: {token!r}")

    replace(base, admin_shared_token="lP2y5ER0m3k9FQ6uX8cV1aZ7nD4sT0wB").validate()
