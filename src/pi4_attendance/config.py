from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    source: int | str


@dataclass(frozen=True)
class Settings:
    data_root: Path
    model_root: Path
    model_pack: str
    pi_id: str
    cameras: tuple[CameraConfig, ...]
    frame_width: int
    frame_height: int
    monitor_fps: float
    motion_min_area: int
    camera_trigger_cooldown_seconds: float
    frame_queue_size: int
    face_det_size: int
    match_threshold: float
    match_margin: float
    min_detection_score: float
    min_face_quality: float
    attendance_cooldown_seconds: int
    save_unknown_images: bool
    unknown_retention_days: int
    api_enabled: bool
    api_host: str
    api_port: int
    api_cors_origins: tuple[str, ...]
    api_auth_mode: str
    firebase_service_account: Path | None
    admin_shared_token: str | None
    cloud_summary_enabled: bool
    firebase_project_id: str | None
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        camera_count = int(os.getenv("CAMERA_COUNT", "3"))
        cameras = tuple(
            CameraConfig(
                os.getenv(f"CAMERA_{number}_ID", f"camera-{number}"),
                os.getenv(f"CAMERA_{number}_NAME", f"Camera {number}"),
                _source(os.getenv(f"CAMERA_{number}_SOURCE", str(number - 1))),
            )
            for number in range(1, camera_count + 1)
        )
        data_root = Path(os.getenv("DATA_ROOT", "./data")).expanduser().resolve()
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        settings = cls(
            data_root=data_root,
            model_root=Path(os.getenv("MODEL_ROOT", ".")).expanduser().resolve(),
            model_pack=os.getenv("MODEL_PACK", "pi4_fp32"),
            pi_id=os.getenv("PI_ID", "pi-classroom-01"),
            cameras=cameras,
            frame_width=int(os.getenv("FRAME_WIDTH", "640")),
            frame_height=int(os.getenv("FRAME_HEIGHT", "480")),
            monitor_fps=float(os.getenv("MONITOR_FPS", "4")),
            motion_min_area=int(os.getenv("MOTION_MIN_AREA", "1800")),
            camera_trigger_cooldown_seconds=float(os.getenv("CAMERA_TRIGGER_COOLDOWN_SECONDS", "0.8")),
            frame_queue_size=int(os.getenv("FRAME_QUEUE_SIZE", "9")),
            face_det_size=int(os.getenv("FACE_DET_SIZE", "320")),
            match_threshold=float(os.getenv("MATCH_THRESHOLD", "0.45")),
            match_margin=float(os.getenv("MATCH_MARGIN", "0.06")),
            min_detection_score=float(os.getenv("MIN_DETECTION_SCORE", "0.60")),
            min_face_quality=float(os.getenv("MIN_FACE_QUALITY", "0.35")),
            attendance_cooldown_seconds=int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", "300")),
            save_unknown_images=_bool("SAVE_UNKNOWN_IMAGES", True),
            unknown_retention_days=int(os.getenv("UNKNOWN_RETENTION_DAYS", "14")),
            api_enabled=_bool("API_ENABLED", True),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8080")),
            api_cors_origins=tuple(
                origin.strip() for origin in os.getenv("API_CORS_ORIGINS", "").split(",") if origin.strip()
            ),
            api_auth_mode=os.getenv("API_AUTH_MODE", "firebase").strip().lower(),
            firebase_service_account=Path(service_account).expanduser() if service_account else None,
            admin_shared_token=os.getenv("ADMIN_SHARED_TOKEN") or None,
            cloud_summary_enabled=_bool("CLOUD_SUMMARY_ENABLED", False),
            firebase_project_id=os.getenv("FIREBASE_PROJECT_ID") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if (
            not self.cameras
            or any(not camera.camera_id.strip() for camera in self.cameras)
            or len({camera.camera_id for camera in self.cameras}) != len(self.cameras)
        ):
            raise ValueError("At least one uniquely identified camera is required")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("FRAME_WIDTH and FRAME_HEIGHT must be positive")
        if self.monitor_fps <= 0:
            raise ValueError("MONITOR_FPS must be positive")
        if self.camera_trigger_cooldown_seconds <= 0:
            raise ValueError("CAMERA_TRIGGER_COOLDOWN_SECONDS must be positive")
        if self.face_det_size <= 0:
            raise ValueError("FACE_DET_SIZE must be positive")
        if self.frame_queue_size < len(self.cameras):
            raise ValueError("FRAME_QUEUE_SIZE must provide at least one slot per camera")
        thresholds = {
            "MATCH_THRESHOLD": self.match_threshold,
            "MATCH_MARGIN": self.match_margin,
            "MIN_DETECTION_SCORE": self.min_detection_score,
            "MIN_FACE_QUALITY": self.min_face_quality,
        }
        for name, value in thresholds.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.motion_min_area < 0:
            raise ValueError("MOTION_MIN_AREA must be nonnegative")
        if self.attendance_cooldown_seconds < 0:
            raise ValueError("ATTENDANCE_COOLDOWN_SECONDS must be nonnegative")
        if self.unknown_retention_days < 0:
            raise ValueError("UNKNOWN_RETENTION_DAYS must be nonnegative")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        if self.api_auth_mode not in {"firebase", "shared_token"}:
            raise ValueError("API_AUTH_MODE must be firebase or shared_token")
        if "*" in self.api_cors_origins:
            raise ValueError("Wildcard API CORS origins are not allowed")
        if self.api_enabled and self.api_auth_mode == "firebase" and not self.firebase_service_account:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is required for Firebase API authentication")
        if self.cloud_summary_enabled and not self.firebase_service_account:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is required when cloud summary sync is enabled")
        if self.api_enabled and self.api_auth_mode == "shared_token" and not self.admin_shared_token:
            raise ValueError("ADMIN_SHARED_TOKEN is required for shared-token API authentication")
        if self.api_enabled and self.api_auth_mode == "shared_token":
            token = self.admin_shared_token or ""
            normalized = token.lower().replace("_", "-")
            if (
                len(token) < 32
                or len(set(token)) < 8
                or any(marker in normalized for marker in ("replace", "change-me", "changeme", "example", "your-token"))
            ):
                raise ValueError("ADMIN_SHARED_TOKEN must be a long random value, not a sample token")
