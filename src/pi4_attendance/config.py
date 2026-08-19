from __future__ import annotations

import os
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
    vote_window_seconds: float
    vote_required_cameras: int
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
        cameras = tuple(
            CameraConfig(
                os.getenv(f"CAMERA_{number}_ID", f"camera-{number}"),
                os.getenv(f"CAMERA_{number}_NAME", f"Camera {number}"),
                _source(os.getenv(f"CAMERA_{number}_SOURCE", str(number - 1))),
            )
            for number in range(1, 4)
        )
        data_root = Path(os.getenv("DATA_ROOT", "./data")).expanduser().resolve()
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        settings = cls(
            data_root=data_root,
            model_root=Path(os.getenv("MODEL_ROOT", "./models/generated")).expanduser().resolve(),
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
            vote_window_seconds=float(os.getenv("VOTE_WINDOW_SECONDS", "2.0")),
            vote_required_cameras=int(os.getenv("VOTE_REQUIRED_CAMERAS", "2")),
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
        if len(self.cameras) != 3 or len({camera.camera_id for camera in self.cameras}) != 3:
            raise ValueError("Exactly three uniquely identified cameras are required")
        if not 2 <= self.vote_required_cameras <= 3:
            raise ValueError("VOTE_REQUIRED_CAMERAS must be 2 or 3")
        if self.frame_queue_size < 3:
            raise ValueError("FRAME_QUEUE_SIZE must be at least 3")
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
