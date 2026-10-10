from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import struct
import threading
import time
import zlib
from typing import Annotated, Protocol

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from .auth import build_admin_dependency
from .config import Settings


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ENROLLMENT_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304
MAX_ENROLLMENT_PIXELS = 16_777_216
ALLOWED_IMAGE_TYPES = {"image/jpeg": "jpeg", "image/png": "png", "image/webp": "webp"}

# Web preview settings.
PREVIEW_STREAM_FPS = 10.0
PREVIEW_JPEG_QUALITY = 70

PREVIEW_SESSION_COOKIE = "sic_preview_session"
PREVIEW_SESSION_SECONDS = 12 * 60 * 60
PREVIEW_PASSWORD_ITERATIONS = 300_000
_PREVIEW_SESSIONS: dict[str, float] = {}
_PREVIEW_SESSIONS_LOCK = threading.Lock()


def _preview_credentials_ready() -> bool:
    return bool(
        os.getenv("PREVIEW_USERNAME")
        and os.getenv("PREVIEW_PASSWORD_HASH")
        and os.getenv("PREVIEW_PASSWORD_SALT")
    )


def _preview_password_valid(password: str) -> bool:
    stored_hash = os.getenv("PREVIEW_PASSWORD_HASH", "").strip().lower()
    salt_hex = os.getenv("PREVIEW_PASSWORD_SALT", "").strip()

    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(stored_hash)
    except ValueError:
        return False

    if len(salt) < 16 or len(expected) != 32:
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PREVIEW_PASSWORD_ITERATIONS,
        dklen=32,
    )
    return hmac.compare_digest(derived, expected)


def _create_preview_session() -> str:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + PREVIEW_SESSION_SECONDS
    with _PREVIEW_SESSIONS_LOCK:
        now = time.time()
        for old in [old for old, expiry in _PREVIEW_SESSIONS.items() if expiry <= now]:
            _PREVIEW_SESSIONS.pop(old, None)
        _PREVIEW_SESSIONS[token] = expires_at
    return token


def _delete_preview_session(token: str | None) -> None:
    if not token:
        return
    with _PREVIEW_SESSIONS_LOCK:
        _PREVIEW_SESSIONS.pop(token, None)


def _preview_session_valid(token: str | None) -> bool:
    if not token:
        return False

    now = time.time()
    with _PREVIEW_SESSIONS_LOCK:
        expires_at = _PREVIEW_SESSIONS.get(token)
        if expires_at is None:
            return False
        if expires_at <= now:
            _PREVIEW_SESSIONS.pop(token, None)
            return False
        return True


def _require_preview_session(request: Request) -> None:
    if not _preview_credentials_ready():
        raise HTTPException(
            status_code=503,
            detail="Preview authentication is not configured",
        )

    token = request.cookies.get(PREVIEW_SESSION_COOKIE)
    if not _preview_session_valid(token):
        raise HTTPException(
            status_code=401,
            detail="Preview login required",
        )


def _preview_login_page(error: str = "") -> str:
    error_html = (
        f'<div class="error">{error}</div>'
        if error
        else ""
    )
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>SIC Pi 4 Camera Login</title>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                background: #111;
                color: #eee;
                font-family: Arial, sans-serif;
            }}
            .box {{
                width: min(92vw, 380px);
                padding: 28px;
                background: #222;
                border-radius: 12px;
            }}
            h1 {{
                margin-top: 0;
                font-size: 24px;
            }}
            label {{
                display: block;
                margin: 14px 0 6px;
            }}
            input {{
                width: 100%;
                padding: 11px;
                border: 1px solid #555;
                border-radius: 6px;
                background: #111;
                color: #eee;
                font-size: 16px;
            }}
            button {{
                width: 100%;
                margin-top: 20px;
                padding: 11px;
                border: 0;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }}
            .error {{
                margin: 10px 0;
                color: #ff8a8a;
            }}
        </style>
    </head>
    <body>
        <form class="box" method="post" action="/preview/login">
            <h1>SIC Pi 4 Camera Preview</h1>
            {error_html}
            <label for="username">Username</label>
            <input id="username" name="username" autocomplete="username" required>

            <label for="password">Password</label>
            <input id="password" name="password" type="password"
                   autocomplete="current-password" required>

            <button type="submit">Login</button>
        </form>
    </body>
    </html>
    """




def _detected_image_type(raw: bytes) -> str | None:
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def _image_dimensions(raw: bytes, kind: str) -> tuple[int, int]:
    if kind == "png":
        if len(raw) < 33 or raw[8:16] != b"\x00\x00\x00\rIHDR":
            raise ValueError("Invalid PNG header")
        if zlib.crc32(raw[12:29]) != int.from_bytes(raw[29:33], "big"):
            raise ValueError("Invalid PNG header checksum")
        return struct.unpack(">II", raw[16:24])

    if kind == "jpeg":
        offset = 2
        dimensions = None
        while offset < len(raw):
            if raw[offset] != 0xFF:
                raise ValueError("Invalid JPEG marker")
            while offset < len(raw) and raw[offset] == 0xFF:
                offset += 1
            if offset + 3 > len(raw):
                break
            marker = raw[offset]
            offset += 1
            length = int.from_bytes(raw[offset:offset + 2], "big")
            if length < 2 or offset + length > len(raw):
                raise ValueError("Invalid JPEG segment")

            if marker in {0xC0, 0xC1, 0xC2}:
                if dimensions is not None or length < 8 or length != 8 + 3 * raw[offset + 7]:
                    raise ValueError("Invalid JPEG frame header")
                height, width = struct.unpack(">HH", raw[offset + 3:offset + 7])
                dimensions = (width, height)

            elif marker == 0xDA:
                if dimensions is not None:
                    return dimensions
                break

            elif marker in {0, 0xD8, 0xD9, 0xDC} or 0xD0 <= marker <= 0xD7:
                raise ValueError("Unsupported JPEG marker")

            offset += length

        raise ValueError("Missing JPEG frame or scan header")

    if kind == "webp":
        if int.from_bytes(raw[4:8], "little") + 8 != len(raw):
            raise ValueError("Invalid WebP container size")

        offset = 12
        canvas = None
        dimensions = None

        while offset + 8 <= len(raw):
            tag = raw[offset:offset + 4]
            size = int.from_bytes(raw[offset + 4:offset + 8], "little")
            start = offset + 8
            end = start + size

            if end + size % 2 > len(raw):
                raise ValueError("Invalid WebP chunk")

            if tag == b"VP8X":
                if size != 10 or canvas is not None or raw[start] & 2:
                    raise ValueError("Invalid or animated WebP header")

                canvas = (
                    1 + int.from_bytes(raw[start + 4:start + 7], "little"),
                    1 + int.from_bytes(raw[start + 7:start + 10], "little"),
                )

            elif tag in {b"VP8 ", b"VP8L"}:
                if dimensions is not None:
                    raise ValueError("Multiple WebP image chunks")

                if tag == b"VP8 ":
                    if size < 10 or raw[start] & 1 or raw[start + 3:start + 6] != b"\x9d\x01\x2a":
                        raise ValueError("Invalid WebP frame header")

                    width, height = struct.unpack("<HH", raw[start + 6:start + 10])
                    dimensions = (width & 0x3FFF, height & 0x3FFF)

                else:
                    if size < 5 or raw[start] != 0x2F or raw[start + 4] & 0xE0:
                        raise ValueError("Invalid lossless WebP header")

                    bits = int.from_bytes(raw[start + 1:start + 5], "little")
                    dimensions = (
                        (bits & 0x3FFF) + 1,
                        ((bits >> 14) & 0x3FFF) + 1,
                    )

            elif tag in {b"ANIM", b"ANMF"}:
                raise ValueError("Animated WebP is not supported")

            offset = end + size % 2

        if offset != len(raw) or dimensions is None or (canvas is not None and canvas != dimensions):
            raise ValueError("Invalid WebP image dimensions")

        return dimensions

    raise ValueError("Unsupported image type")


class RuntimeContext(Protocol):
    settings: Settings
    preview: object
    stop_event: object

    def health(self) -> dict: ...
    def users(self) -> list[dict]: ...
    def cameras(self) -> list[dict]: ...
    def attendance(self, limit: int) -> list[dict]: ...
    def recognitions(self, limit: int) -> list[dict]: ...
    def recognition_events(self, limit: int) -> list[dict]: ...
    def enroll(self, name: str, student_id: str, images: list[np.ndarray], actor: str) -> dict: ...
    def set_user_status(self, user_id: str, user_status: str, actor: str) -> bool: ...
    def control_status(self) -> dict: ...
    def select_camera(self, camera_id: str) -> dict: ...
    def set_recognition(self, enabled: bool) -> dict: ...
    def begin_enrollment(self, name: str, student_id: str, actor: str) -> dict: ...
    def capture_enrollment_frame(self) -> dict: ...
    def submit_enrollment(self) -> dict: ...
    def cancel_enrollment(self) -> dict: ...


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ControlUpdate(BaseModel):
    camera_id: NonEmptyText | None = None
    recognition_enabled: bool | None = None


class EnrollmentStart(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    student_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


def require_loopback(request: Request) -> None:
    try:
        hosts = request.headers.getlist("host")

        if (
            request.client is not None
            and ipaddress.ip_address(request.client.host).is_loopback
            and "origin" not in request.headers
            and len(hosts) == 1
        ):
            authority = re.fullmatch(
                r"(\[[0-9a-fA-F:]+\]|[A-Za-z0-9.]+)(?::([0-9]{1,5}))?",
                hosts[0],
            )

            if authority is not None:
                host, port = authority.groups()

                if port is None or int(port) <= 65535:
                    if host.lower() == "localhost":
                        return

                    address = ipaddress.ip_address(host.strip("[]"))

                    if address.is_loopback and (
                        (address.version == 6) == host.startswith("[")
                    ):
                        return

    except ValueError:
        pass

    raise HTTPException(status_code=403, detail="Local control access only")


def create_app(runtime: RuntimeContext) -> FastAPI:
    app = FastAPI(
        title="SIC Pi 4 Attendance API",
        version="1.0.0",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if (
            request.url.path == "/api/control"
            or request.url.path.startswith("/api/control/enrollment")
        ):
            return JSONResponse(
                status_code=400,
                content={"detail": jsonable_encoder(exc.errors())},
            )

        return await request_validation_exception_handler(request, exc)

    if runtime.settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime.settings.api_cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH"],
            allow_headers=["Authorization", "Content-Type"],
        )

    require_admin = build_admin_dependency(runtime.settings)

    # ------------------------------------------------------------------
    # WEB CAMERA PREVIEW
    # ------------------------------------------------------------------

    @app.get("/preview/login", response_class=HTMLResponse)
    def preview_login_page() -> str:
        return _preview_login_page()

    @app.post("/preview/login")
    def preview_login(username: str = Form(...), password: str = Form(...)):
        expected_username = os.getenv("PREVIEW_USERNAME", "").strip()

        if (
            not _preview_credentials_ready()
            or not expected_username
            or not hmac.compare_digest(username, expected_username)
            or not _preview_password_valid(password)
        ):
            return HTMLResponse(
                _preview_login_page("Invalid username or password"),
                status_code=401,
            )

        token = _create_preview_session()
        response = RedirectResponse(url="/preview", status_code=303)
        response.set_cookie(
            key=PREVIEW_SESSION_COOKIE,
            value=token,
            max_age=PREVIEW_SESSION_SECONDS,
            httponly=True,
            samesite="lax",
            secure=os.getenv("PREVIEW_COOKIE_SECURE", "true").strip().lower() not in {"0", "false", "no", "off"},
            path="/preview",
        )
        return response

    @app.get("/preview/logout")
    def preview_logout(request: Request):
        token = request.cookies.get(PREVIEW_SESSION_COOKIE)
        _delete_preview_session(token)
        response = RedirectResponse(url="/preview/login", status_code=303)
        response.delete_cookie(PREVIEW_SESSION_COOKIE, path="/preview")
        return response

    @app.get("/preview", response_class=HTMLResponse)
    def preview_page(request: Request):
        """
        Web page showing all configured cameras.

        Frames come from the existing CameraWorker -> PreviewWindow path.
        No additional cv2.VideoCapture is created here.
        """
        if not _preview_credentials_ready():
            raise HTTPException(
                status_code=503,
                detail="Preview authentication is not configured",
            )

        token = request.cookies.get(PREVIEW_SESSION_COOKIE)
        if not _preview_session_valid(token):
            return RedirectResponse(
                url="/preview/login",
                status_code=303,
            )

        camera_ids = tuple(runtime.preview.camera_ids)
        camera_blocks = []

        for camera_id in camera_ids:
            camera_blocks.append(
                f"""
                <div class="camera">
                    <div class="camera-title">{camera_id}</div>
                    <img
                        src="/preview/{camera_id}"
                        alt="{camera_id}"
                    >
                </div>
                """
            )

        cameras_html = "\n".join(camera_blocks)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1"
            >
            <title>SIC Pi 4 Camera Preview</title>

            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    margin: 0;
                    padding: 20px;
                    background: #111;
                    color: #eee;
                    font-family: Arial, sans-serif;
                }}

                h1 {{
                    margin-top: 0;
                    font-size: 24px;
                }}

                .topbar {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                }}

                .logout {{
                    color: #eee;
                    text-decoration: none;
                    padding: 8px 12px;
                    background: #333;
                    border-radius: 6px;
                }}

                .grid {{
                    display: grid;
                    grid-template-columns: repeat(
                        auto-fit,
                        minmax(300px, 1fr)
                    );
                    gap: 16px;
                }}

                .camera {{
                    background: #222;
                    border-radius: 8px;
                    padding: 10px;
                    overflow: hidden;
                }}

                .camera-title {{
                    font-size: 16px;
                    font-weight: bold;
                    margin-bottom: 8px;
                }}

                img {{
                    display: block;
                    width: 100%;
                    height: auto;
                    background: #000;
                    border-radius: 4px;
                }}
            </style>
        </head>

        <body>
            <div class="topbar">
                <h1>SIC Pi 4 Camera Preview</h1>
                <a class="logout" href="/preview/logout">Logout</a>
            </div>

            <div class="grid">
                {cameras_html}
            </div>
        </body>
        </html>
        """

    @app.get("/preview/{camera_id}")
    def camera_preview(camera_id: str, request: Request):
        """
        MJPEG stream for one camera.

        The camera frame is obtained from PreviewWindow.latest_sample().
        This does NOT open another camera device.
        """
        _require_preview_session(request)

        camera_ids = tuple(runtime.preview.camera_ids)

        if camera_id not in camera_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown camera: {camera_id}",
            )

        def generate():
            last_sequence = -1
            frame_interval = 1.0 / PREVIEW_STREAM_FPS
            token = request.cookies.get(PREVIEW_SESSION_COOKIE)

            while not runtime.stop_event.is_set():
                if not _preview_credentials_ready() or not _preview_session_valid(token):
                    break
                started = time.monotonic()

                try:
                    sample = getattr(runtime.preview, "latest_display_sample", runtime.preview.latest_sample)
                    frame, sequence = sample(camera_id)
                except Exception:
                    break

                if frame is not None and sequence != last_sequence:
                    last_sequence = sequence

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            PREVIEW_JPEG_QUALITY,
                        ],
                    )

                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Cache-Control: no-cache\r\n"
                            b"Pragma: no-cache\r\n"
                            b"\r\n"
                            + encoded.tobytes()
                            + b"\r\n"
                        )

                elapsed = time.monotonic() - started
                remaining = frame_interval - elapsed

                if remaining > 0:
                    time.sleep(remaining)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


    # ------------------------------------------------------------------
    # EXISTING API
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return runtime.health()

    @app.get("/api/users")
    def users(_: str = Depends(require_admin)) -> list[dict]:
        return runtime.users()

    @app.get("/api/cameras")
    def cameras(_: str = Depends(require_admin)) -> list[dict]:
        return runtime.cameras()

    @app.get("/api/attendance")
    def attendance(
        limit: int = 100,
        _: str = Depends(require_admin),
    ) -> list[dict]:
        return runtime.attendance(limit)

    @app.get("/api/recognitions")
    def recognitions(
        limit: int = 100,
        _: str = Depends(require_admin),
    ) -> list[dict]:
        return runtime.recognitions(max(1, min(limit, 500)))

    @app.get("/api/recognition-events")
    def recognition_events(limit: int = 100, _: str = Depends(require_admin)) -> list[dict]:
        return runtime.recognition_events(max(1, min(limit, 500)))

    @app.get("/api/control/remote")
    def remote_control(_: str = Depends(require_admin)) -> dict:
        return runtime.control_status()

    @app.patch("/api/control/remote")
    def update_remote_control(update: ControlUpdate, _: str = Depends(require_admin)) -> dict:
        try:
            result = None
            if update.camera_id is not None:
                result = runtime.select_camera(update.camera_id)
            if update.recognition_enabled is not None:
                result = runtime.set_recognition(update.recognition_enabled)
            return runtime.control_status() if result is None else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/api/control",
        dependencies=[Depends(require_loopback)],
    )
    def control() -> dict:
        return runtime.control_status()

    @app.patch(
        "/api/control",
        dependencies=[Depends(require_loopback)],
    )
    def update_control(update: ControlUpdate) -> dict:
        result = None

        try:
            if update.camera_id is not None:
                result = runtime.select_camera(update.camera_id)

            if update.recognition_enabled is not None:
                result = runtime.set_recognition(
                    update.recognition_enabled
                )

            return (
                runtime.control_status()
                if result is None
                else result
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/api/control/enrollment/start",
        dependencies=[Depends(require_loopback)],
    )
    def start_control_enrollment(
        enrollment: EnrollmentStart,
    ) -> dict:
        try:
            return runtime.begin_enrollment(
                enrollment.name,
                enrollment.student_id,
                "local-console",
            )

        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/api/control/enrollment/capture",
        dependencies=[Depends(require_loopback)],
    )
    def capture_control_enrollment() -> dict:
        try:
            return runtime.capture_enrollment_frame()

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/api/control/enrollment/submit",
        dependencies=[Depends(require_loopback)],
    )
    def submit_control_enrollment() -> dict:
        try:
            return runtime.submit_enrollment()

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.delete(
        "/api/control/enrollment",
        dependencies=[Depends(require_loopback)],
    )
    def cancel_control_enrollment() -> dict:
        try:
            return runtime.cancel_enrollment()

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/api/users",
        status_code=status.HTTP_201_CREATED,
    )
    def enroll(
        name: str = Form(..., min_length=1, max_length=120),
        student_id: str = Form(..., min_length=1, max_length=80),
        face_images: list[UploadFile] = File(...),
        actor: str = Depends(require_admin),
    ) -> dict:
        if not 3 <= len(face_images) <= 8:
            raise HTTPException(
                status_code=400,
                detail="Upload 3 to 8 face images",
            )

        total_bytes = 0

        for upload in face_images:
            upload.file.seek(0, 2)
            size = upload.file.tell()
            upload.file.seek(0)

            total_bytes += size

            if size <= 0 or size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Each image must contain 1 byte to 8 MB",
                )

            if total_bytes > MAX_ENROLLMENT_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Enrollment images must total no more than 32 MB",
                )

        encoded = []
        total_pixels = 0

        for upload in face_images:
            declared_type = (upload.content_type or "").lower()

            if declared_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail="Images must be JPEG, PNG, or WebP",
                )

            raw = upload.file.read(MAX_UPLOAD_BYTES + 1)

            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Each image must be no larger than 8 MB",
                )

            if (
                not raw
                or _detected_image_type(raw)
                != ALLOWED_IMAGE_TYPES[declared_type]
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Image content does not match its declared file type",
                )

            try:
                width, height = _image_dimensions(
                    raw,
                    ALLOWED_IMAGE_TYPES[declared_type],
                )

            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            pixels = width * height
            total_pixels += pixels

            if (
                width <= 0
                or height <= 0
                or width > 4096
                or height > 4096
                or pixels > MAX_IMAGE_PIXELS
                or total_pixels > MAX_ENROLLMENT_PIXELS
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions exceed the enrollment limit",
                )

            encoded.append((raw, width, height))

        decoded: list[np.ndarray] = []

        for raw, width, height in encoded:
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            if (
                image is None
                or image.shape[:2] not in {
                    (height, width),
                    (width, height),
                }
            ):
                raise HTTPException(
                    status_code=400,
                    detail="An image could not be decoded with its declared dimensions",
                )

            decoded.append(image)

        try:
            return runtime.enroll(
                name,
                student_id,
                decoded,
                actor,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise HTTPException(
                    status_code=409,
                    detail="Student ID is already registered",
                ) from exc

            raise

    @app.patch("/api/users/{user_id}/{user_status}")
    def set_status(
        user_id: str,
        user_status: str,
        actor: str = Depends(require_admin),
    ) -> dict:
        if user_status not in {"active", "inactive"}:
            raise HTTPException(
                status_code=400,
                detail="Status must be active or inactive",
            )

        if not runtime.set_user_status(
            user_id,
            user_status,
            actor,
        ):
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        return {
            "user_id": user_id,
            "status": user_status,
        }

    return app
