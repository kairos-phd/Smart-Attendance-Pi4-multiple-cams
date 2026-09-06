from __future__ import annotations

import struct
import zlib
from typing import Protocol

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from .auth import build_admin_dependency
from .config import Settings


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ENROLLMENT_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304
MAX_ENROLLMENT_PIXELS = 16_777_216
ALLOWED_IMAGE_TYPES = {"image/jpeg": "jpeg", "image/png": "png", "image/webp": "webp"}


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
                canvas = (1 + int.from_bytes(raw[start + 4:start + 7], "little"),
                          1 + int.from_bytes(raw[start + 7:start + 10], "little"))
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
                    dimensions = ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
            elif tag in {b"ANIM", b"ANMF"}:
                raise ValueError("Animated WebP is not supported")
            offset = end + size % 2
        if offset != len(raw) or dimensions is None or (canvas is not None and canvas != dimensions):
            raise ValueError("Invalid WebP image dimensions")
        return dimensions
    raise ValueError("Unsupported image type")


class RuntimeContext(Protocol):
    settings: Settings

    def health(self) -> dict: ...
    def users(self) -> list[dict]: ...
    def cameras(self) -> list[dict]: ...
    def attendance(self, limit: int) -> list[dict]: ...
    def enroll(self, name: str, student_id: str, images: list[np.ndarray], actor: str) -> dict: ...
    def set_user_status(self, user_id: str, user_status: str, actor: str) -> bool: ...


def create_app(runtime: RuntimeContext) -> FastAPI:
    app = FastAPI(title="SIC Pi 4 Attendance API", version="1.0.0")
    if runtime.settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime.settings.api_cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH"],
            allow_headers=["Authorization", "Content-Type"],
        )
    require_admin = build_admin_dependency(runtime.settings)

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
    def attendance(limit: int = 100, _: str = Depends(require_admin)) -> list[dict]:
        return runtime.attendance(limit)

    @app.post("/api/users", status_code=status.HTTP_201_CREATED)
    def enroll(
        name: str = Form(..., min_length=1, max_length=120),
        student_id: str = Form(..., min_length=1, max_length=80),
        face_images: list[UploadFile] = File(...),
        actor: str = Depends(require_admin),
    ) -> dict:
        if not 3 <= len(face_images) <= 8:
            raise HTTPException(status_code=400, detail="Upload 3 to 8 face images")
        total_bytes = 0
        for upload in face_images:
            upload.file.seek(0, 2)
            size = upload.file.tell()
            upload.file.seek(0)
            total_bytes += size
            if size <= 0 or size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="Each image must contain 1 byte to 8 MB")
            if total_bytes > MAX_ENROLLMENT_BYTES:
                raise HTTPException(status_code=400, detail="Enrollment images must total no more than 32 MB")
        encoded = []
        total_pixels = 0
        for upload in face_images:
            declared_type = (upload.content_type or "").lower()
            if declared_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(status_code=400, detail="Images must be JPEG, PNG, or WebP")
            raw = upload.file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="Each image must be no larger than 8 MB")
            if not raw or _detected_image_type(raw) != ALLOWED_IMAGE_TYPES[declared_type]:
                raise HTTPException(status_code=400, detail="Image content does not match its declared file type")
            try:
                width, height = _image_dimensions(raw, ALLOWED_IMAGE_TYPES[declared_type])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            pixels = width * height
            total_pixels += pixels
            if width <= 0 or height <= 0 or width > 4096 or height > 4096 or pixels > MAX_IMAGE_PIXELS or total_pixels > MAX_ENROLLMENT_PIXELS:
                raise HTTPException(status_code=400, detail="Image dimensions exceed the enrollment limit")
            encoded.append((raw, width, height))
        decoded: list[np.ndarray] = []
        for raw, width, height in encoded:
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] not in {(height, width), (width, height)}:
                raise HTTPException(status_code=400, detail="An image could not be decoded with its declared dimensions")
            decoded.append(image)
        try:
            return runtime.enroll(name, student_id, decoded, actor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                raise HTTPException(status_code=409, detail="Student ID is already registered") from exc
            raise

    @app.patch("/api/users/{user_id}/{user_status}")
    def set_status(user_id: str, user_status: str, actor: str = Depends(require_admin)) -> dict:
        if user_status not in {"active", "inactive"}:
            raise HTTPException(status_code=400, detail="Status must be active or inactive")
        if not runtime.set_user_status(user_id, user_status, actor):
            raise HTTPException(status_code=404, detail="User not found")
        return {"user_id": user_id, "status": user_status}

    return app
