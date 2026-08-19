from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from .auth import build_admin_dependency
from .config import Settings


class RuntimeContext(Protocol):
    settings: Settings

    def health(self) -> dict: ...
    def users(self) -> list[dict]: ...
    def cameras(self) -> list[dict]: ...
    def attendance(self, limit: int) -> list[dict]: ...
    def enroll(self, name: str, student_id: str, images: list[np.ndarray], raw_images: list[bytes], actor: str) -> dict: ...
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
    async def enroll(
        name: str = Form(..., min_length=1, max_length=120),
        student_id: str = Form(..., min_length=1, max_length=80),
        face_images: list[UploadFile] = File(...),
        actor: str = Depends(require_admin),
    ) -> dict:
        if not 3 <= len(face_images) <= 8:
            raise HTTPException(status_code=400, detail="Upload 3 to 8 face images")
        decoded: list[np.ndarray] = []
        raw_images: list[bytes] = []
        for upload in face_images:
            if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(status_code=400, detail="Images must be JPEG, PNG, or WebP")
            raw = await upload.read()
            if len(raw) > 8 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Each image must be no larger than 8 MB")
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="An image could not be decoded")
            decoded.append(image)
            raw_images.append(raw)
        try:
            return runtime.enroll(name, student_id, decoded, raw_images, actor)
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
