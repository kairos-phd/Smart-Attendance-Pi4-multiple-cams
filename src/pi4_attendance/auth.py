from __future__ import annotations

import secrets
from collections.abc import Callable

import firebase_admin
from fastapi import Header, HTTPException, status
from firebase_admin import auth, credentials

from .config import Settings


def build_admin_dependency(settings: Settings) -> Callable:
    if settings.api_auth_mode == "firebase":
        assert settings.firebase_service_account is not None
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(str(settings.firebase_service_account)))

        def require_admin(authorization: str | None = Header(default=None)) -> str:
            token = _bearer(authorization)
            try:
                claims = auth.verify_id_token(token, check_revoked=True)
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc
            if claims.get("admin") is not True:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access is required")
            return str(claims.get("uid", "firebase-admin"))

        return require_admin

    expected = settings.admin_shared_token or ""

    def require_shared_token(authorization: str | None = Header(default=None)) -> str:
        supplied = _bearer(authorization)
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid administrator token")
        return "shared-token-admin"

    return require_shared_token


def _bearer(value: str | None) -> str:
    if not value or not value.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is required")
    return value[7:].strip()

