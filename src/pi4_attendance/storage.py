from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import Observation, RecognitionDecision, RegisteredUser


SCHEMA_VERSION = 2


class PiStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.enrollment_root = data_root / "enrollments"
        self.unknown_root = data_root / "unknown"
        self.tmp_root = data_root / "tmp"
        for directory in (data_root, self.enrollment_root, self.unknown_root, self.tmp_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        self.database_path = data_root / "smart_attendance.sqlite3"
        self._conn = sqlite3.connect(self.database_path, check_same_thread=False, timeout=30)
        self.database_path.chmod(0o600)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                  user_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  student_id TEXT NOT NULL UNIQUE,
                  embedding_int8 BLOB NOT NULL,
                  embedding_dim INTEGER NOT NULL,
                  embedding_scale REAL NOT NULL,
                  model_version TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','inactive')),
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS enrollment_images (
                  image_id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                  relative_path TEXT NOT NULL UNIQUE,
                  sha256 TEXT NOT NULL,
                  width INTEGER NOT NULL,
                  height INTEGER NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                  observation_id TEXT PRIMARY KEY,
                  session_id TEXT,
                  camera_id TEXT NOT NULL,
                  observed_at TEXT NOT NULL,
                  candidate_user_id TEXT,
                  similarity_scaled INTEGER NOT NULL,
                  margin_scaled INTEGER NOT NULL,
                  detection_scaled INTEGER NOT NULL,
                  quality_scaled INTEGER NOT NULL,
                  bbox_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS observations_time_idx ON observations(observed_at);
                CREATE TABLE IF NOT EXISTS vote_sessions (
                  session_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  user_id TEXT,
                  name TEXT,
                  confidence_scaled INTEGER NOT NULL,
                  camera_ids_json TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  decided_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_events (
                  observation_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  user_id TEXT,
                  name TEXT,
                  camera_id TEXT NOT NULL,
                  observed_at TEXT NOT NULL,
                  similarity_scaled INTEGER NOT NULL,
                  margin_scaled INTEGER NOT NULL,
                  detection_scaled INTEGER NOT NULL,
                  quality_scaled INTEGER NOT NULL,
                  bbox_json TEXT NOT NULL,
                  model_version TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  decided_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS recognition_events_time_idx ON recognition_events(observed_at);
                CREATE TABLE IF NOT EXISTS attendance (
                  attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL UNIQUE REFERENCES vote_sessions(session_id),
                  user_id TEXT NOT NULL REFERENCES users(user_id),
                  name TEXT NOT NULL,
                  confidence_scaled INTEGER NOT NULL,
                  camera_ids_json TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  sync_state TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE TABLE IF NOT EXISTS camera_health (
                  camera_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  last_seen TEXT NOT NULL,
                  last_error TEXT,
                  dropped_frames INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS unknown_events (
                  event_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL REFERENCES vote_sessions(session_id),
                  relative_image_path TEXT,
                  reason TEXT NOT NULL,
                  camera_ids_json TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  review_status TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                  audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  actor TEXT NOT NULL,
                  action TEXT NOT NULL,
                  target TEXT,
                  details_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                "INSERT INTO schema_meta(key,value) VALUES('version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _scaled(value: float) -> int:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Stored recognition metrics must be finite")
        return int(round(max(-1.0, min(1.0, value)) * 10_000))

    def active_users(self) -> list[RegisteredUser]:
        with self._lock:
            return self._active_users_unlocked()

    def _active_users_unlocked(self) -> list[RegisteredUser]:
        rows = self._conn.execute("SELECT * FROM users WHERE status='active' ORDER BY name").fetchall()
        return [
            RegisteredUser(
                user_id=row["user_id"],
                name=row["name"],
                student_id=row["student_id"],
                embedding_q=np.frombuffer(row["embedding_int8"], dtype=np.int8, count=row["embedding_dim"]).copy(),
                embedding_scale=float(row["embedding_scale"]),
                model_version=row["model_version"],
            )
            for row in rows
        ]

    def user_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id,name,student_id,status,model_version,created_at,updated_at FROM users ORDER BY name"
            ).fetchall()
        return [dict(row) for row in rows]

    def camera_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM camera_health ORDER BY camera_id").fetchall()
        return [dict(row) for row in rows]

    def recent_attendance(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT attendance_id,session_id,user_id,name,confidence_scaled,camera_ids_json,occurred_at,sync_state "
                "FROM attendance ORDER BY attendance_id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_attendance(self, limit: int = 50) -> list[RecognitionDecision]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event.* FROM attendance AS attendance "
                "JOIN recognition_events AS event ON event.observation_id=attendance.session_id "
                "WHERE attendance.sync_state='pending' ORDER BY attendance.attendance_id LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [
            RecognitionDecision(
                observation_id=row["observation_id"],
                status=row["status"],
                user_id=row["user_id"],
                name=row["name"],
                camera_id=row["camera_id"],
                observed_at=datetime.fromisoformat(row["observed_at"]),
                similarity=row["similarity_scaled"] / 10_000,
                margin=row["margin_scaled"] / 10_000,
                detection_score=row["detection_scaled"] / 10_000,
                quality=row["quality_scaled"] / 10_000,
                bbox=tuple(json.loads(row["bbox_json"])),
                model_version=row["model_version"],
                reason=row["reason"],
                decided_at=datetime.fromisoformat(row["decided_at"]),
            )
            for row in rows
        ]

    def mark_attendance_synced(self, observation_id: str) -> bool:
        with self._lock, self._conn:
            result = self._conn.execute(
                "UPDATE attendance SET sync_state='synced' WHERE session_id=? AND sync_state='pending'",
                (observation_id,),
            )
        return result.rowcount == 1

    def create_user(
        self, user_id: str, name: str, student_id: str, embedding_q: np.ndarray, scale: float, model_version: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        raw_embedding = np.asarray(embedding_q)
        if raw_embedding.dtype != np.int8 or raw_embedding.ndim != 1 or raw_embedding.size <= 0:
            raise ValueError("Embedding must be a non-empty one-dimensional INT8 array")
        if not np.any(raw_embedding):
            raise ValueError("Embedding must be a non-zero vector")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Embedding scale must be positive and finite")
        if not str(model_version).strip():
            raise ValueError("Model version is required")
        embedding = raw_embedding
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO users(user_id,name,student_id,embedding_int8,embedding_dim,embedding_scale,model_version,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (user_id, name.strip(), student_id.strip(), embedding.tobytes(), embedding.size, scale, model_version, now, now),
            )

    def add_enrollment_image(
        self, image_id: str, user_id: str, relative_path: str, sha256: str, width: int, height: int
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO enrollment_images VALUES(?,?,?,?,?,?,?)",
                (image_id, user_id, relative_path, sha256, width, height, datetime.now(timezone.utc).isoformat()),
            )

    def enroll_user(
        self,
        user_id: str,
        name: str,
        student_id: str,
        embedding_q: np.ndarray,
        scale: float,
        model_version: str,
        images: list[tuple[str, str, str, int, int]],
        actor: str,
        prepare_catalog,
    ):
        now = datetime.now(timezone.utc).isoformat()
        embedding = np.asarray(embedding_q)
        if embedding.dtype != np.int8 or embedding.ndim != 1 or embedding.size <= 0:
            raise ValueError("Embedding must be a non-empty one-dimensional INT8 array")
        if not np.any(embedding):
            raise ValueError("Embedding must be a non-zero vector")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Embedding scale must be positive and finite")
        if not str(model_version).strip():
            raise ValueError("Model version is required")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO users(user_id,name,student_id,embedding_int8,embedding_dim,embedding_scale,model_version,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (user_id, name.strip(), student_id.strip(), embedding.tobytes(), embedding.size, scale, model_version, now, now),
            )
            self._conn.executemany(
                "INSERT INTO enrollment_images VALUES(?,?,?,?,?,?,?)",
                [(image_id, user_id, path, digest, width, height, now) for image_id, path, digest, width, height in images],
            )
            self._conn.execute(
                "INSERT INTO audit_log(actor,action,target,details_json,created_at) VALUES(?,?,?,?,?)",
                (actor, "enroll_user", user_id, json.dumps({"image_count": len(images), "student_id": student_id.strip()}, sort_keys=True), now),
            )
            return prepare_catalog(self._active_users_unlocked())

    def set_user_status(self, user_id: str, status: str) -> bool:
        if status not in {"active", "inactive"}:
            raise ValueError("Invalid user status")
        with self._lock, self._conn:
            result = self._conn.execute(
                "UPDATE users SET status=?,updated_at=? WHERE user_id=?",
                (status, datetime.now(timezone.utc).isoformat(), user_id),
            )
        return result.rowcount == 1

    def set_user_status_audited(self, user_id: str, status: str, actor: str) -> bool:
        if status not in {"active", "inactive"}:
            raise ValueError("Invalid user status")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            result = self._conn.execute(
                "UPDATE users SET status=?,updated_at=? WHERE user_id=?", (status, now, user_id)
            )
            if result.rowcount == 1:
                self._conn.execute(
                    "INSERT INTO audit_log(actor,action,target,details_json,created_at) VALUES(?,?,?,?,?)",
                    (actor, f"set_user_{status}", user_id, "{}", now),
                )
        return result.rowcount == 1

    def delete_incomplete_user(self, user_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM users WHERE user_id=?", (user_id,))

    def record_observation(self, observation: Observation, session_id: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    observation.observation_id,
                    session_id,
                    observation.camera_id,
                    observation.observed_at.isoformat(),
                    observation.candidate_user_id,
                    self._scaled(observation.similarity),
                    self._scaled(observation.margin),
                    self._scaled(observation.detection_score),
                    self._scaled(observation.quality),
                    json.dumps(observation.bbox),
                ),
            )

    def record_decision(self, decision: RecognitionDecision, cooldown_seconds: int) -> bool:
        cameras_json = json.dumps((decision.camera_id,))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO recognition_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.observation_id,
                    decision.status,
                    decision.user_id,
                    decision.name,
                    decision.camera_id,
                    decision.observed_at.isoformat(),
                    self._scaled(decision.similarity),
                    self._scaled(decision.margin),
                    self._scaled(decision.detection_score),
                    self._scaled(decision.quality),
                    json.dumps(decision.bbox),
                    decision.model_version,
                    decision.reason,
                    decision.decided_at.isoformat(),
                ),
            )
            if decision.status != "accepted" or not decision.user_id or not decision.name:
                return False
            active = self._conn.execute(
                "SELECT 1 FROM users WHERE user_id=? AND status='active'", (decision.user_id,)
            ).fetchone()
            if not active:
                return False
            cutoff = (decision.decided_at - timedelta(seconds=cooldown_seconds)).isoformat()
            duplicate = self._conn.execute(
                "SELECT 1 FROM attendance WHERE user_id=? AND occurred_at>=? LIMIT 1",
                (decision.user_id, cutoff),
            ).fetchone()
            if duplicate:
                return False
            self._conn.execute(
                "INSERT OR IGNORE INTO vote_sessions VALUES(?,?,?,?,?,?,?,?)",
                (
                    decision.observation_id,
                    "confirmed",
                    decision.user_id,
                    decision.name,
                    self._scaled(decision.similarity),
                    cameras_json,
                    "single_observation_policy",
                    decision.decided_at.isoformat(),
                ),
            )
            self._conn.execute(
                "INSERT INTO attendance(session_id,user_id,name,confidence_scaled,camera_ids_json,occurred_at) VALUES(?,?,?,?,?,?)",
                (
                    decision.observation_id,
                    decision.user_id,
                    decision.name,
                    self._scaled(decision.similarity),
                    cameras_json,
                    decision.decided_at.isoformat(),
                ),
            )
            return True

    def recognition_event_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM recognition_events ORDER BY observed_at").fetchall()
        return [dict(row) for row in rows]

    def embedding_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        return [
            {
                **dict(row),
                "embedding_q": np.frombuffer(row["embedding_int8"], dtype=np.int8, count=row["embedding_dim"]).copy(),
            }
            for row in rows
        ]

    def import_users(self, rows: list[tuple[str, str, str, np.ndarray, float, str, str]]) -> None:
        dimensions: dict[str, int] = {}
        for user_id, name, student_id, embedding, scale, model, status in rows:
            if embedding.dtype != np.int8 or embedding.ndim != 1 or embedding.size <= 0:
                raise ValueError("Embedding must be a non-empty one-dimensional INT8 array")
            if not np.any(embedding):
                raise ValueError("Embedding must be a non-zero vector")
            if not np.isfinite(scale) or scale <= 0 or status not in {"active", "inactive"} or not str(model).strip():
                raise ValueError("Invalid imported embedding metadata")
            if model in dimensions and dimensions[model] != embedding.size:
                raise ValueError("Imported embeddings have inconsistent dimensions")
            dimensions[model] = embedding.size
        now = datetime.now(timezone.utc).isoformat()
        values = [
            (user_id, name, student_id, embedding.tobytes(), embedding.size, scale, model, status, now, now)
            for user_id, name, student_id, embedding, scale, model, status in rows
        ]
        with self._lock, self._conn:
            for model, dimension in dimensions.items():
                existing = self._conn.execute(
                    "SELECT DISTINCT embedding_dim FROM users WHERE model_version=?", (model,)
                ).fetchall()
                if any(row["embedding_dim"] != dimension for row in existing):
                    raise ValueError("Embedding dimension is incompatible with existing records for this model")
            self._conn.executemany(
                "INSERT INTO users(user_id,name,student_id,embedding_int8,embedding_dim,embedding_scale,model_version,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                values,
            )

    def update_camera(self, camera_id: str, name: str, status: str, error: str | None = None, dropped: int = 0) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO camera_health(camera_id,name,status,last_seen,last_error,dropped_frames) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name,status=excluded.status,last_seen=excluded.last_seen,"
                "last_error=excluded.last_error,dropped_frames=camera_health.dropped_frames+excluded.dropped_frames",
                (camera_id, name, status, datetime.now(timezone.utc).isoformat(), error, dropped),
            )

    def record_unknown(self, event_id: str, decision: RecognitionDecision, relative_image_path: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO vote_sessions VALUES(?,?,?,?,?,?,?,?)",
                (
                    decision.observation_id,
                    "rejected",
                    None,
                    None,
                    self._scaled(decision.similarity),
                    json.dumps((decision.camera_id,)),
                    decision.reason,
                    decision.decided_at.isoformat(),
                ),
            )
            self._conn.execute(
                "INSERT INTO unknown_events(event_id,session_id,relative_image_path,reason,camera_ids_json,occurred_at) "
                "VALUES(?,?,?,?,?,?)",
                (
                    event_id,
                    decision.observation_id,
                    relative_image_path,
                    decision.reason,
                    json.dumps((decision.camera_id,)),
                    decision.decided_at.isoformat(),
                ),
            )

    def audit(self, actor: str, action: str, target: str | None = None, **details: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_log(actor,action,target,details_json,created_at) VALUES(?,?,?,?,?)",
                (actor, action, target, json.dumps(details, sort_keys=True), datetime.now(timezone.utc).isoformat()),
            )

    def prune_unknown(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86_400
        removed = 0
        for path in self.unknown_root.rglob("*.jpg"):
            if path.stat().st_mtime < cutoff:
                relative = path.relative_to(self.data_root).as_posix()
                path.unlink()
                with self._lock, self._conn:
                    self._conn.execute(
                        "UPDATE unknown_events SET relative_image_path=NULL WHERE relative_image_path=?", (relative,)
                    )
                removed += 1
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
