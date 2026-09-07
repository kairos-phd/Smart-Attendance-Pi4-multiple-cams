import tempfile
import unittest
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pi4_attendance.models import Observation, RecognitionDecision
from pi4_attendance.storage import PiStore


class StorageTests(unittest.TestCase):
    def test_non_finite_recognition_metrics_are_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            base = Observation(
                "camera-1", 1.0, datetime.now(timezone.utc), "u1", "User",
                0.8, 0.1, 0.9, (1, 2, 3, 4), 0.8, "model-v1",
            )
            try:
                for field in ("similarity", "margin", "detection_score", "quality"):
                    for value in (float("nan"), float("inf"), float("-inf")):
                        with self.subTest(field=field, value=value):
                            with self.assertRaisesRegex(ValueError, "finite"):
                                store.record_observation(replace(base, **{field: value}))
                count = store._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                store.close()

    def test_user_embedding_must_be_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            try:
                with self.assertRaisesRegex(ValueError, "non-zero"):
                    store.create_user("u1", "Test User", "S001", np.zeros(4, dtype=np.int8), 0.01, "model-v1")
            finally:
                store.close()

    def test_create_user_rejects_invalid_embedding_and_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            with self.assertRaises(ValueError):
                store.create_user("u1", "User", "S1", np.array([], dtype=np.int8), 0.1, "model-v1")
            with self.assertRaises(ValueError):
                store.create_user("u2", "User", "S2", np.array([1], dtype=np.int8), float("nan"), "model-v1")
            self.assertEqual(store.user_rows(), [])
            store.close()

    def test_status_and_audit_roll_back_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            store.create_user("u1", "User", "S1", np.array([1], dtype=np.int8), 0.1, "model-v1")
            with self.assertRaises(Exception):
                store.set_user_status_audited("u1", "inactive", object())
            self.assertEqual(store.user_rows()[0]["status"], "active")
            store.close()
    def test_users_are_stored_as_int8_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            embedding = np.array([1, -2, 3, -4], dtype=np.int8)
            store.create_user("u1", "Test User", "S001", embedding, 0.01, "model-v1")
            [user] = store.active_users()
            self.assertEqual(user.embedding_q.dtype, np.int8)
            self.assertEqual(user.embedding_q.tolist(), embedding.tolist())
            store.close()

    def test_attendance_cooldown_is_global_across_camera_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")
            observed_at = datetime.now(timezone.utc)
            first = RecognitionDecision("o1", "accepted", "u1", "Test User", "camera-1", observed_at, 0.8, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed")
            second = RecognitionDecision("o2", "accepted", "u1", "Test User", "camera-2", observed_at, 0.9, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed")
            self.assertTrue(store.record_decision(first, 300))
            self.assertFalse(store.record_decision(second, 300))
            store.close()

    def test_recent_attendance_is_a_readable_deduplicated_recognition_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            try:
                store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")
                decision = RecognitionDecision(
                    "o1", "accepted", "u1", "Test User", "camera-2", datetime.now(timezone.utc),
                    0.8123, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
                )

                self.assertTrue(store.record_decision(decision, 300))
                [event] = store.recent_attendance()

                self.assertEqual(event["student_id"], "S001")
                self.assertEqual(event["camera_ids"], ["camera-2"])
                self.assertEqual(event["similarity"], 0.8123)
            finally:
                store.close()

    def test_decision_result_distinguishes_recorded_cooldown_and_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            try:
                store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")
                observed_at = datetime.now(timezone.utc)
                first = RecognitionDecision(
                    "o1", "accepted", "u1", "Test User", "camera-1", observed_at,
                    0.8, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
                )
                second = replace(first, observation_id="o2", camera_id="camera-2")

                self.assertEqual(store.record_decision_result(first, 300), (True, "recorded"))
                self.assertEqual(store.record_decision_result(second, 300), (False, "cooldown"))
                store.set_user_status("u1", "inactive")
                self.assertEqual(
                    store.record_decision_result(replace(second, observation_id="o3"), 300),
                    (False, "inactive_user"),
                )
            finally:
                store.close()

    def test_inactive_user_cannot_create_attendance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")
            store.set_user_status("u1", "inactive")
            decision = RecognitionDecision(
                "o1", "accepted", "u1", "Test User", "camera-1", datetime.now(timezone.utc),
                0.8, 0.1, 0.9, 0.8, (1, 2, 3, 4), "model-v1", "thresholds_passed",
            )

            inserted = store.record_decision(decision, 300)
            attendance = store.recent_attendance()
            store.close()
            self.assertFalse(inserted)
            self.assertEqual(attendance, [])

    def test_observation_and_unknown_metadata_round_trip_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            decision = RecognitionDecision(
                "o-unknown", "rejected", None, None, "camera-3", datetime.now(timezone.utc),
                0.2, 0.01, 0.9, 0.7, (1, 2, 3, 4), "model-v1", "similarity_below_threshold"
            )
            self.assertFalse(store.record_decision(decision, 300))
            store.record_unknown("unknown-1", decision, "unknown/2026/08/16/example.jpg")
            [event] = store.recognition_event_rows()
            self.assertEqual(event["observation_id"], "o-unknown")
            self.assertEqual(event["camera_id"], "camera-3")
            self.assertEqual(event["model_version"], "model-v1")
            self.assertEqual(event["reason"], "similarity_below_threshold")
            store.close()

    def test_migration_preserves_existing_vote_and_attendance_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "smart_attendance.sqlite3"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta VALUES ('version', '1');
                CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT NOT NULL, student_id TEXT NOT NULL UNIQUE, embedding_int8 BLOB NOT NULL, embedding_dim INTEGER NOT NULL, embedding_scale REAL NOT NULL, model_version TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','inactive')), created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                INSERT INTO users VALUES ('u1','Legacy User','S001',X'0102',2,0.01,'model-v1','active','2026-08-01T00:00:00+00:00','2026-08-01T00:00:00+00:00');
                CREATE TABLE enrollment_images (image_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, relative_path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE observations (observation_id TEXT PRIMARY KEY, session_id TEXT, camera_id TEXT NOT NULL, observed_at TEXT NOT NULL, candidate_user_id TEXT, similarity_scaled INTEGER NOT NULL, margin_scaled INTEGER NOT NULL, detection_scaled INTEGER NOT NULL, quality_scaled INTEGER NOT NULL, bbox_json TEXT NOT NULL);
                CREATE TABLE vote_sessions (session_id TEXT PRIMARY KEY, status TEXT NOT NULL, user_id TEXT, name TEXT, confidence_scaled INTEGER NOT NULL, camera_ids_json TEXT NOT NULL, reason TEXT NOT NULL, decided_at TEXT NOT NULL);
                INSERT INTO vote_sessions VALUES ('legacy-session', 'confirmed', 'u1', 'Legacy User', 8000, '[\"camera-1\",\"camera-2\"]', '2_camera_consensus', '2026-08-01T00:00:00+00:00');
                CREATE TABLE attendance (attendance_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL UNIQUE REFERENCES vote_sessions(session_id), user_id TEXT NOT NULL REFERENCES users(user_id), name TEXT NOT NULL, confidence_scaled INTEGER NOT NULL, camera_ids_json TEXT NOT NULL, occurred_at TEXT NOT NULL, sync_state TEXT NOT NULL DEFAULT 'pending');
                INSERT INTO attendance(session_id,user_id,name,confidence_scaled,camera_ids_json,occurred_at) VALUES ('legacy-session','u1','Legacy User',8000,'[\"camera-1\",\"camera-2\"]','2026-08-01T00:00:00+00:00');
                CREATE TABLE camera_health (camera_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, last_seen TEXT NOT NULL, last_error TEXT, dropped_frames INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE unknown_events (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES vote_sessions(session_id), relative_image_path TEXT, reason TEXT NOT NULL, camera_ids_json TEXT NOT NULL, occurred_at TEXT NOT NULL, review_status TEXT NOT NULL DEFAULT 'pending');
                CREATE TABLE audit_log (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT, details_json TEXT NOT NULL, created_at TEXT NOT NULL);
                """
            )
            connection.commit()
            connection.close()

            store = PiStore(Path(directory))
            accepted = RecognitionDecision(
                "new-accepted", "accepted", "u1", "Legacy User", "camera-1",
                datetime(2026, 8, 27, tzinfo=timezone.utc), 0.8, 0.1, 0.9, 0.8,
                (1, 2, 3, 4), "model-v1", "thresholds_passed",
            )
            rejected = RecognitionDecision(
                "new-rejected", "rejected", None, None, "camera-2",
                datetime(2026, 8, 27, tzinfo=timezone.utc), 0.2, 0.01, 0.9, 0.7,
                (4, 5, 6, 7), "model-v1", "similarity_below_threshold",
            )
            self.assertTrue(store.record_decision(accepted, 300))
            self.assertFalse(store.record_decision(rejected, 300))
            store.record_unknown("new-unknown", rejected, None)

            rows = store.recent_attendance()
            self.assertEqual([row["session_id"] for row in rows], ["new-accepted", "legacy-session"])
            self.assertEqual(rows[1]["name"], "Legacy User")
            self.assertEqual(
                [row["observation_id"] for row in store.recognition_event_rows()],
                ["new-accepted", "new-rejected"],
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
