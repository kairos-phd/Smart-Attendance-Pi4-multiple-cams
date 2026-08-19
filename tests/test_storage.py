import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pi4_attendance.models import Observation, VoteDecision
from pi4_attendance.storage import PiStore


class StorageTests(unittest.TestCase):
    def test_users_are_stored_as_int8_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            embedding = np.array([1, -2, 3, -4], dtype=np.int8)
            store.create_user("u1", "Test User", "S001", embedding, 0.01, "model-v1")
            [user] = store.active_users()
            self.assertEqual(user.embedding_q.dtype, np.int8)
            self.assertEqual(user.embedding_q.tolist(), embedding.tolist())
            store.close()

    def test_attendance_cooldown_is_global_after_camera_vote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            store.create_user("u1", "Test User", "S001", np.array([1, 2], dtype=np.int8), 0.01, "model-v1")
            first = VoteDecision("s1", "confirmed", "u1", "Test User", 0.8, ("camera-1", "camera-2"), "2_camera_consensus")
            second = VoteDecision("s2", "confirmed", "u1", "Test User", 0.9, ("camera-2", "camera-3"), "2_camera_consensus")
            self.assertTrue(store.record_decision(first, 300))
            self.assertFalse(store.record_decision(second, 300))
            store.close()

    def test_observation_and_unknown_metadata_round_trip_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PiStore(Path(directory))
            item = Observation(
                "camera-3", 1.0, datetime.now(timezone.utc), None, None, 0.2, 0.01, 0.9, (1, 2, 3, 4), 0.7
            )
            store.record_observation(item, "s-unknown")
            decision = VoteDecision("s-unknown", "rejected", None, None, 0.0, ("camera-3",), "insufficient_cross_camera_agreement")
            self.assertFalse(store.record_decision(decision, 300))
            store.record_unknown("unknown-1", decision, "unknown/2026/08/16/example.jpg")
            store.close()


if __name__ == "__main__":
    unittest.main()
