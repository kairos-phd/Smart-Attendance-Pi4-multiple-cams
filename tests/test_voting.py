from datetime import datetime, timezone
import unittest

from pi4_attendance.models import Observation
from pi4_attendance.voting import VoteCoordinator


def observation(camera: str, user: str | None, score: float, when: float, margin: float = 0.10) -> Observation:
    return Observation(
        camera_id=camera,
        observed_monotonic=when,
        observed_at=datetime.now(timezone.utc),
        candidate_user_id=user,
        candidate_name=f"Name {user}" if user else None,
        similarity=score,
        margin=margin,
        detection_score=0.9,
        bbox=(1, 2, 30, 40),
        quality=0.8,
    )


class VotingTests(unittest.TestCase):
    def test_two_distinct_cameras_confirm(self) -> None:
        votes = VoteCoordinator(2.0, 2, 0.45, 0.06)
        votes.add(observation("camera-1", "user-1", 0.70, 10.0))
        votes.add(observation("camera-2", "user-1", 0.66, 10.5))
        [decision] = votes.finalize_due(12.1)
        self.assertEqual(decision.status, "confirmed")
        self.assertEqual(decision.user_id, "user-1")
        self.assertEqual(decision.camera_ids, ("camera-1", "camera-2"))

    def test_duplicate_frames_from_one_camera_do_not_form_quorum(self) -> None:
        votes = VoteCoordinator(2.0, 2, 0.45, 0.06)
        votes.add(observation("camera-1", "user-1", 0.70, 10.0))
        votes.add(observation("camera-1", "user-1", 0.75, 10.5))
        [decision] = votes.finalize_due(12.1)
        self.assertEqual(decision.status, "rejected")

    def test_weak_or_ambiguous_match_does_not_vote(self) -> None:
        votes = VoteCoordinator(2.0, 2, 0.45, 0.06)
        votes.add(observation("camera-1", "user-1", 0.70, 10.0, margin=0.01))
        votes.add(observation("camera-2", "user-1", 0.40, 10.5))
        [decision] = votes.finalize_due(12.1)
        self.assertEqual(decision.status, "rejected")

    def test_two_conflicting_quorums_are_rejected(self) -> None:
        votes = VoteCoordinator(2.0, 2, 0.45, 0.06)
        for camera, user in (("camera-1", "user-1"), ("camera-2", "user-1"), ("camera-2", "user-2"), ("camera-3", "user-2")):
            votes.add(observation(camera, user, 0.7, 10.0))
        [decision] = votes.finalize_due(12.1)
        self.assertEqual(decision.status, "conflict")

    def test_new_window_does_not_discard_previous_window(self) -> None:
        votes = VoteCoordinator(2.0, 2, 0.45, 0.06)
        votes.add(observation("camera-1", "user-1", 0.70, 10.0))
        votes.add(observation("camera-2", "user-1", 0.68, 10.4))
        votes.add(observation("camera-1", "user-2", 0.72, 13.0))
        [previous] = votes.finalize_due(13.0)
        self.assertEqual(previous.status, "confirmed")
        self.assertEqual(previous.user_id, "user-1")


if __name__ == "__main__":
    unittest.main()
