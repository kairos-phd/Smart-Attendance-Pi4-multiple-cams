import unittest
from datetime import datetime, timezone

import numpy as np

from pi4_attendance.frame_queue import FairFrameQueue
from pi4_attendance.models import FrameCandidate


def frame(camera: str, marker: int) -> FrameCandidate:
    return FrameCandidate(camera, float(marker), datetime.now(timezone.utc), np.array([[marker]], dtype=np.uint8))


class FairFrameQueueTests(unittest.TestCase):
    def test_round_robin_prevents_busy_camera_starvation(self) -> None:
        frames = FairFrameQueue(["c1", "c2", "c3"], 6)
        frames.put(frame("c1", 1))
        frames.put(frame("c1", 2))
        frames.put(frame("c2", 3))
        frames.put(frame("c3", 4))
        self.assertEqual([frames.get().camera_id for _ in range(3)], ["c1", "c2", "c3"])

    def test_pressure_replaces_only_same_camera_oldest(self) -> None:
        frames = FairFrameQueue(["c1", "c2", "c3"], 3)
        self.assertFalse(frames.put(frame("c1", 1)))
        self.assertTrue(frames.put(frame("c1", 2)))
        frames.put(frame("c2", 3))
        self.assertEqual(int(frames.get().frame[0, 0]), 2)
        self.assertEqual(frames.get().camera_id, "c2")

    def test_non_divisible_capacity_is_distributed_without_exceeding_total(self) -> None:
        frames = FairFrameQueue(["c1", "c2", "c3"], 5)

        for marker in range(3):
            frames.put(frame("c1", marker))
            frames.put(frame("c2", marker + 10))
            frames.put(frame("c3", marker + 20))

        self.assertEqual(frames.qsize(), 5)
        drained = [frames.get() for _ in range(5)]
        self.assertEqual([item.camera_id for item in drained].count("c1"), 2)
        self.assertEqual([item.camera_id for item in drained].count("c2"), 2)
        self.assertEqual([item.camera_id for item in drained].count("c3"), 1)


if __name__ == "__main__":
    unittest.main()
