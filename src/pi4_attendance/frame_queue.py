from __future__ import annotations

import queue
import threading
import time
from collections import deque

from .models import FrameCandidate


class FairFrameQueue:
    """Bounded per-camera queues with round-robin dequeue to prevent camera starvation."""

    def __init__(self, camera_ids: list[str], total_capacity: int) -> None:
        if not camera_ids:
            raise ValueError("At least one camera is required")
        if total_capacity < len(camera_ids):
            raise ValueError("Total capacity must provide at least one slot per camera")
        self._order = list(camera_ids)
        base_capacity, extra_slots = divmod(total_capacity, len(camera_ids))
        self._capacities = {
            camera_id: base_capacity + (1 if index < extra_slots else 0)
            for index, camera_id in enumerate(camera_ids)
        }
        self._queues = {camera_id: deque() for camera_id in camera_ids}
        self._next = 0
        self._condition = threading.Condition()

    def put(self, candidate: FrameCandidate) -> bool:
        """Add a frame and return True if an older frame from that camera was replaced."""
        with self._condition:
            target = self._queues.get(candidate.camera_id)
            if target is None:
                raise ValueError(f"Unknown camera: {candidate.camera_id}")
            dropped = len(target) >= self._capacities[candidate.camera_id]
            if dropped:
                target.popleft()
            target.append(candidate)
            self._condition.notify()
            return dropped

    def get(self, timeout: float | None = None) -> FrameCandidate:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not any(self._queues.values()):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            for offset in range(len(self._order)):
                index = (self._next + offset) % len(self._order)
                camera_id = self._order[index]
                if self._queues[camera_id]:
                    self._next = (index + 1) % len(self._order)
                    return self._queues[camera_id].popleft()
        raise queue.Empty

    def qsize(self) -> int:
        with self._condition:
            return sum(len(items) for items in self._queues.values())

