from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, minimum_area: int) -> None:
        self.minimum_area = minimum_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=30, detectShadows=False)
        self.kernel = np.ones((3, 3), np.uint8)

    def has_activity(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = self.subtractor.apply(gray)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = (320 / frame.shape[1]) * (240 / frame.shape[0])
        threshold = max(100, self.minimum_area * scale)
        return any(cv2.contourArea(contour) >= threshold for contour in contours)

