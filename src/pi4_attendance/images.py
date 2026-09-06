from __future__ import annotations

import math
from numbers import Real

import numpy as np


MIN_FRAME_DIMENSION = 16
UNKNOWN_CROP_PADDING = 12
MIN_UNKNOWN_CROP_DIMENSION = 16
MAX_BBOX_COORDINATE = 1_000_000


def validate_frame(frame: np.ndarray, *, normalize_noncontiguous: bool = False) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError("Frame must use uint8 pixels")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a BGR HxWx3 array")
    height, width = frame.shape[:2]
    if height < MIN_FRAME_DIMENSION or width < MIN_FRAME_DIMENSION or frame.size == 0:
        raise ValueError(f"Frame dimensions must be at least {MIN_FRAME_DIMENSION}x{MIN_FRAME_DIMENSION}")
    if not frame.flags.c_contiguous:
        if not normalize_noncontiguous:
            raise ValueError("Frame must be contiguous")
        frame = np.ascontiguousarray(frame)
    return frame


def validate_bbox(bbox, *, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    try:
        if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise ValueError
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in bbox):
            raise ValueError
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounding box must contain four numeric coordinates") from exc
    if not all(math.isfinite(value) and abs(value) <= MAX_BBOX_COORDINATE for value in values):
        raise ValueError("bounding box coordinates must be finite and safe")
    x1, y1, x2, y2 = (int(value) for value in values)
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
        raise ValueError("bounding box has invalid geometry")
    return max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)


def crop_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    padding: int = UNKNOWN_CROP_PADDING,
) -> np.ndarray | None:
    frame = validate_frame(frame)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = validate_bbox(bbox, frame_width=width, frame_height=height)
    x1 = max(0, min(width, x1 - padding))
    y1 = max(0, min(height, y1 - padding))
    x2 = max(0, min(width, x2 + padding))
    y2 = max(0, min(height, y2 + padding))
    if x2 <= x1 or y2 <= y1:
        return None
    if x1 == 0 and y1 == 0 and x2 == width and y2 == height:
        return None
    crop = frame[y1:y2, x1:x2]
    if (
        crop.size == 0
        or crop.shape[0] < MIN_UNKNOWN_CROP_DIMENSION
        or crop.shape[1] < MIN_UNKNOWN_CROP_DIMENSION
    ):
        return None
    return np.ascontiguousarray(crop)
