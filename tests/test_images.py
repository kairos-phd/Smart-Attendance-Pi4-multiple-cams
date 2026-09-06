import numpy as np
import pytest

from pi4_attendance import images as image_tools
from pi4_attendance.images import crop_face, validate_frame


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.array([], dtype=np.uint8),
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 4), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float32),
        np.zeros((15, 32, 3), dtype=np.uint8),
    ],
)
def test_validate_frame_rejects_invalid_bgr_inputs(frame) -> None:
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_validate_frame_rejects_or_normalizes_noncontiguous_input_explicitly() -> None:
    frame = np.zeros((32, 64, 3), dtype=np.uint8)[:, ::2]
    assert not frame.flags.c_contiguous

    with pytest.raises(ValueError, match="contiguous"):
        validate_frame(frame)

    normalized = validate_frame(frame, normalize_noncontiguous=True)
    assert normalized.flags.c_contiguous
    assert normalized.shape == frame.shape


def test_crop_face_clamps_bbox_and_adds_fixed_padding() -> None:
    frame = np.arange(40 * 50 * 3, dtype=np.uint8).reshape(40, 50, 3)

    crop = crop_face(frame, (-5, 5, 45, 50), padding=4)

    assert crop is not None
    assert crop.shape == (39, 49, 3)
    assert np.array_equal(crop, frame[1:40, 0:49])


def test_crop_face_rejects_empty_or_tiny_clamped_crop() -> None:
    frame = np.zeros((40, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="bounding box"):
        crop_face(frame, (60, 60, 80, 80))
    assert crop_face(frame, (1, 1, 3, 3), padding=0) is None


def test_crop_face_rejects_bbox_that_would_retain_the_full_frame() -> None:
    frame = np.zeros((40, 50, 3), dtype=np.uint8)

    assert crop_face(frame, (-100, -100, 200, 200)) is None


def test_crop_face_keeps_valid_close_face_without_returning_full_frame() -> None:
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    crop = crop_face(frame, (10, 10, 30, 30))

    assert crop is not None
    assert crop.shape == (42, 42, 3)
    assert crop.shape != frame.shape


@pytest.mark.parametrize(
    "bbox",
    [
        (20, 10, 10, 30),
        (10, 10, 10, 30),
        (10, 10, 30, 10),
        (10, 10, float("nan"), 30),
        (10, 10, float("inf"), 30),
        (10, 10, float("-inf"), 30),
        (10, 10, 30),
        "10,10,30,30",
        None,
        (10, 10, "30", 30),
        (0, 0, 10**20, 30),
        (60, 60, 80, 80),
    ],
)
def test_validate_bbox_rejects_invalid_detector_geometry(bbox) -> None:
    with pytest.raises(ValueError, match="bounding box"):
        image_tools.validate_bbox(bbox, frame_width=50, frame_height=40)


def test_reversed_bbox_is_rejected_before_padding_can_rescue_it() -> None:
    frame = np.zeros((40, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="bounding box"):
        crop_face(frame, (20, 10, 10, 30), padding=12)


def test_validate_bbox_clamps_reasonable_partial_detector_geometry() -> None:
    assert image_tools.validate_bbox((-5.8, 5.2, 45.9, 50.1), frame_width=50, frame_height=40) == (0, 5, 45, 40)
