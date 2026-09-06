from pathlib import Path

import cv2
import numpy as np

from pi4_attendance.tools import prepare_models
from pi4_attendance.tools.prepare_models import model_role, should_quantize


def test_only_detector_uses_full_frame_calibration_images() -> None:
    assert model_role(Path("det_500m.onnx")) == "detector"
    assert model_role(Path("scrfd_2.5g.onnx")) == "detector"
    assert model_role(Path("w600k_mbf.onnx")) == "recognizer"
    assert should_quantize(Path("det_500m.onnx"))
    assert not should_quantize(Path("w600k_mbf.onnx"))


def test_detector_preprocess_matches_scrfd_runtime_for_nonsquare_image() -> None:
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]],
        ],
        dtype=np.uint8,
    )
    canvas = np.zeros((8, 8, 3), dtype=np.uint8)
    canvas[:4, :8] = cv2.resize(image, (8, 4))
    runtime_reference = cv2.dnn.blobFromImage(
        canvas,
        1.0 / 128.0,
        (8, 8),
        (127.5, 127.5, 127.5),
        swapRB=True,
    )

    calibration = prepare_models.detector_preprocess(image, height=8, width=8)

    np.testing.assert_array_equal(calibration, runtime_reference)
    np.testing.assert_array_equal(calibration[0, :, 4:, :], np.full((3, 4, 8), -127.5 / 128.0, np.float32))
    np.testing.assert_array_equal(
        calibration[0, :, 0, 0],
        np.array([(30 - 127.5) / 128.0, (20 - 127.5) / 128.0, (10 - 127.5) / 128.0], np.float32),
    )
