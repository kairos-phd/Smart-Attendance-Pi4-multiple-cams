from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .embedding_catalog import EmbeddingCatalog
from .models import Observation
from .images import validate_bbox, validate_frame
from .model_identity import model_identity
from .quantization import normalize_embedding, quantize_embedding

LOG = logging.getLogger(__name__)

def _output_quantization(path: Path, output_name: str) -> tuple[float, int]:
    """Read a scalar output quantizer from the ONNX graph instead of guessing it."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    producer = producers.get(output_name)
    visited: set[str] = set()
    while producer is not None and producer.op_type == "Identity":
        name = producer.input[0]
        if name in visited:
            raise ValueError("Cyclic recognizer output graph")
        visited.add(name)
        producer = producers.get(name)
    if producer is None or producer.op_type != "QuantizeLinear":
        raise ValueError("Quantized recognizer output must expose a QuantizeLinear output contract")
    scale_array = constants.get(producer.input[1])
    zero_array = constants.get(producer.input[2]) if len(producer.input) > 2 else np.array(0, np.uint8)
    if scale_array is None or zero_array is None or scale_array.size != 1 or zero_array.size != 1:
        raise ValueError("Recognizer requires constant scalar output scale and zero point")
    scale, zero = float(scale_array.reshape(-1)[0]), int(zero_array.reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Recognizer output scale must be positive and finite")
    return scale, zero


def _face_quality(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    area_score = min(
        1.0,
        (crop.shape[0] * crop.shape[1]) / max(1.0, width * height * 0.08),
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0)

    brightness = float(np.mean(gray))
    exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)

    return float(0.45 * area_score + 0.35 * sharpness + 0.20 * exposure)


class FaceEngine:
    def __init__(
        self,
        model_root: Path,
        model_pack: str,
        det_size: int,
        threshold: float,
        margin: float,
        recognizer_model: str | Path = "auto",
    ) -> None:
        self.model_root = model_root
        self.model_pack = model_pack
        self.det_size = det_size
        self.threshold = threshold
        self.margin = margin
        self.recognizer_model = recognizer_model

        self._app = None
        self._model_lock = threading.Lock()

        self._recognizer = None
        self._recognizer_input = None
        self._recognizer_output = None
        self._uses_external_recognizer = False
        self._output_scale = 1.0
        self._output_zero_point = 0
        self._output_type = "tensor(float)"
        self.model_info: dict = {}

        self.model_version = model_pack
        self.model_label = model_pack

    def start(self) -> None:
        selection = str(self.recognizer_model)
        legacy = self.model_root / "models" / "w600k_mbf_qdq_int8.onnx"
        if selection == "pack":
            recognizer_path = None
        elif selection == "auto":
            recognizer_path = legacy if legacy.is_file() else None
        else:
            recognizer_path = Path(selection).expanduser()
            if not recognizer_path.is_absolute():
                recognizer_path = self.model_root / recognizer_path
            if not recognizer_path.is_file():
                raise FileNotFoundError(f"External recognizer not found: {recognizer_path}")
        self._uses_external_recognizer = recognizer_path is not None
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=self.model_pack,
            root=str(self.model_root),
            allowed_modules=["detection"] if self._uses_external_recognizer else ["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )

        self._app.prepare(
            ctx_id=-1,
            det_size=(self.det_size, self.det_size),
        )

        if recognizer_path is None:
            loaded = self._app.models
            detector = loaded.get("detection")
            recognizer = loaded.get("recognition")
            if detector is None or recognizer is None:
                raise RuntimeError("Pack must load both detection and recognition models")
            files = [Path(detector.model_file), Path(recognizer.model_file)]
            contract = {
                "recognizer_source": "pack", "alignment": "insightface-arcface-112",
                "color": "RGB", "layout": "NCHW",
                "mean": float(recognizer.input_mean), "std": float(recognizer.input_std),
                "embedding": "l2-normalized-signed-int8",
            }
            identity = model_identity(self.model_root, self.model_pack, model_files=files, preprocessing=contract)
            self.model_label, self.model_version = identity.label, identity.compatibility_key
            self.model_info = {
                "recognizer_source": "pack", "selection": selection,
                "files": [str(p) for p in files], **contract,
            }
            LOG.info("Loaded pack recognizer %s (%s)", self.model_pack, self.model_version)
            return

        self._recognizer = ort.InferenceSession(
            str(recognizer_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self._recognizer.get_inputs()
        outputs = self._recognizer.get_outputs()

        if len(inputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 recognizer input, got {len(inputs)}"
            )

        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 recognizer output, got {len(outputs)}"
            )

        self._recognizer_input = inputs[0].name
        self._recognizer_output = outputs[0].name

        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        output_type = outputs[0].type

        LOG.info(
            "Loaded INT8 recognizer: %s input=%s shape=%s output=%s shape=%s",
            recognizer_path,
            self._recognizer_input,
            input_shape,
            self._recognizer_output,
            output_shape,
        )

        if inputs[0].type != "tensor(float)":
            raise RuntimeError(f"Recognizer input must be tensor(float), got {inputs[0].type}")
        if len(input_shape) != 4 or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(input_shape, (1, 3, 112, 112), strict=True)
        ):
            raise RuntimeError(f"Recognizer input must be NCHW 1x3x112x112, got {input_shape}")
        if output_type not in {"tensor(int8)", "tensor(uint8)", "tensor(float)"}:
            raise RuntimeError(
                f"Unsupported recognizer output type: {output_type}"
            )
        self._output_type = output_type
        if output_type != "tensor(float)":
            self._output_scale, self._output_zero_point = _output_quantization(recognizer_path, self._recognizer_output)
        else:
            self._output_scale, self._output_zero_point = 1.0, 0
        app_models = getattr(self._app, "models", {})
        detector = app_models.get("detection")
        detector_path = getattr(detector, "model_file", None)
        pack = self.model_root / "models" / self.model_pack
        detector_files = [Path(detector_path)] if detector_path else sorted(pack.glob("det*.onnx"))
        if not detector_files:
            raise RuntimeError("Cannot determine the actual detector file for model identity")
        files = detector_files + [recognizer_path]
        contract = {
            "recognizer_source": "external",
            "alignment": "insightface-arcface-112", "layout": "NCHW", "color": "RGB",
            "mean": 127.5, "std": 127.5, "output_type": self._output_type,
            "output_scale": self._output_scale, "output_zero_point": self._output_zero_point,
        }
        identity = model_identity(self.model_root, self.model_pack, model_files=files, preprocessing=contract)
        self.model_label, self.model_version = identity.label, identity.compatibility_key
        self.model_info = {"recognizer_source": "external", "selection": selection,
                           "files": [str(p) for p in files], **contract}

        LOG.info(
            "Loaded model pack %s (%s)",
            self.model_pack,
            self.model_version,
        )

    def _faces(self, image: np.ndarray):
        if self._app is None:
            raise RuntimeError("Face engine is not started")

        with self._model_lock:
            return self._app.get(image)

    def _embedding_int8(self, image: np.ndarray, face) -> tuple[np.ndarray, float]:
        if not getattr(self, "_uses_external_recognizer", False):
            return quantize_embedding(np.asarray(face.normed_embedding, dtype=np.float32))
        if self._recognizer is None:
            raise RuntimeError("External recognizer is not started")

        if not hasattr(face, "kps") or face.kps is None:
            raise RuntimeError("Face landmarks are required for ArcFace alignment")

        # InsightFace's own ArcFace alignment.
        from insightface.utils import face_align

        aligned = face_align.norm_crop(
            image,
            landmark=np.asarray(face.kps, dtype=np.float32),
            image_size=112,
            mode="arcface",
        )

        # InsightFace ArcFace models expect RGB CHW FP32,
        # normalized from [0,255] to approximately [-1,1].
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32)
        tensor = (tensor - 127.5) / 127.5
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        output = self._recognizer.run(
            [self._recognizer_output],
            {self._recognizer_input: tensor},
        )[0]

        raw = np.asarray(output).reshape(-1)

        if raw.shape[0] != 512:
            raise RuntimeError(
                f"Expected embedding of 512 values, got {raw.shape}"
            )
        if raw.dtype == np.int8 and self._output_zero_point == 0:
            return raw, self._output_scale
        embedding = (raw.astype(np.float32) - self._output_zero_point) * self._output_scale
        return quantize_embedding(normalize_embedding(embedding))

    def _embedding_float(self, image: np.ndarray, face) -> np.ndarray:
        embedding_q, scale = self._embedding_int8(image, face)

        embedding = embedding_q.astype(np.float32) * float(scale)

        return normalize_embedding(embedding)

    def analyze(self, image: np.ndarray) -> list[tuple[object, np.ndarray]]:
        """Benchmark the same alignment, recognizer and INT8 storage boundary as runtime."""
        image = validate_frame(image)
        return [(face, self._embedding_float(image, face)) for face in self._faces(image)]

    def enrollment_embedding(
        self,
        images: list[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        embeddings: list[np.ndarray] = []

        for image in images:
            image = validate_frame(image)

            faces = self._faces(image)

            if len(faces) != 1:
                raise ValueError(
                    "Each enrollment image must contain exactly one clear foreground face"
                )

            face = faces[0]

            if float(face.det_score) < 0.60:
                raise ValueError("Face detection confidence is too low")

            bbox = validate_bbox(
                face.bbox,
                frame_width=image.shape[1],
                frame_height=image.shape[0],
            )

            if _face_quality(image, bbox) < 0.35:
                raise ValueError(
                    "An enrollment image is too small, blurry, dark, or overexposed"
                )

            embeddings.append(self._embedding_float(image, face))

        if not 3 <= len(embeddings) <= 8:
            raise ValueError("Enrollment requires 3 to 8 images")

        reference = normalize_embedding(
            np.mean(np.stack(embeddings), axis=0)
        )

        consistency = min(
            float(np.dot(reference, item))
            for item in embeddings
        )

        if consistency < 0.55:
            raise ValueError(
                "Enrollment images do not appear to show the same person"
            )

        return quantize_embedding(reference)

    def recognize(
        self,
        image: np.ndarray,
        users: EmbeddingCatalog,
        camera_id: str,
        observed_monotonic: float,
        observed_at: datetime,
    ) -> list[Observation]:
        image = validate_frame(image)

        if not isinstance(users, EmbeddingCatalog):
            users = EmbeddingCatalog.from_users(
                users,
                self.model_version,
            )

        results: list[Observation] = []

        for face in self._faces(image):
            try:
                bbox = validate_bbox(
                    face.bbox,
                    frame_width=image.shape[1],
                    frame_height=image.shape[0],
                )
            except ValueError:
                continue

            try:
                query_q, query_scale = self._embedding_int8(
                    image,
                    face,
                )
            except Exception:
                LOG.exception("INT8 face embedding failed")
                continue

            best_user, best_score, second_score = users.top_two(
                query_q,
                query_scale,
            )

            match_margin = best_score - second_score

            results.append(
                Observation(
                    camera_id=camera_id,
                    observed_monotonic=observed_monotonic,
                    observed_at=observed_at,
                    candidate_user_id=best_user.user_id if best_user else None,
                    candidate_name=best_user.name if best_user else None,
                    similarity=float(best_score),
                    margin=float(match_margin),
                    detection_score=float(face.det_score),
                    bbox=bbox,
                    quality=_face_quality(image, bbox),
                    model_version=self.model_version,
                )
            )

        return results
