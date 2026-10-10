import json
import sys
from types import SimpleNamespace
from pathlib import Path

from pi4_attendance.face_engine import FaceEngine
from pi4_attendance.model_identity import model_identity


def test_model_identity_changes_with_onnx_bytes_and_preprocessing_contract(tmp_path: Path) -> None:
    pack = tmp_path / "models" / "pack"
    pack.mkdir(parents=True)
    model = pack / "recognizer.onnx"
    model.write_bytes(b"weights-a")
    (pack / "manifest.json").write_text(json.dumps({"model_version": "friendly", "preprocessing": {"rgb": True}}))
    first = model_identity(tmp_path, "pack")
    model.write_bytes(b"weights-b")
    second = model_identity(tmp_path, "pack")
    (pack / "manifest.json").write_text(json.dumps({"model_version": "friendly", "preprocessing": {"rgb": False}}))
    third = model_identity(tmp_path, "pack")
    assert first.label == second.label == third.label == "friendly"
    assert len({first.compatibility_key, second.compatibility_key, third.compatibility_key}) == 3


def test_face_engine_allows_insightface_to_acquire_pack_before_fingerprinting(tmp_path: Path, monkeypatch) -> None:
    constructed = False

    class FakeFaceAnalysis:
        def __init__(self, **_: object) -> None:
            nonlocal constructed
            constructed = True
            pack = tmp_path / "models" / "pack"
            pack.mkdir(parents=True)
            detector, recognizer = pack / "det.onnx", pack / "rec.onnx"
            detector.write_bytes(b"detector")
            recognizer.write_bytes(b"recognizer")
            self.models = {"detection": SimpleNamespace(model_file=str(detector)),
                           "recognition": SimpleNamespace(model_file=str(recognizer), input_mean=127.5, input_std=127.5)}

        def prepare(self, **_: object) -> None:
            return None

    def fingerprint_after_construction(*_: object, **kwargs: object) -> SimpleNamespace:
        assert constructed
        return SimpleNamespace(label="friendly", compatibility_key="friendly@sha256:abc")

    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=FakeFaceAnalysis))
    monkeypatch.setattr("pi4_attendance.face_engine.model_identity", fingerprint_after_construction)
    engine = FaceEngine(tmp_path, "pack", 320, 0.45, 0.06)

    engine.start()

    assert engine.model_version == "friendly@sha256:abc"
