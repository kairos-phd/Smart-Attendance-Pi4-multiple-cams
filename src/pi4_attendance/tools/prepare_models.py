from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_spec(model_path: Path) -> tuple[str, int, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    height = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else (112 if "w600k" in model_path.name else 320)
    width = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else height
    return model_input.name, height, width


def preprocess(image: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None, ...]


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path: Path, calibration_dir: Path, limit: int) -> None:
        name, height, width = input_spec(model_path)
        samples = []
        for path in sorted(calibration_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is not None:
                samples.append({name: preprocess(image, height, width)})
            if len(samples) >= limit:
                break
        if len(samples) < 20:
            raise ValueError("At least 20 representative calibration images are required")
        self._iterator = iter(samples)

    def get_next(self):
        return next(self._iterator, None)


def prepare(args: argparse.Namespace) -> None:
    root = args.model_root.resolve()
    source_dir = root / "models" / args.source_pack
    if not source_dir.exists():
        from insightface.app import FaceAnalysis

        # Construction performs InsightFace's standard model-pack acquisition.
        FaceAnalysis(name=args.source_pack, root=str(root), providers=["CPUExecutionProvider"])
    if not source_dir.exists():
        raise FileNotFoundError(f"InsightFace model pack was not found at {source_dir}")

    fp32_dir = root / "models" / args.fp32_pack
    int8_dir = root / "models" / args.int8_pack
    fp32_dir.mkdir(parents=True, exist_ok=True)
    if args.calibration_dir:
        int8_dir.mkdir(parents=True, exist_ok=True)

    fp32_files: list[dict] = []
    int8_files: list[dict] = []
    for source in sorted(source_dir.glob("*.onnx")):
        optimized = fp32_dir / source.name
        quant_pre_process(str(source), str(optimized), skip_symbolic_shape=True, skip_optimization=False)
        onnx.checker.check_model(onnx.load(str(optimized)))
        fp32_files.append({"name": optimized.name, "sha256": sha256(optimized), "bytes": optimized.stat().st_size})

        if args.calibration_dir:
            target = int8_dir / source.name
            # Quantize the CNN detector and recognizer; auxiliary landmark/attribute models stay optimized FP32.
            lower_name = source.stem.lower()
            is_primary = lower_name.startswith("det_") or "scrfd" in lower_name or "w600k" in lower_name
            if is_primary:
                reader = ImageCalibrationReader(optimized, args.calibration_dir.resolve(), args.calibration_limit)
                quantize_static(
                    str(optimized),
                    str(target),
                    reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                )
                onnx.checker.check_model(onnx.load(str(target)))
            else:
                shutil.copy2(optimized, target)
            int8_files.append({"name": target.name, "sha256": sha256(target), "bytes": target.stat().st_size})

    fp32_manifest = {
        "model_version": f"{args.source_pack}-ort-fp32-v1",
        "source_pack": args.source_pack,
        "format": "onnx-fp32-optimized",
        "files": fp32_files,
    }
    (fp32_dir / "manifest.json").write_text(json.dumps(fp32_manifest, indent=2), encoding="utf-8")
    if args.calibration_dir:
        int8_manifest = {
            "model_version": f"{args.source_pack}-ort-int8-v1",
            "source_pack": args.source_pack,
            "format": "onnx-qdq-int8",
            "calibration_images": str(args.calibration_dir.resolve()),
            "files": int8_files,
        }
        (int8_dir / "manifest.json").write_text(json.dumps(int8_manifest, indent=2), encoding="utf-8")
    print(f"Prepared optimized FP32 pack: {fp32_dir}")
    if args.calibration_dir:
        print(f"Prepared candidate INT8 pack: {int8_dir}")
        print("Run sic-pi4-benchmark on the Raspberry Pi before selecting the INT8 pack.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optimized InsightFace packs for Raspberry Pi 4")
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--source-pack", default="buffalo_sc")
    parser.add_argument("--fp32-pack", default="pi4_fp32")
    parser.add_argument("--int8-pack", default="pi4_int8")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-limit", type=int, default=200)
    parser.add_argument("--accept-insightface-noncommercial-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_insightface_noncommercial_license:
        parser.error("Review InsightFace model licensing and pass --accept-insightface-noncommercial-license")
    prepare(args)


if __name__ == "__main__":
    main()
