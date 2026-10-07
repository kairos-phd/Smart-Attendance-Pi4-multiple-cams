from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MIN_FACE_IOU = 0.50


@dataclass(frozen=True)
class BenchmarkSample:
    image: np.ndarray
    label: str | None


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def verification_metrics(
    embeddings: list[list[float] | None], labels: list[str | None], threshold: float
) -> dict:
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    for left_index, (left, left_label) in enumerate(zip(embeddings, labels, strict=True)):
        if left is None or left_label is None:
            continue
        left_vector = np.asarray(left, dtype=np.float32)
        for right, right_label in zip(embeddings[left_index + 1 :], labels[left_index + 1 :], strict=True):
            if right is None or right_label is None:
                continue
            score = float(np.dot(left_vector, np.asarray(right, dtype=np.float32)))
            (genuine_scores if left_label == right_label else impostor_scores).append(score)
    false_rejects = sum(score < threshold for score in genuine_scores)
    false_accepts = sum(score >= threshold for score in impostor_scores)
    return {
        "threshold": threshold,
        "genuine_pairs": len(genuine_scores),
        "impostor_pairs": len(impostor_scores),
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "false_reject_rate": false_rejects / len(genuine_scores) if genuine_scores else None,
        "false_accept_rate": false_accepts / len(impostor_scores) if impostor_scores else None,
    }


def load_images(directory: Path, limit: int) -> list[BenchmarkSample]:
    samples = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is not None:
            relative = path.relative_to(directory)
            label = relative.parts[0] if len(relative.parts) > 1 else None
            samples.append(BenchmarkSample(image, label))
        if len(samples) >= limit:
            break
    if len(samples) < 20:
        raise ValueError("Benchmark dataset requires at least 20 images")
    return samples


def run_pack(
    model_root: Path, pack: str, det_size: int, samples: list[BenchmarkSample], match_threshold: float,
    *, recognizer_model: str | Path = "pack",
) -> dict:
    from ..face_engine import FaceEngine

    engine = FaceEngine(model_root, pack, det_size, match_threshold, 0.0,
                        recognizer_model=recognizer_model)
    engine.start()
    for sample in samples[:3]:
        engine.analyze(sample.image)
    latencies = []
    detections = 0
    faces_by_sample: list[dict | None] = []
    embeddings: list[list[float] | None] = []
    labels = [sample.label for sample in samples]
    for sample in samples:
        started = time.perf_counter()
        faces = engine.analyze(sample.image)
        latencies.append((time.perf_counter() - started) * 1000)
        detections += len(faces)
        if faces:
            face, vector = max(faces, key=lambda item: float(item[0].det_score))
            embedding = np.asarray(vector, dtype=np.float32).tolist()
            embeddings.append(embedding)
            faces_by_sample.append({"bbox": np.asarray(face.bbox, dtype=np.float32).tolist(), "embedding": embedding})
        else:
            embeddings.append(None)
            faces_by_sample.append(None)
    ordered = sorted(latencies)
    return {
        "pack": pack,
        "model_version": engine.model_version,
        "model_info": engine.model_info,
        "images_without_faces": sum(value is None for value in embeddings),
        "images": len(samples),
        "detections": detections,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "max_ms": max(latencies),
        "faces": faces_by_sample,
        "verification": verification_metrics(embeddings, labels, match_threshold),
    }


def compare(reference: dict, candidate: dict) -> dict:
    agreements = []
    comparable = 0
    for left, right in zip(reference["faces"], candidate["faces"], strict=True):
        if left is not None and right is not None:
            comparable += 1
            if bbox_iou(left["bbox"], right["bbox"]) >= MIN_FACE_IOU:
                agreements.append(float(np.dot(np.asarray(left["embedding"]), np.asarray(right["embedding"]))))
    speedup = reference["median_ms"] / candidate["median_ms"] if candidate["median_ms"] else 0.0
    detection_delta = candidate["detections"] - reference["detections"]
    return {
        "embedding_pairs": len(agreements),
        "skipped_unmatched_faces": comparable - len(agreements),
        "mean_embedding_cosine": statistics.mean(agreements) if agreements else 0.0,
        "minimum_embedding_cosine": min(agreements) if agreements else 0.0,
        "detection_count_delta": detection_delta,
        "median_speedup": speedup,
        "candidate_passes_default_gate": bool(
            agreements
            and statistics.mean(agreements) >= 0.98
            and detection_delta >= -max(1, round(reference["detections"] * 0.01))
            and speedup >= 1.05
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 and INT8 face packs on the target Pi")
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", default="pi4_fp32")
    parser.add_argument("--candidate", default="pi4_int8")
    parser.add_argument("--reference-recognizer", default="pack", help="pack, auto, or external ONNX path")
    parser.add_argument("--candidate-recognizer", default="pack", help="pack, auto, or external ONNX path")
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    args = parser.parse_args()
    samples = load_images(args.dataset, args.limit)
    reference = run_pack(args.model_root.resolve(), args.reference, args.det_size, samples, args.match_threshold,
                         recognizer_model=args.reference_recognizer)
    candidate = run_pack(args.model_root.resolve(), args.candidate, args.det_size, samples, args.match_threshold,
                         recognizer_model=args.candidate_recognizer)
    result = {"reference": reference, "candidate": candidate, "comparison": compare(reference, candidate)}
    result["reference"].pop("faces")
    result["candidate"].pop("faces")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
