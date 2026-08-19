from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_images(directory: Path, limit: int) -> list[np.ndarray]:
    images = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
        if len(images) >= limit:
            break
    if len(images) < 20:
        raise ValueError("Benchmark dataset requires at least 20 images")
    return images


def run_pack(model_root: Path, pack: str, det_size: int, images: list[np.ndarray]) -> dict:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=pack, root=str(model_root), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    for image in images[:3]:
        app.get(image)
    latencies = []
    detections = 0
    embeddings: list[list[float] | None] = []
    for image in images:
        started = time.perf_counter()
        faces = app.get(image)
        latencies.append((time.perf_counter() - started) * 1000)
        detections += len(faces)
        if faces:
            face = max(faces, key=lambda item: float(item.det_score))
            embeddings.append(np.asarray(face.normed_embedding, dtype=np.float32).tolist())
        else:
            embeddings.append(None)
    ordered = sorted(latencies)
    return {
        "pack": pack,
        "images": len(images),
        "detections": detections,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "max_ms": max(latencies),
        "embeddings": embeddings,
    }


def compare(reference: dict, candidate: dict) -> dict:
    agreements = []
    for left, right in zip(reference["embeddings"], candidate["embeddings"], strict=True):
        if left is not None and right is not None:
            agreements.append(float(np.dot(np.asarray(left), np.asarray(right))))
    speedup = reference["median_ms"] / candidate["median_ms"] if candidate["median_ms"] else 0.0
    detection_delta = candidate["detections"] - reference["detections"]
    return {
        "embedding_pairs": len(agreements),
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
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    args = parser.parse_args()
    images = load_images(args.dataset, args.limit)
    reference = run_pack(args.model_root.resolve(), args.reference, args.det_size, images)
    candidate = run_pack(args.model_root.resolve(), args.candidate, args.det_size, images)
    result = {"reference": reference, "candidate": candidate, "comparison": compare(reference, candidate)}
    # Embeddings are used only for comparison and are not persisted in benchmark output.
    result["reference"].pop("embeddings")
    result["candidate"].pop("embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

