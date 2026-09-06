import numpy as np

from pi4_attendance.tools.benchmark import bbox_iou, verification_metrics, compare


def test_bbox_iou_matches_same_face_and_rejects_different_face() -> None:
    assert bbox_iou([0, 0, 20, 20], [2, 2, 18, 18]) == 0.64
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_verification_metrics_reports_far_frr_from_labeled_embeddings() -> None:
    embeddings = [
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [0.6, 0.8],
    ]
    metrics = verification_metrics(embeddings, ["alice", "alice", "bob", "bob"], 0.75)
    assert metrics["genuine_pairs"] == 2
    assert metrics["impostor_pairs"] == 4
    assert metrics["false_rejects"] == 0
    assert metrics["false_accepts"] == 1
    assert metrics["false_reject_rate"] == 0.0
    assert metrics["false_accept_rate"] == 0.25


def test_compare_uses_only_iou_matched_faces_for_embedding_agreement() -> None:
    reference = {
        "detections": 2,
        "median_ms": 20.0,
        "faces": [
            {"bbox": [0, 0, 20, 20], "embedding": [1.0, 0.0]},
            {"bbox": [0, 0, 10, 10], "embedding": [0.0, 1.0]},
        ],
    }
    candidate = {
        "detections": 2,
        "median_ms": 10.0,
        "faces": [
            {"bbox": [1, 1, 19, 19], "embedding": [1.0, 0.0]},
            {"bbox": [30, 30, 40, 40], "embedding": [1.0, 0.0]},
        ],
    }

    result = compare(reference, candidate)

    assert result["embedding_pairs"] == 1
    assert result["skipped_unmatched_faces"] == 1
    assert np.isclose(result["mean_embedding_cosine"], 1.0)
