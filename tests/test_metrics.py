from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
    confusion_matrix_frame,
    evaluate_split,
)


def test_known_image_metrics() -> None:
    # TP=2, FN=1, FP=1, TN=2
    ground_truth = np.array(["defect", "defect", "defect", "good", "good", "good"])
    predictions = np.array(["NG", "NG", "GOOD", "NG", "GOOD", "GOOD"])
    scores = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    m = compute_image_metrics(ground_truth, predictions, scores)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["auroc"] == pytest.approx(8 / 9)
    assert m["num_images"] == 6
    assert (m["num_good"], m["num_defect"]) == (3, 3)


def test_perfect_prediction() -> None:
    m = compute_image_metrics(
        np.array(["good", "defect"]), np.array(["GOOD", "NG"]), np.array([0.1, 0.9])
    )
    assert m["recall"] == m["precision"] == m["f1"] == m["accuracy"] == 1.0
    assert m["auroc"] == m["aupr"] == 1.0


def test_single_class_has_no_auroc() -> None:
    m = compute_image_metrics(
        np.array(["good", "good"]), np.array(["GOOD", "NG"]), np.array([0.1, 0.9])
    )
    assert m["auroc"] is None
    assert m["aupr"] is None
    assert m["recall"] == 0.0  # zero_division=0


def test_confusion_matrix_frame() -> None:
    frame = confusion_matrix_frame({"tn": 5, "fp": 1, "fn": 2, "tp": 7})
    assert frame.loc["defect", "GOOD"] == 2  # 見逃し
    assert frame.loc["good", "NG"] == 1  # 過検知


def test_pixel_metrics_perfect_localization() -> None:
    masks = np.zeros((3, 16, 16), np.uint8)
    masks[1:, 4:8, 4:8] = 1
    maps = masks.astype(np.float32) + 0.1
    m = compute_pixel_metrics(maps, masks)
    assert m["pixel_auroc"] == pytest.approx(1.0)
    assert m["pixel_aupro"] == pytest.approx(1.0)


def test_pixel_metrics_without_defect_pixels() -> None:
    m = compute_pixel_metrics(np.zeros((2, 8, 8), np.float32), np.zeros((2, 8, 8), np.uint8))
    assert m["pixel_auroc"] is None


def test_evaluate_split_uses_only_that_split() -> None:
    table = pd.DataFrame(
        {
            "split": ["validation", "test", "test"],
            "ground_truth": ["defect", "good", "defect"],
            "prediction": ["GOOD", "GOOD", "NG"],
            "anomaly_score": [0.1, 0.2, 0.9],
        }
    )
    maps = np.zeros((3, 4, 4), np.float32)
    masks = np.zeros((3, 4, 4), np.uint8)
    masks[2, 1:3, 1:3] = 1
    maps[2, 1:3, 1:3] = 1.0
    result = evaluate_split(table, "test", maps, masks)
    assert result["image"]["num_images"] == 2
    assert result["image"]["fn"] == 0
    assert result["pixel"]["pixel_auroc"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="split=train"):
        evaluate_split(table, "train")
