from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.evaluation.threshold import (
    ThresholdError,
    apply_threshold,
    compute_f1_threshold,
    resolve_threshold,
    with_predictions,
)
from anomaly_detection.utils.config import ThresholdConfig


def prediction_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [f"{i}.png" for i in range(8)],
            "split": ["validation"] * 4 + ["test"] * 4,
            "ground_truth": ["good", "good", "defect", "defect"] * 2,
            # testの異常スコアを極端に低くしても、autoのThresholdへ影響しないことを確認する
            "anomaly_score": [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, -5.0, -6.0],
        }
    )


def test_apply_threshold_known_scores() -> None:
    result = apply_threshold(np.array([0.1, 0.49, 0.5, 0.9]), 0.5)
    assert result.tolist() == ["GOOD", "GOOD", "NG", "NG"]  # 境界値はNG


def test_f1_threshold_separates_classes() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    threshold = compute_f1_threshold(scores, labels)
    assert 0.3 < threshold <= 0.8
    assert apply_threshold(scores, threshold).tolist() == ["GOOD", "GOOD", "GOOD", "NG", "NG"]


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_f1_threshold_requires_both_classes(labels: list[int]) -> None:
    with pytest.raises(ThresholdError, match="正常画像と異常画像の両方"):
        compute_f1_threshold(np.array([0.1, 0.2, 0.3]), np.array(labels))


def test_f1_threshold_requires_images() -> None:
    with pytest.raises(ThresholdError, match="0件"):
        compute_f1_threshold(np.array([]), np.array([]))


def test_auto_uses_validation_only() -> None:
    table = prediction_table()
    result = resolve_threshold(ThresholdConfig(mode="auto"), table)
    validation = table[table["split"] == "validation"]
    expected = compute_f1_threshold(validation["anomaly_score"].to_numpy(), np.array([0, 0, 1, 1]))
    assert result.value == pytest.approx(expected)
    assert result.value > 0.2
    assert result.source == "validation"
    assert result.num_images == 4


def test_auto_fails_without_validation() -> None:
    table = prediction_table()
    with pytest.raises(ThresholdError):
        resolve_threshold(ThresholdConfig(mode="auto"), table[table["split"] == "test"])


def test_manual_uses_config_value() -> None:
    result = resolve_threshold(
        ThresholdConfig(mode="manual", manual_value=0.65), prediction_table()
    )
    assert result.value == 0.65
    assert result.mode == "manual"
    assert result.source == "manual"
    assert result.num_images == 0


def test_with_predictions_adds_columns_without_modifying_input() -> None:
    table = prediction_table()
    out = with_predictions(table, 0.5)
    assert "prediction" not in table.columns
    assert (out["threshold"] == 0.5).all()
    assert out["prediction"].tolist()[:4] == ["GOOD", "GOOD", "NG", "NG"]
