from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from conftest import write_png
from matplotlib.figure import Figure

from anomaly_detection.evaluation.error_analysis import (
    get_errors,
    get_false_negatives,
    get_false_positives,
    plot_errors,
)
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.visualization.anomaly_visualizer import HeatmapScale


def prediction_table(tmp_path: Path) -> pd.DataFrame:
    rows = [
        ("validation", "good", "NG", 0.9),  # validationの誤判定は対象外
        ("test", "good", "NG", 0.7),  # FP
        ("test", "good", "NG", 0.95),  # FP
        ("test", "good", "GOOD", 0.1),
        ("test", "defect", "GOOD", 0.3),  # FN
        ("test", "defect", "GOOD", 0.2),  # FN
        ("test", "defect", "NG", 0.8),
    ]
    records = []
    for i, (split, gt, pred, score) in enumerate(rows):
        path = tmp_path / f"{i}.png"
        write_png(path, np.full((20, 30, 3), 100, np.uint8))
        records.append(
            {
                "image_path": str(path),
                "split": split,
                "ground_truth": gt,
                "defect_type": "",
                "object_id": "",
                "anomaly_score": score,
                "threshold": 0.5,
                "prediction": pred,
            }
        )
    return pd.DataFrame(records)


def test_false_positives_sorted_by_score_desc(tmp_path: Path) -> None:
    fp = get_false_positives(prediction_table(tmp_path))
    assert fp["anomaly_score"].tolist() == [0.95, 0.7]
    assert (fp["split"] == "test").all()


def test_false_negatives_sorted_by_score_asc(tmp_path: Path) -> None:
    fn = get_false_negatives(prediction_table(tmp_path))
    assert fn["anomaly_score"].tolist() == [0.2, 0.3]


def test_errors_fn_first(tmp_path: Path) -> None:
    table = prediction_table(tmp_path)
    assert get_errors(table)["error_type"].tolist() == ["FN", "FN", "FP", "FP"]
    assert get_errors(table, fn_first=False)["error_type"].tolist() == ["FP", "FP", "FN", "FN"]


def test_plot_errors_uses_matching_anomaly_maps(tmp_path: Path, small_config) -> None:
    table = prediction_table(tmp_path)
    maps = np.random.default_rng(0).random((len(table), 16, 24)).astype(np.float32)
    errors = get_errors(table)
    figures = plot_errors(
        errors,
        maps,
        ImagePreprocessor(small_config.preprocessing),
        HeatmapScale.from_maps(maps),
        max_items=3,
    )
    assert len(figures) == 3
    assert all(isinstance(f, Figure) for f in figures)
    assert figures[0].get_suptitle().startswith("[FN]")
    assert (
        len(
            plot_errors(
                errors,
                maps,
                ImagePreprocessor(small_config.preprocessing),
                HeatmapScale.from_maps(maps),
                max_items=None,
            )
        )
        == 4
    )
