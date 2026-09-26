from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.experiments.run_manager import (
    SUMMARY_COLUMNS,
    Run,
    RunNotFoundError,
    append_summary,
    load_run,
    load_summary,
)

NOW = datetime(2026, 9, 26, 16, 30, 0)


def test_run_id_and_collision(tmp_path: Path) -> None:
    first = Run.create(tmp_path, "phase1", now=NOW)
    second = Run.create(tmp_path, "phase1", now=NOW)
    assert first.run_id == "20260926_163000_phase1"
    assert second.run_id == "20260926_163000_phase1_2"
    assert first.run_dir.is_dir() and second.run_dir.is_dir()


def test_save_and_load_run(tmp_path: Path, small_config) -> None:
    run = Run.create(tmp_path, "phase1", now=NOW)
    run.save_config(small_config)
    run.save_metadata({"device": {"device": "cpu"}})
    run.save_metrics({"image": {"recall": 1.0}})
    predictions = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png"],
            "split": ["test", "test"],
            "ground_truth": ["good", "defect"],
            "defect_type": ["", "crack"],
            "object_id": ["", ""],
            "anomaly_score": [0.123456789, 2.5],
            "threshold": [1.0, 1.0],
            "prediction": ["GOOD", "NG"],
        }
    )
    run.save_predictions(predictions)
    maps = np.random.default_rng(0).random((2, 4, 4)).astype(np.float32)
    masks = np.zeros((2, 4, 4), np.uint8)
    run.save_anomaly_maps(maps, masks)

    loaded = load_run(run.run_dir)
    assert loaded.config.to_dict() == small_config.to_dict()
    assert loaded.metadata["device"]["device"] == "cpu"
    assert loaded.metrics["image"]["recall"] == 1.0
    pd.testing.assert_frame_equal(loaded.predictions, predictions)
    np.testing.assert_allclose(loaded.anomaly_maps, maps, atol=1e-3)  # float16で保存
    assert loaded.gt_masks is not None


def test_load_run_missing(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError, match="存在しません"):
        load_run(tmp_path / "nothing")
    run = Run.create(tmp_path, "phase1", now=NOW)
    with pytest.raises(RunNotFoundError, match="predictions.csv"):
        load_run(run.run_dir)


def test_summary_append(tmp_path: Path) -> None:
    assert load_summary(tmp_path).empty
    append_summary(tmp_path, {"run_id": "r1", "recall": 0.8, "unknown_key": "ignored"})
    append_summary(tmp_path, {"run_id": "r2", "recall": 0.9})
    summary = load_summary(tmp_path)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary["run_id"].tolist() == ["r1", "r2"]
    assert summary["recall"].tolist() == [0.8, 0.9]
