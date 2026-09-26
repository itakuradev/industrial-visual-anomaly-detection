"""合成Dataset・小さなModel（CPU）で run_experiment / reevaluate の保存内容を確認する。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.datasets.base import DatasetNotFoundError
from anomaly_detection.evaluation.threshold import compute_f1_threshold
from anomaly_detection.experiments.pipeline import reevaluate, run_experiment
from anomaly_detection.experiments.run_manager import load_summary


def test_run_experiment_and_reevaluate(small_config) -> None:
    result = run_experiment(small_config, pre_trained=False)
    run_dir = result.run.run_dir
    for name in (
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "samples.csv",
        "predictions.csv",
        "anomaly_maps.npz",
        "run.log",
        "model/model.ckpt",
    ):
        assert (run_dir / name).is_file(), name
    assert len(list((run_dir / "images" / "overlays").glob("*.png"))) == 12
    assert len(list((run_dir / "images" / "heatmaps").glob("*.png"))) == 12

    # Thresholdはvalidationだけから決まり、最終評価はtestのみ
    predictions = result.predictions
    validation = predictions[predictions["split"] == "validation"]
    expected = compute_f1_threshold(
        validation["anomaly_score"].to_numpy(),
        (validation["ground_truth"] == "defect").to_numpy(dtype=np.int64),
    )
    assert result.threshold.value == pytest.approx(expected)
    assert result.metrics["evaluation_split"] == "test"
    assert result.metrics["image"]["num_images"] == 6
    assert result.metrics["pixel"]["pixel_auroc"] is not None
    assert result.metrics["threshold_selection_reference"]["split"] == "validation"
    assert set(predictions["prediction"]) <= {"GOOD", "NG"}

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["device"]["device"] == "cpu"
    assert metadata["retrained"] is True
    assert metadata["environment"]["packages"]["anomalib"]
    assert metadata["dataset"]["counts"]["test"] == {"good": 3, "defect": 3}
    assert "Training開始" in (run_dir / "run.log").read_text(encoding="utf-8")

    # 閾値だけを変えて再評価（再学習なし）
    threshold = float(predictions["anomaly_score"].max()) + 1.0
    re = reevaluate(run_dir, threshold)
    assert re.run.run_dir != run_dir
    assert not (re.run.run_dir / "model").exists()
    assert (re.predictions["prediction"] == "GOOD").all()
    assert re.metrics["image"]["recall"] == 0.0
    assert re.metrics["notes"]  # manual閾値の注意書き
    re_metadata = json.loads((re.run.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert re_metadata["retrained"] is False
    assert re_metadata["parent_run_id"] == result.run.run_id
    assert re_metadata["model_path"] == metadata["model_path"]

    summary = load_summary(run_dir.parent)
    assert summary["run_id"].tolist() == [result.run.run_id, re.run.run_id]
    assert summary["threshold_mode"].tolist() == ["auto", "manual"]
    assert summary["retrained"].tolist() == [True, False]


def test_missing_dataset_does_not_create_run(small_config, mvtec_root: Path) -> None:
    shutil.rmtree(mvtec_root)
    with pytest.raises(DatasetNotFoundError):
        run_experiment(small_config, pre_trained=False)
    run_root = Path(small_config.output.run_root)
    assert not run_root.exists() or not any(run_root.iterdir())
