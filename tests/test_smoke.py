"""合成Dataset・小さなModel（resnet18、事前学習なし、CPU）で学習〜推論の経路を確認する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import IMAGE_SIZE

from anomaly_detection.datasets import create_adapter
from anomaly_detection.inference.predictor import PREDICTION_COLUMNS, predict
from anomaly_detection.models.patchcore import MODEL_FILENAME, ModelNotFoundError, load_patchcore
from anomaly_detection.training.trainer import train_patchcore


def test_train_save_load_predict(small_config, tmp_path: Path) -> None:
    samples = create_adapter(small_config).load_samples()
    run_dir = tmp_path / "run"

    result = train_patchcore(small_config, samples, "cpu", run_dir, pre_trained=False)
    assert result.num_train_images == 6
    assert result.memory_bank_size > 0
    assert result.model_path == run_dir / "model" / MODEL_FILENAME
    assert result.model_path.is_file()

    model = load_patchcore(run_dir / "model")
    assert model.model.memory_bank.shape == result.model.model.memory_bank.shape

    targets = samples[samples["split"] != "train"]
    prediction = predict(model, targets, small_config, "cpu", run_dir)
    table = prediction.table
    assert tuple(table.columns) == PREDICTION_COLUMNS
    assert table["image_path"].tolist() == targets["image_path"].tolist()
    assert np.isfinite(table["anomaly_score"]).all()
    assert prediction.anomaly_maps.shape == (len(targets), *IMAGE_SIZE)
    assert prediction.gt_masks is not None
    assert prediction.gt_masks.shape == (len(targets), *IMAGE_SIZE)
    defect_rows = (table["ground_truth"] == "defect").to_numpy()
    assert prediction.gt_masks[defect_rows].any(axis=(1, 2)).all()
    assert not prediction.gt_masks[~defect_rows].any()


def test_load_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="Model Fileが存在しません"):
        load_patchcore(tmp_path / "model")
