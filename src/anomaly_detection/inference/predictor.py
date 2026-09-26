"""学習済みPatchCoreによる推論。画像ごとの生スコアとAnomaly Mapを返す。

GOOD / NG の判定（Threshold適用）は評価モジュール側で行う。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from anomalib.models import Patchcore

from anomaly_detection.datasets.base import DatasetError
from anomaly_detection.datasets.datamodule import SampleDataModule
from anomaly_detection.models.patchcore import create_engine
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.utils.config import AppConfig

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = (
    "image_path",
    "split",
    "ground_truth",
    "defect_type",
    "object_id",
    "anomaly_score",
)


@dataclass(frozen=True)
class PredictionResult:
    """推論結果。``anomaly_maps`` と ``gt_masks`` の並びは ``table`` の行順と一致する。"""

    table: pd.DataFrame
    anomaly_maps: np.ndarray  # (N, H, W) float32
    gt_masks: np.ndarray | None  # (N, H, W) uint8。Maskを持つDatasetのみ


def predict(
    model: Patchcore,
    samples: pd.DataFrame,
    config: AppConfig,
    device: str,
    work_dir: Path,
) -> PredictionResult:
    """``samples`` の全画像を推論する。前処理は学習時と同じ設定を使うこと。"""
    if samples.empty:
        raise DatasetError("推論対象の画像が0件です。")

    preprocessor = ImagePreprocessor(config.preprocessing)
    datamodule = SampleDataModule.for_prediction(
        samples, preprocessor, config.runtime.batch_size, config.runtime.num_workers
    )
    engine = create_engine(device, work_dir)

    logger.info("Inference開始: %d 件, device=%s", len(samples), device)
    start = time.perf_counter()
    batches = engine.predict(model=model, datamodule=datamodule) or []
    logger.info("Inference完了: %.1f 秒", time.perf_counter() - start)

    paths: list[str] = []
    scores: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in batches:
        paths.extend(batch.image_path)
        scores.append(batch.pred_score.detach().cpu().reshape(-1).numpy())
        anomaly_map = batch.anomaly_map.detach().cpu()
        maps.append(anomaly_map.reshape(-1, *anomaly_map.shape[-2:]).numpy())  # (B, H, W)
        if batch.gt_mask is not None:
            masks.append(batch.gt_mask.detach().cpu().numpy().astype(np.uint8))

    # 入力samplesの行順へ揃える
    position = {path: i for i, path in enumerate(paths)}
    missing = [p for p in samples["image_path"] if p not in position]
    if missing:
        raise RuntimeError(f"推論結果が得られなかった画像があります: {missing[:3]}")
    order = np.array([position[p] for p in samples["image_path"]])

    score_array = np.concatenate(scores)[order]
    table = pd.DataFrame(
        {
            "image_path": samples["image_path"].to_numpy(),
            "split": samples["split"].to_numpy(),
            "ground_truth": samples["label"].to_numpy(),
            "defect_type": samples["defect_type"].to_numpy(),
            "object_id": samples["object_id"].to_numpy(),
            "anomaly_score": score_array.astype(float),
        }
    )
    return PredictionResult(
        table=table,
        anomaly_maps=np.concatenate(maps)[order].astype(np.float32),
        gt_masks=np.concatenate(masks)[order] if len(masks) == len(batches) and masks else None,
    )
