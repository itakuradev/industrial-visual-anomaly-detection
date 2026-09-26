"""異常スコアをGOOD / NGへ変換するThresholdの決定と適用。

- auto: split=validation の画像だけからF1最大のThresholdを決める（Anomalib F1AdaptiveThreshold）。
  ``resolve_threshold`` は受け取った推論結果からvalidation行だけを取り出して使うため、
  test（最終評価用）のスコアがThreshold決定に混入しない
- manual: 設定値（PatchCoreの生スコア単位）を使う
- 判定は ``anomaly_score >= threshold`` をNGとする（境界値は見逃し防止側に倒す）
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from anomalib.metrics import F1AdaptiveThreshold

from anomaly_detection.utils.config import ThresholdConfig

logger = logging.getLogger(__name__)

PREDICTION_GOOD = "GOOD"
PREDICTION_NG = "NG"
THRESHOLD_SPLIT = "validation"


class ThresholdError(ValueError):
    """Thresholdを決定できないことを表す。"""


@dataclass(frozen=True)
class ThresholdResult:
    value: float
    mode: str  # auto / manual
    source: str  # validation（autoで決定したデータ） / manual（設定値）
    num_images: int  # 決定に使った画像数（manualは0）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """F1 Scoreが最大となるThresholdを返す。``labels`` は1が異常。

    ThresholdはスコアのいずれかになるためFloat64のまま計算する。
    Float32へ丸めると境界上の画像がThresholdをわずかに下回り、判定が変わることがある。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        raise ThresholdError("Threshold決定用の画像が0件です。")
    if not ((labels == 0).any() and (labels == 1).any()):
        raise ThresholdError(
            "Thresholdの自動決定には、validationに正常画像と異常画像の両方が必要です"
            f"（正常 {int((labels == 0).sum())} 件, 異常 {int((labels == 1).sum())} 件）。"
            "Datasetの分割を見直すか、threshold.mode: manual を使用してください。"
        )
    metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    metric.update(SimpleNamespace(pred_score=torch.tensor(scores), gt_label=torch.tensor(labels)))
    return float(metric.compute())


def resolve_threshold(config: ThresholdConfig, predictions: pd.DataFrame) -> ThresholdResult:
    """設定に従いThresholdを決める。autoではsplit=validationの行だけを使う。"""
    if config.mode == "manual":
        result = ThresholdResult(
            value=float(config.manual_value), mode="manual", source="manual", num_images=0
        )
    else:
        validation = predictions[predictions["split"] == THRESHOLD_SPLIT]
        value = compute_f1_threshold(
            validation["anomaly_score"].to_numpy(),
            (validation["ground_truth"] == "defect").to_numpy(dtype=np.int64),
        )
        result = ThresholdResult(
            value=value, mode="auto", source=THRESHOLD_SPLIT, num_images=len(validation)
        )
    logger.info(
        "Threshold: %.6f (mode=%s, source=%s, images=%d)",
        result.value,
        result.mode,
        result.source,
        result.num_images,
    )
    return result


def apply_threshold(scores: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    """スコアをGOOD / NGへ変換する。"""
    return np.where(np.asarray(scores) >= threshold, PREDICTION_NG, PREDICTION_GOOD)


def with_predictions(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """推論結果の表へ ``threshold`` と ``prediction`` 列を付けた新しい表を返す。"""
    return predictions.assign(
        threshold=float(threshold),
        prediction=apply_threshold(predictions["anomaly_score"], threshold),
    )
