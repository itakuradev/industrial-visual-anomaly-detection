"""Image-level / Pixel-level評価指標。

異常（defect / NG）を陽性クラスとする。外観検査では異常の見逃し（False Negative）を
重要なエラーとして扱うため、Recallを主要指標として確認する。
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from anomalib.metrics import AUPRO, AUROC
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from anomaly_detection.evaluation.threshold import PREDICTION_NG

logger = logging.getLogger(__name__)

AUPRO_FPR_LIMIT = 0.3


def compute_image_metrics(
    ground_truth: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """GOOD / NG判定と異常スコアからImage-level指標を計算する。

    AUROC / AUPR（Average Precision）はThresholdに依存しない。
    正常・異常の片方しか含まれない場合は計算できないため ``None`` とする。
    """
    y_true = np.asarray(ground_truth) == "defect"
    y_pred = np.asarray(predictions) == PREDICTION_NG
    y_score = np.asarray(scores, dtype=float)
    if y_true.size == 0:
        raise ValueError("評価対象の画像が0件です。")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    both_classes = bool(y_true.any() and (~y_true).any())
    if not both_classes:
        logger.warning("評価データに正常・異常の片方しかないため、AUROC / AUPRは算出しません。")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero_divisionは0として扱う
        return {
            "num_images": int(y_true.size),
            "num_good": int((~y_true).sum()),
            "num_defect": int(y_true.sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auroc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
            "aupr": float(average_precision_score(y_true, y_score)) if both_classes else None,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }


def compute_pixel_metrics(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> dict[str, Any]:
    """Anomaly MapとGT MaskからPixel AUROCとAUPRO（FPR上限0.3）を計算する。"""
    if not gt_masks.any():
        logger.warning("異常Pixelが存在しないため、Pixel-level指標は算出しません。")
        return {"pixel_auroc": None, "pixel_aupro": None, "aupro_fpr_limit": AUPRO_FPR_LIMIT}
    batch = SimpleNamespace(
        anomaly_map=torch.from_numpy(np.asarray(anomaly_maps, dtype=np.float32)),
        gt_mask=torch.from_numpy(np.asarray(gt_masks, dtype=bool)),
    )
    auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=AUPRO_FPR_LIMIT)
    auroc.update(batch)
    aupro.update(batch)
    return {
        "pixel_auroc": float(auroc.compute()),
        # 浮動小数点誤差で1をわずかに超えることがあるため上限を1にする
        "pixel_aupro": min(float(aupro.compute()), 1.0),
        "aupro_fpr_limit": AUPRO_FPR_LIMIT,
    }


def evaluate_split(
    predictions: pd.DataFrame,
    split: str,
    anomaly_maps: np.ndarray | None = None,
    gt_masks: np.ndarray | None = None,
) -> dict[str, Any]:
    """指定splitの行だけでImage-level（とMaskがあればPixel-level）指標を計算する。

    ``anomaly_maps`` / ``gt_masks`` は ``predictions`` の行順（位置）に対応していること。
    """
    selected = (predictions["split"] == split).to_numpy()
    if not selected.any():
        raise ValueError(f"split={split} の推論結果がありません。")
    rows = predictions[selected]
    result: dict[str, Any] = {
        "split": split,
        "image": compute_image_metrics(
            rows["ground_truth"], rows["prediction"], rows["anomaly_score"]
        ),
        "pixel": None,
    }
    if anomaly_maps is not None and gt_masks is not None:
        result["pixel"] = compute_pixel_metrics(anomaly_maps[selected], gt_masks[selected])
    return result


def confusion_matrix_frame(image_metrics: dict[str, Any]) -> pd.DataFrame:
    """Notebook表示用のConfusion Matrix（行: Ground Truth、列: Prediction）。"""
    m = image_metrics
    return pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=pd.Index(["good", "defect"], name="ground_truth"),
        columns=pd.Index(["GOOD", "NG"], name="prediction"),
    )
