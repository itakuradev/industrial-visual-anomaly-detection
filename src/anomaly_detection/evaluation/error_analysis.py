"""誤判定（False Positive / False Negative）の抽出と一覧表示。

推論結果の表は行番号（RangeIndex）がAnomaly Map配列の位置と対応している前提で、
抽出結果の index からAnomaly Mapを参照する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from anomaly_detection.evaluation.threshold import PREDICTION_GOOD, PREDICTION_NG
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.visualization.anomaly_visualizer import (
    HeatmapScale,
    load_display_image,
    render_result,
)

FALSE_POSITIVE = "FP"
FALSE_NEGATIVE = "FN"


def get_false_positives(predictions: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """正常なのにNGと判定された画像（過検知）。スコアの高い順。"""
    rows = predictions[
        (predictions["split"] == split)
        & (predictions["ground_truth"] == "good")
        & (predictions["prediction"] == PREDICTION_NG)
    ]
    return rows.sort_values("anomaly_score", ascending=False)


def get_false_negatives(predictions: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """異常なのにGOODと判定された画像（見逃し）。スコアの低い順（Thresholdから遠い順）。"""
    rows = predictions[
        (predictions["split"] == split)
        & (predictions["ground_truth"] == "defect")
        & (predictions["prediction"] == PREDICTION_GOOD)
    ]
    return rows.sort_values("anomaly_score", ascending=True)


def get_errors(
    predictions: pd.DataFrame, split: str = "test", fn_first: bool = True
) -> pd.DataFrame:
    """誤判定をまとめて返す。``fn_first=True`` なら見逃し（FN）を先に並べる。"""
    fn = get_false_negatives(predictions, split).assign(error_type=FALSE_NEGATIVE)
    fp = get_false_positives(predictions, split).assign(error_type=FALSE_POSITIVE)
    return pd.concat([fn, fp] if fn_first else [fp, fn])


def plot_errors(
    errors: pd.DataFrame,
    anomaly_maps: np.ndarray,
    preprocessor: ImagePreprocessor,
    scale: HeatmapScale,
    max_items: int | None = 10,
) -> list[Figure]:
    """誤判定ごとに Original / Heatmap / Overlay とGT・判定・スコアを表示するFigureを返す。"""
    figures = []
    shown = errors if max_items is None else errors.head(max_items)
    for position, row in shown.iterrows():
        anomaly_map = anomaly_maps[position]
        image = load_display_image(row["image_path"], preprocessor, anomaly_map.shape)
        prefix = f"[{row['error_type']}] " if "error_type" in row else ""
        figures.append(render_result(image, anomaly_map, row, scale, title_prefix=prefix))
    return figures
