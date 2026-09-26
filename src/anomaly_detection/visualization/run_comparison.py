"""Experiment Summary（``runs/summary.csv``）のRun間比較グラフ。

指標ごとにパネルを分けた横棒グラフ（Small Multiples）で表示する。
Thresholdはスコアの尺度が異なるため別パネルにし、2軸は使わない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
from matplotlib.figure import Figure

BAR_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
RATE_METRICS = {
    "recall",
    "precision",
    "f1",
    "accuracy",
    "auroc",
    "aupr",
    "pixel_auroc",
    "pixel_aupro",
}


def plot_run_comparison(
    summary: pd.DataFrame,
    metrics: Sequence[str] = ("recall", "precision", "f1", "threshold"),
    label_column: str = "run_id",
) -> Figure:
    """Runごとの指標を比較するFigureを返す（行はsummaryの順）。値のないRunは N/A と表示する。"""
    if summary.empty:
        raise ValueError("比較するRunがありません。")
    labels = summary[label_column].astype(str).tolist()[::-1]  # 上から古い順に並べる
    height = max(2.2, 0.45 * len(labels) + 1.2)
    figure = Figure(figsize=(3.4 * len(metrics), height), facecolor=SURFACE_COLOR)
    axes = figure.subplots(1, len(metrics), sharey=True, squeeze=False)[0]

    for ax, metric in zip(axes, metrics, strict=True):
        values = pd.to_numeric(summary[metric], errors="coerce").tolist()[::-1]
        plotted = [0.0 if _missing(v) else v for v in values]
        ax.set_facecolor(SURFACE_COLOR)
        ax.barh(labels, plotted, height=0.55, color=BAR_COLOR)
        upper = 1.0 if metric in RATE_METRICS else max(max(plotted), 1e-9) * 1.15
        ax.set_xlim(0, upper * 1.18)
        for y, value in enumerate(values):
            text = "N/A" if _missing(value) else f"{value:.3f}"
            x = 0 if _missing(value) else value
            ax.text(x + upper * 0.02, y, text, va="center", fontsize=8, color=TEXT_PRIMARY)
        ax.set_title(metric, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
    figure.tight_layout()
    return figure


def _missing(value: float) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
