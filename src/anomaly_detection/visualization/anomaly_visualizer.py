"""推論結果の可視化（Original / Anomaly Heatmap / Overlay）と画像保存。

- Heatmapの色の範囲はRun内で共通（``HeatmapScale``）にし、画像間で色を比較できるようにする
- OverlayにはAnomaly MapがThreshold以上の領域のContourとBounding Boxを描く
- 表示する元画像は幾何変換（crop / resize）だけを適用し、Anomaly Mapと位置を合わせる
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from anomaly_detection.datasets.base import read_image
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor

logger = logging.getLogger(__name__)

CONTOUR_COLOR = (255, 255, 255)
BBOX_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class HeatmapScale:
    """Heatmap表示の値域。表示専用で、判定や評価には使わない。"""

    vmin: float
    vmax: float

    @classmethod
    def from_maps(cls, anomaly_maps: np.ndarray) -> HeatmapScale:
        return cls(vmin=float(np.min(anomaly_maps)), vmax=float(np.max(anomaly_maps)))

    def normalize(self, anomaly_map: np.ndarray) -> np.ndarray:
        span = max(self.vmax - self.vmin, 1e-12)
        return np.clip((anomaly_map - self.vmin) / span, 0.0, 1.0)


def colorize(anomaly_map: np.ndarray, scale: HeatmapScale) -> np.ndarray:
    """Anomaly MapをJET配色のRGB画像（uint8）へ変換する。"""
    gray = (scale.normalize(anomaly_map) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


def draw_regions(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float,
    contour: bool = True,
    bbox: bool = True,
) -> np.ndarray:
    """Anomaly MapがThreshold以上の領域のContour / Bounding Boxを描いた画像を返す。"""
    out = image.copy()
    region = (anomaly_map >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contour:
        cv2.drawContours(out, contours, -1, CONTOUR_COLOR, 1)
    if bbox:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), BBOX_COLOR, 1)
    return out


def load_display_image(
    image_path: str | Path, preprocessor: ImagePreprocessor, size: tuple[int, int]
) -> np.ndarray:
    """元画像に幾何変換を適用し、Anomaly Mapと同じサイズ (height, width) にして返す。"""
    image = preprocessor.apply_geometry(read_image(image_path))
    height, width = size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def render_result(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    row: pd.Series,
    scale: HeatmapScale,
    title_prefix: str = "",
) -> Figure:
    """1画像分の Original / Heatmap / Overlay を横並びにしたFigureを返す。

    ``row`` は推論結果の1行（ground_truth, prediction, anomaly_score, threshold, image_path）。
    """
    heatmap = colorize(anomaly_map, scale)
    marked = draw_regions(overlay(image, heatmap), anomaly_map, row["threshold"])

    figure = Figure(figsize=(12, 4.6))
    axes = figure.subplots(1, 3)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    shown = axes[1].imshow(anomaly_map, cmap="jet", vmin=scale.vmin, vmax=scale.vmax)
    axes[1].set_title("Anomaly Heatmap")
    figure.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(marked)
    axes[2].set_title("Overlay (region >= threshold)")
    for ax in axes:
        ax.axis("off")

    correct = (row["ground_truth"] == "defect") == (row["prediction"] == "NG")
    figure.suptitle(
        f"{title_prefix}GT: {row['ground_truth']} | Pred: {row['prediction']} | "
        f"Score: {row['anomaly_score']:.4f} | Threshold: {row['threshold']:.4f}",
        color="black" if correct else "red",
        fontsize=12,
    )
    # 絶対パス（ユーザー名等）を画像に残さないよう、末尾3階層だけを表示する
    short_path = "/".join(Path(row["image_path"]).parts[-3:])
    figure.text(0.5, 0.02, short_path, ha="center", fontsize=8, color="gray")
    return figure


def output_filename(row: pd.Series) -> str:
    """保存用のファイル名。good/badで同名の画像があるため、split・GT・親Directory名を付ける。"""
    path = Path(row["image_path"])
    return f"{row['split']}_{row['ground_truth']}_{path.parent.name}_{path.stem}.png"


def save_visualizations(
    predictions: pd.DataFrame,
    anomaly_maps: np.ndarray,
    preprocessor: ImagePreprocessor,
    scale: HeatmapScale,
    images_dir: Path,
    save_heatmaps: bool = True,
    save_overlays: bool = True,
) -> int:
    """全画像のHeatmap（``heatmaps/``）と結果画像（``overlays/``）を保存し、保存枚数を返す。

    ``anomaly_maps`` は ``predictions`` の行順（位置）に対応していること。
    """
    if not (save_heatmaps or save_overlays):
        return 0
    heatmap_dir = images_dir / "heatmaps"
    overlay_dir = images_dir / "overlays"
    for directory, enabled in ((heatmap_dir, save_heatmaps), (overlay_dir, save_overlays)):
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    for position, (_, row) in enumerate(predictions.iterrows()):
        anomaly_map = anomaly_maps[position]
        name = output_filename(row)
        if save_heatmaps:
            heatmap = colorize(anomaly_map, scale)
            cv2.imwrite(str(heatmap_dir / name), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        if save_overlays:
            image = load_display_image(row["image_path"], preprocessor, anomaly_map.shape)
            render_result(image, anomaly_map, row, scale).savefig(overlay_dir / name, dpi=100)
    logger.info("可視化画像を保存: %d 件 (%s)", len(predictions), images_dir)
    return len(predictions)
