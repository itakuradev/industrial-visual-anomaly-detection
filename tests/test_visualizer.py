from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from conftest import write_png

from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.visualization.anomaly_visualizer import (
    HeatmapScale,
    colorize,
    draw_regions,
    load_display_image,
    output_filename,
    save_visualizations,
)


def test_scale_and_colorize() -> None:
    maps = np.array([[[0.0, 1.0], [2.0, 4.0]]], np.float32)
    scale = HeatmapScale.from_maps(maps)
    assert (scale.vmin, scale.vmax) == (0.0, 4.0)
    assert scale.normalize(np.array([-1.0, 2.0, 9.0])).tolist() == [0.0, 0.5, 1.0]
    heatmap = colorize(maps[0], scale)
    assert heatmap.shape == (2, 2, 3)
    assert heatmap.dtype == np.uint8


def test_draw_regions_marks_only_above_threshold() -> None:
    image = np.zeros((20, 20, 3), np.uint8)
    anomaly_map = np.zeros((20, 20), np.float32)
    anomaly_map[5:10, 5:10] = 1.0
    marked = draw_regions(image, anomaly_map, threshold=0.5)
    assert marked[5, 5].any()  # Contour / Bounding Box
    assert not marked[15, 15].any()
    assert not image.any()  # 入力は変更しない
    assert not draw_regions(image, anomaly_map, threshold=2.0).any()


def test_output_filename_distinguishes_good_and_bad() -> None:
    good = pd.Series(
        {"image_path": "x/test_public/good/000.png", "split": "test", "ground_truth": "good"}
    )
    bad = pd.Series(
        {"image_path": "x/test_public/bad/000.png", "split": "test", "ground_truth": "defect"}
    )
    assert output_filename(good) == "test_good_good_000.png"
    assert output_filename(bad) == "test_defect_bad_000.png"


def test_load_display_image_matches_map_size(tmp_path: Path, small_config) -> None:
    path = tmp_path / "a.png"
    write_png(path, np.zeros((100, 80, 3), np.uint8))
    image = load_display_image(path, ImagePreprocessor(small_config.preprocessing), (16, 24))
    assert image.shape == (16, 24, 3)


def test_save_visualizations(tmp_path: Path, small_config) -> None:
    rows = []
    for name, gt in (("good", "good"), ("bad", "defect")):
        path = tmp_path / name / "000.png"
        write_png(path, np.full((48, 64, 3), 120, np.uint8))
        rows.append(
            {
                "image_path": str(path),
                "split": "test",
                "ground_truth": gt,
                "anomaly_score": 0.3 if gt == "good" else 0.9,
                "threshold": 0.5,
                "prediction": "GOOD" if gt == "good" else "NG",
            }
        )
    table = pd.DataFrame(rows)
    maps = np.random.default_rng(0).random((2, 48, 64)).astype(np.float32)
    count = save_visualizations(
        table,
        maps,
        ImagePreprocessor(small_config.preprocessing),
        HeatmapScale.from_maps(maps),
        tmp_path / "images",
    )
    assert count == 2
    for kind in ("heatmaps", "overlays"):
        files = sorted(p.name for p in (tmp_path / "images" / kind).iterdir())
        assert files == ["test_defect_bad_000.png", "test_good_good_000.png"]
    heatmap = cv2.imread(str(tmp_path / "images" / "heatmaps" / "test_good_good_000.png"))
    assert heatmap.shape == (48, 64, 3)


def test_save_visualizations_disabled(tmp_path: Path, small_config) -> None:
    count = save_visualizations(
        pd.DataFrame(),
        np.zeros((0, 4, 4)),
        ImagePreprocessor(small_config.preprocessing),
        HeatmapScale(0.0, 1.0),
        tmp_path / "images",
        save_heatmaps=False,
        save_overlays=False,
    )
    assert count == 0
    assert not (tmp_path / "images").exists()
