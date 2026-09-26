from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from anomaly_detection.utils.config import AppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG = PROJECT_ROOT / "configs" / "phase1_mvtec_vial.yaml"
PHASE2_CONFIG = PROJECT_ROOT / "configs" / "phase2_walnut.yaml"
IMAGE_SIZE = (48, 64)  # (height, width)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def good_image(rng: np.random.Generator) -> np.ndarray:
    noise = rng.integers(0, 12, (*IMAGE_SIZE, 3), dtype=np.uint8)
    return np.full((*IMAGE_SIZE, 3), 120, np.uint8) + noise


def defect_image_and_mask(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = good_image(rng)
    mask = np.zeros(IMAGE_SIZE, np.uint8)
    mask[10:24, 20:36] = 255
    image[mask > 0] = 255
    return image, mask


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    """MVTec AD 2と同じ構造の小さな合成Dataset（category=vial）。"""
    rng = np.random.default_rng(0)
    category = tmp_path / "mvtec_ad2" / "vial"
    for i in range(6):
        write_png(category / "train" / "good" / f"{i:03d}_regular.png", good_image(rng))
    for i in range(2):
        write_png(category / "validation" / "good" / f"{i:03d}_regular.png", good_image(rng))
    for i in range(6):
        write_png(category / "test_public" / "good" / f"{i:03d}_regular.png", good_image(rng))
    for i in range(6):
        image, mask = defect_image_and_mask(rng)
        write_png(category / "test_public" / "bad" / f"{i:03d}_regular.png", image)
        write_png(
            category / "test_public" / "ground_truth" / "bad" / f"{i:03d}_regular_mask.png", mask
        )
    return tmp_path / "mvtec_ad2"


@pytest.fixture
def small_config(mvtec_root: Path, tmp_path: Path) -> AppConfig:
    """合成Dataset・小さな画像サイズ・resnet18・CPUを使うテスト用設定。"""
    config = load_config(PHASE1_CONFIG)
    height, width = IMAGE_SIZE
    preprocessing = dataclasses.replace(
        config.preprocessing,
        resize=dataclasses.replace(config.preprocessing.resize, width=width, height=height),
    )
    return dataclasses.replace(
        config,
        dataset=dataclasses.replace(config.dataset, root=str(mvtec_root)),
        model=dataclasses.replace(config.model, backbone="resnet18"),
        preprocessing=preprocessing,
        runtime=dataclasses.replace(config.runtime, device="cpu", batch_size=4),
        output=dataclasses.replace(config.output, run_root=str(tmp_path / "runs")),
    )


def small_runtime(config: AppConfig, tmp_path: Path, dataset: dict) -> AppConfig:
    """小さな画像サイズ・resnet18・CPU・一時Run出力先へ置き換える。"""
    height, width = IMAGE_SIZE
    preprocessing = dataclasses.replace(
        config.preprocessing,
        resize=dataclasses.replace(config.preprocessing.resize, width=width, height=height),
    )
    return dataclasses.replace(
        config,
        dataset=dataclasses.replace(config.dataset, **dataset),
        model=dataclasses.replace(config.model, backbone="resnet18"),
        preprocessing=preprocessing,
        runtime=dataclasses.replace(config.runtime, device="cpu", batch_size=4),
        output=dataclasses.replace(config.output, run_root=str(tmp_path / "runs")),
    )


@pytest.fixture
def walnut_root(tmp_path: Path) -> Path:
    """くるみDatasetの配置規約に従う小さな合成Dataset（Manifestなし）。

    good: G001〜G010（各2角度）、crack: C001〜C004、chip: K001〜K004（各1枚）。
    """
    rng = np.random.default_rng(1)
    root = tmp_path / "walnut"
    for i in range(1, 11):
        for shot in (1, 2):
            write_png(root / "raw" / "good" / f"G{i:03d}_{shot}.jpg", good_image(rng))
    for prefix, class_dir in (("C", "crack"), ("K", "chip")):
        for i in range(1, 5):
            image, _ = defect_image_and_mask(rng)
            write_png(root / "raw" / class_dir / f"{prefix}{i:03d}_1.jpg", image)
    return root


@pytest.fixture
def walnut_config(walnut_root: Path, tmp_path: Path) -> AppConfig:
    config = load_config(PHASE2_CONFIG)
    return small_runtime(
        config,
        tmp_path,
        {"root": str(walnut_root), "manifest": str(walnut_root / "manifest.csv")},
    )
