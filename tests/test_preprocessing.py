from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from matplotlib.figure import Figure

from anomaly_detection.preprocessing.pipeline import ImagePreprocessor, compare_images
from anomaly_detection.utils.config import (
    ClaheConfig,
    ContrastConfig,
    CropConfig,
    DenoiseConfig,
    NormalizeConfig,
    PreprocessingConfig,
    ResizeConfig,
)

ALL_DISABLED = PreprocessingConfig(
    resize=ResizeConfig(enabled=False, width=32, height=24),
    crop=CropConfig(enabled=False),
    contrast=ContrastConfig(enabled=False),
    clahe=ClaheConfig(enabled=False),
    denoise=DenoiseConfig(enabled=False),
    normalize=NormalizeConfig(enabled=False),
)


def config_with(**changes) -> PreprocessingConfig:
    return dataclasses.replace(ALL_DISABLED, **changes)


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(80, 140, (60, 80, 3), dtype=np.uint8)


def test_all_disabled_returns_equal_copy(image: np.ndarray) -> None:
    out = ImagePreprocessor(ALL_DISABLED).process(image)
    assert np.array_equal(out, image)
    assert not np.shares_memory(out, image)


def test_resize_shape(image: np.ndarray) -> None:
    pre = ImagePreprocessor(config_with(resize=ResizeConfig(enabled=True, width=32, height=24)))
    assert pre.process(image).shape == (24, 32, 3)
    assert pre.output_size == (24, 32)


def test_crop_shape_and_content(image: np.ndarray) -> None:
    crop = CropConfig(enabled=True, x=10, y=5, width=40, height=30)
    out = ImagePreprocessor(config_with(crop=crop)).process(image)
    assert out.shape == (30, 40, 3)
    assert np.array_equal(out, image[5:35, 10:50])
    assert not np.shares_memory(out, image)


def test_crop_out_of_bounds(image: np.ndarray) -> None:
    crop = CropConfig(enabled=True, x=50, y=0, width=40, height=30)
    with pytest.raises(ValueError, match="crop範囲"):
        ImagePreprocessor(config_with(crop=crop)).process(image)


def test_crop_then_resize(image: np.ndarray) -> None:
    config = config_with(
        crop=CropConfig(enabled=True, x=0, y=0, width=40, height=40),
        resize=ResizeConfig(enabled=True, width=20, height=20),
    )
    assert ImagePreprocessor(config).process(image).shape == (20, 20, 3)


@pytest.mark.parametrize("method", ["gaussian", "median"])
def test_denoise_keeps_shape_and_smooths(image: np.ndarray, method: str) -> None:
    config = config_with(denoise=DenoiseConfig(enabled=True, method=method, kernel_size=5))
    out = ImagePreprocessor(config).process(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.std() < image.std()


def test_contrast(image: np.ndarray) -> None:
    config = config_with(contrast=ContrastConfig(enabled=True, alpha=1.5, beta=10))
    out = ImagePreprocessor(config).process(image)
    assert out.shape == image.shape
    assert out.std() > image.std()


def test_clahe_increases_contrast_of_flat_image(image: np.ndarray) -> None:
    config = config_with(clahe=ClaheConfig(enabled=True, clip_limit=4.0, tile_grid_size=4))
    out = ImagePreprocessor(config).process(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.std() > image.std()


def test_normalize_stretches_to_full_range(image: np.ndarray) -> None:
    out = ImagePreprocessor(config_with(normalize=NormalizeConfig(enabled=True))).process(image)
    assert out.min() == 0
    assert out.max() == 255


def test_input_is_not_modified(image: np.ndarray) -> None:
    original = image.copy()
    config = config_with(
        clahe=ClaheConfig(enabled=True),
        denoise=DenoiseConfig(enabled=True),
        normalize=NormalizeConfig(enabled=True),
    )
    ImagePreprocessor(config).process(image)
    assert np.array_equal(image, original)


def test_rejects_non_rgb_uint8() -> None:
    pre = ImagePreprocessor(ALL_DISABLED)
    with pytest.raises(ValueError, match="RGB"):
        pre.process(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        pre.process(np.zeros((10, 10, 3), np.float32))


def test_mask_follows_geometry_and_stays_binary() -> None:
    mask = np.zeros((60, 80), np.uint8)
    mask[20:40, 30:50] = 1
    config = config_with(
        crop=CropConfig(enabled=True, x=20, y=10, width=40, height=40),
        resize=ResizeConfig(enabled=True, width=20, height=20),
        clahe=ClaheConfig(enabled=True),
    )
    out = ImagePreprocessor(config).process_mask(mask)
    assert out.shape == (20, 20)
    assert set(np.unique(out)) == {0, 1}
    # crop後の (y=10..30, x=10..30) が欠陥領域 → 1/2に縮小して (5..15, 5..15)
    assert out[10, 10] == 1
    assert out[0, 0] == 0


def test_summary() -> None:
    assert ImagePreprocessor(ALL_DISABLED).summary() == "none"
    config = config_with(
        resize=ResizeConfig(enabled=True, width=256, height=256),
        clahe=ClaheConfig(enabled=True),
        denoise=DenoiseConfig(enabled=True, method="median"),
    )
    assert ImagePreprocessor(config).summary() == "resize256x256+denoise-median+clahe"


def test_compare_images(image: np.ndarray) -> None:
    figure = compare_images(image, image[:30])
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
