from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import write_png

from anomaly_detection.datasets.base import (
    DatasetError,
    count_samples,
    read_image,
    read_mask,
    validate_samples,
)


def make_samples(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for name, label, split in [
        ("a.png", "good", "train"),
        ("b.png", "good", "test"),
        ("c.png", "defect", "test"),
    ]:
        path = tmp_path / name
        write_png(path, np.zeros((8, 8, 3), np.uint8))
        rows.append(
            {
                "image_path": str(path),
                "label": label,
                "label_index": int(label == "defect"),
                "mask_path": None,
                "split": split,
                "defect_type": "",
                "object_id": "",
            }
        )
    return pd.DataFrame(rows)


def test_valid_samples(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    validate_samples(samples)
    assert count_samples(samples)["test"] == {"good": 1, "defect": 1}


def test_empty(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="0件"):
        validate_samples(make_samples(tmp_path).iloc[0:0])


def test_missing_column(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="object_id"):
        validate_samples(make_samples(tmp_path).drop(columns="object_id"))


def test_defect_in_train(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    samples.loc[2, "split"] = "train"
    with pytest.raises(DatasetError, match="Trainingに異常画像"):
        validate_samples(samples)


def test_missing_file(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    Path(samples.loc[1, "image_path"]).unlink()
    with pytest.raises(DatasetError, match="ファイルが存在しません"):
        validate_samples(samples)


def test_unsupported_format(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    heic = tmp_path / "d.heic"
    heic.write_bytes(b"")
    samples.loc[1, "image_path"] = str(heic)
    with pytest.raises(DatasetError, match="対応していない画像形式"):
        validate_samples(samples)


def test_duplicate_image(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    samples.loc[1, "image_path"] = samples.loc[0, "image_path"]
    with pytest.raises(DatasetError, match="複数回"):
        validate_samples(samples)


def test_read_image_rgb_and_non_ascii_path(tmp_path: Path) -> None:
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 2] = 255  # OpenCVのBGRで赤
    path = tmp_path / "くるみ" / "赤.png"
    path.parent.mkdir()
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    encoded.tofile(path)  # 非ASCIIパスはcv2.imwriteが使えないためtofileで書き込む
    rgb = read_image(path)
    assert rgb.shape == (4, 6, 3)
    assert (rgb[..., 0] == 255).all()


def test_read_image_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="存在しません"):
        read_image(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="読み込めません"):
        read_image(broken)


def test_read_mask_binary(tmp_path: Path) -> None:
    mask = np.zeros((4, 4), np.uint8)
    mask[1:3, 1:3] = 255
    write_png(tmp_path / "m.png", mask)
    out = read_mask(tmp_path / "m.png")
    assert set(np.unique(out)) == {0, 1}
