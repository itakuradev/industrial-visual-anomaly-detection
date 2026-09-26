from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anomaly_detection.datasets import create_adapter
from anomaly_detection.datasets.base import DatasetNotFoundError, read_image
from anomaly_detection.datasets.walnut import (
    ManifestError,
    WalnutAdapter,
    allocate,
    count_by_defect_type,
    find_object_leaks,
    load_manifest,
    object_id_from_filename,
    validate_manifest,
)
from anomaly_detection.utils.config import SplitRatios


@pytest.fixture
def adapter(walnut_config) -> WalnutAdapter:
    created = create_adapter(walnut_config)
    assert isinstance(created, WalnutAdapter)
    return created


@pytest.fixture
def manifest(adapter: WalnutAdapter) -> pd.DataFrame:
    return adapter.build_manifest()


def test_object_id_from_filename() -> None:
    assert object_id_from_filename("raw/good/G001_2.jpg") == "G001"
    assert object_id_from_filename("raw/good/W001_left_3.jpg") == "W001"
    assert object_id_from_filename("raw/good/walnut01.jpg") == "walnut01"


@pytest.mark.parametrize(
    ("n", "expected"),
    [(10, (6, 2, 2)), (4, (2, 1, 1)), (3, (2, 1, 0)), (1, (1, 0, 0)), (0, (0, 0, 0))],
)
def test_allocate(n: int, expected: tuple[int, int, int]) -> None:
    counts = allocate(n, SplitRatios(train=0.6, validation=0.2, test=0.2))
    assert (counts["train"], counts["validation"], counts["test"]) == expected


class TestBuildManifest:
    def test_split_by_object_and_ratio(self, manifest: pd.DataFrame, adapter) -> None:
        assert adapter.manifest_path.is_file()
        good_objects = manifest[manifest["label"] == "good"].groupby("split")["object_id"].nunique()
        assert good_objects.to_dict() == {"train": 6, "validation": 2, "test": 2}
        defects = manifest[manifest["label"] == "defect"]
        assert (defects["split"] != "train").all()
        assert defects.groupby(["defect_type", "split"]).size().to_dict() == {
            ("chip", "test"): 2,
            ("chip", "validation"): 2,
            ("crack", "test"): 2,
            ("crack", "validation"): 2,
        }
        assert find_object_leaks(manifest).empty

    def test_reproducible_with_seed(self, adapter: WalnutAdapter, manifest: pd.DataFrame) -> None:
        again = adapter.build_manifest(overwrite=True)
        pd.testing.assert_frame_equal(manifest, again)

    def test_does_not_overwrite_without_flag(self, adapter: WalnutAdapter, manifest) -> None:
        with pytest.raises(ManifestError, match="overwrite=True"):
            adapter.build_manifest()

    def test_unsupported_format(self, adapter: WalnutAdapter, walnut_root: Path) -> None:
        (walnut_root / "raw" / "good" / "G099_1.HEIC").write_bytes(b"")
        with pytest.raises(ManifestError, match="HEIC"):
            adapter.build_manifest()

    def test_object_id_shared_between_classes(self, adapter, walnut_root: Path) -> None:
        image = cv2.imread(str(walnut_root / "raw" / "good" / "G001_1.jpg"))
        cv2.imwrite(str(walnut_root / "raw" / "crack" / "G001_9.jpg"), image)
        with pytest.raises(ManifestError, match="複数のクラス"):
            adapter.build_manifest()

    def test_no_images(self, adapter: WalnutAdapter, walnut_root: Path) -> None:
        for path in (walnut_root / "raw").rglob("*.jpg"):
            path.unlink()
        with pytest.raises(DatasetNotFoundError, match="raw/ に画像がありません"):
            adapter.build_manifest()


class TestValidateManifest:
    def test_valid(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        validate_manifest(load_manifest(walnut_root / "manifest.csv"), walnut_root)

    def test_missing_column(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        with pytest.raises(ManifestError, match="object_id"):
            validate_manifest(manifest.drop(columns="object_id"), walnut_root)

    def test_missing_file(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        (walnut_root / manifest.loc[0, "image_path"]).unlink()
        with pytest.raises(ManifestError, match="画像ファイルが存在しません"):
            validate_manifest(manifest, walnut_root)

    @pytest.mark.parametrize(
        ("column", "value", "message"),
        [
            ("label", "ok", "label"),
            ("split", "val", "split"),
            ("object_id", "", "object_id"),
        ],
    )
    def test_invalid_values(self, manifest, walnut_root: Path, column, value, message) -> None:
        manifest.loc[0, column] = value
        with pytest.raises(ManifestError, match=message):
            validate_manifest(manifest, walnut_root)

    def test_defect_type_rules(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        good = manifest.index[manifest["label"] == "good"][0]
        broken = manifest.copy()
        broken.loc[good, "defect_type"] = "crack"
        with pytest.raises(ManifestError, match="label=good"):
            validate_manifest(broken, walnut_root)
        defect = manifest.index[manifest["label"] == "defect"][0]
        broken = manifest.copy()
        broken.loc[defect, "defect_type"] = "scratch"
        with pytest.raises(ManifestError, match="label=defect"):
            validate_manifest(broken, walnut_root)

    def test_defect_in_train(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        defect = manifest.index[manifest["label"] == "defect"][0]
        manifest.loc[defect, "split"] = "train"
        with pytest.raises(ManifestError, match="Trainingに異常画像"):
            validate_manifest(manifest, walnut_root)

    def test_object_id_leakage(self, manifest: pd.DataFrame, walnut_root: Path) -> None:
        rows = manifest.index[manifest["object_id"] == "G001"]
        assert len(rows) == 2
        manifest.loc[rows[0], "split"] = "train"
        manifest.loc[rows[1], "split"] = "test"
        assert find_object_leaks(manifest)["object_id"].tolist() == ["G001"]
        with pytest.raises(ManifestError, match="複数のsplitに跨がって"):
            validate_manifest(manifest, walnut_root)


class TestAdapter:
    def test_missing_manifest_shows_guide(self, adapter: WalnutAdapter) -> None:
        with pytest.raises(DatasetNotFoundError, match="build_manifest"):
            adapter.validate()

    def test_missing_root(self, walnut_config, walnut_root: Path) -> None:
        import shutil

        shutil.rmtree(walnut_root)
        with pytest.raises(DatasetNotFoundError, match="Dataset Rootが存在しません"):
            create_adapter(walnut_config).validate()

    def test_load_samples(self, adapter: WalnutAdapter, manifest: pd.DataFrame) -> None:
        samples = adapter.load_samples()
        assert len(samples) == len(manifest)
        assert samples["mask_path"].isna().all()  # Maskなし → Image-level評価のみ
        assert samples["image_path"].map(lambda p: Path(p).is_absolute()).all()
        table = count_by_defect_type(samples)
        assert table.loc["total"].to_dict() == {"good": 20, "crack": 4, "chip": 4}
        assert table.loc["train", "crack"] == 0

    def test_describe(self, adapter: WalnutAdapter, manifest: pd.DataFrame) -> None:
        info = adapter.describe()
        assert info["dataset"] == "walnut"
        assert info["num_objects"] == 18
        assert info["pixel_level_evaluation"] is False


def test_smartphone_exif_orientation_is_applied(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # 90度回転して表示する指定
    Image.new("RGB", (40, 20)).save(buffer, "JPEG", exif=exif.tobytes())
    path = tmp_path / "rotated.jpg"
    path.write_bytes(buffer.getvalue())
    assert read_image(path).shape == (40, 20, 3)
    assert np.asarray(Image.open(path)).shape == (20, 40, 3)  # 回転情報を無視した場合
