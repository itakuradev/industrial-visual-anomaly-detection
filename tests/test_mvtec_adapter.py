from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.datasets import create_adapter
from anomaly_detection.datasets.base import DatasetError, DatasetNotFoundError
from anomaly_detection.datasets.mvtec import MVTecAD2Adapter


def adapter(root: Path, ratio: float = 0.5, seed: int = 42) -> MVTecAD2Adapter:
    return MVTecAD2Adapter(root=root, category="vial", threshold_split_ratio=ratio, seed=seed)


def test_create_adapter_from_config(small_config, mvtec_root: Path) -> None:
    created = create_adapter(small_config)
    assert isinstance(created, MVTecAD2Adapter)
    assert created.category_dir == mvtec_root / "vial"


class TestValidation:
    def test_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset Rootが存在しません") as e:
            adapter(tmp_path / "nowhere").validate()
        assert "test_public" in str(e.value)  # 配置方法を案内する

    def test_missing_category(self, mvtec_root: Path) -> None:
        shutil.rmtree(mvtec_root / "vial")
        with pytest.raises(DatasetNotFoundError, match="Category 'vial'"):
            adapter(mvtec_root).validate()

    @pytest.mark.parametrize(
        ("relative", "message"),
        [
            ("train/good", "Training画像"),
            ("test_public/good", "Test画像（正常）"),
            ("test_public/bad", "Test画像（異常）"),
            ("test_public/ground_truth/bad", "Ground Truth Mask"),
        ],
    )
    def test_missing_directory(self, mvtec_root: Path, relative: str, message: str) -> None:
        shutil.rmtree(mvtec_root / "vial" / relative)
        with pytest.raises(DatasetNotFoundError, match=message):
            adapter(mvtec_root).validate()

    def test_missing_mask_for_one_image(self, mvtec_root: Path) -> None:
        (mvtec_root / "vial/test_public/ground_truth/bad/000_regular_mask.png").unlink()
        with pytest.raises(DatasetError, match="Missing mask"):
            adapter(mvtec_root).load_samples()


class TestSamples:
    def test_split_counts_and_labels(self, mvtec_root: Path) -> None:
        samples = adapter(mvtec_root).load_samples()
        counts = samples.groupby(["split", "label"]).size().to_dict()
        assert counts == {
            ("train", "good"): 6,
            ("validation", "good"): 3,
            ("validation", "defect"): 3,
            ("test", "good"): 3,
            ("test", "defect"): 3,
        }
        # MVTecの validation/good（正常のみ）は使用しない
        assert not samples["image_path"].str.contains(r"[\\/]validation[\\/]").any()

    def test_validation_and_test_do_not_overlap(self, mvtec_root: Path) -> None:
        samples = adapter(mvtec_root).load_samples()
        validation = set(samples.loc[samples["split"] == "validation", "image_path"])
        test = set(samples.loc[samples["split"] == "test", "image_path"])
        assert validation.isdisjoint(test)
        assert len(validation | test) == 12

    def test_split_is_reproducible_with_seed(self, mvtec_root: Path) -> None:
        first = adapter(mvtec_root, seed=7).load_samples()
        second = adapter(mvtec_root, seed=7).load_samples()
        pd.testing.assert_frame_equal(first, second)

    def test_ratio_changes_split_size(self, mvtec_root: Path) -> None:
        samples = adapter(mvtec_root, ratio=0.34).load_samples()
        counts = samples["split"].value_counts()
        assert counts["validation"] == 4
        assert counts["test"] == 8

    def test_masks_only_for_defects(self, mvtec_root: Path) -> None:
        samples = adapter(mvtec_root).load_samples()
        assert samples.loc[samples["label"] == "defect", "mask_path"].notna().all()
        assert samples.loc[samples["label"] == "good", "mask_path"].isna().all()

    def test_too_few_images_to_stratify(self, mvtec_root: Path) -> None:
        for path in list((mvtec_root / "vial/test_public/bad").glob("*.png"))[1:]:
            path.unlink()
        with pytest.raises(DatasetError, match="2枚以上"):
            adapter(mvtec_root).load_samples()

    def test_describe(self, mvtec_root: Path) -> None:
        info = adapter(mvtec_root).describe()
        assert info["dataset"] == "mvtec_ad2"
        assert info["threshold_split"]["source"] == "test_public"
        assert info["mvtec_validation_good_count"] == 2
