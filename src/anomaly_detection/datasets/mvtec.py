"""MVTec AD 2 Dataset Adapter（Phase 1）。

Directory構造の解析はAnomalib公式の ``make_mvtec2_dataset`` を利用する。

MVTec AD 2の ``validation/`` は正常画像のみのため、F1による閾値決定に使えない。
そこで正解付きの ``test_public/`` をlabelで層化分割し、
``validation``（閾値決定用）と ``test``（最終評価用）を作る。
同じseed・割合なら同じ分割になる。``validation/good`` と ``test_private*`` は使用しない。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from anomalib.data.datasets.image.mvtecad2 import TestType, make_mvtec2_dataset
from sklearn.model_selection import train_test_split

from anomaly_detection.datasets.base import (
    LABEL_DEFECT,
    LABEL_GOOD,
    SAMPLE_COLUMNS,
    DatasetError,
    DatasetNotFoundError,
    log_sample_counts,
    validate_samples,
)
from anomaly_detection.utils.config import AppConfig

logger = logging.getLogger(__name__)

IMAGE_PATTERN = "*.png"
PLACEMENT_GUIDE = """\
MVTec AD 2 は公式ページ https://www.mvtec.com/company/research/datasets/mvtec-ad-2 から取得し
（License: CC BY-NC-SA 4.0）、次の構造で配置してください。

{root}/
└─ {category}/
   ├─ train/good/*.png
   ├─ validation/good/*.png
   └─ test_public/
      ├─ good/*.png
      ├─ bad/*.png
      └─ ground_truth/bad/*_mask.png"""


class MVTecAD2Adapter:
    """MVTec AD 2の1 Categoryを共通Sample Tableへ変換する。"""

    def __init__(self, root: Path, category: str, threshold_split_ratio: float, seed: int) -> None:
        self.root = Path(root)
        self.category = category
        self.threshold_split_ratio = threshold_split_ratio
        self.seed = seed

    @classmethod
    def from_config(cls, config: AppConfig) -> MVTecAD2Adapter:
        dataset = config.dataset
        return cls(
            root=config.resolve_path(dataset.root),
            category=dataset.category,
            threshold_split_ratio=dataset.threshold_split_ratio,
            seed=config.project.seed,
        )

    @property
    def category_dir(self) -> Path:
        return self.root / self.category

    def validate(self) -> None:
        """Dataset Root / Category / Training画像 / Test画像 / Maskの存在を順に確認する。"""
        guide = PLACEMENT_GUIDE.format(root=self.root, category=self.category)
        if not self.root.is_dir():
            raise DatasetNotFoundError(f"Dataset Rootが存在しません: {self.root}\n\n{guide}")
        if not self.category_dir.is_dir():
            raise DatasetNotFoundError(
                f"Category '{self.category}' が存在しません: {self.category_dir}\n\n{guide}"
            )
        required = {
            "Training画像": self.category_dir / "train" / "good",
            "Test画像（正常）": self.category_dir / "test_public" / "good",
            "Test画像（異常）": self.category_dir / "test_public" / "bad",
        }
        for name, directory in required.items():
            if not any(directory.glob(IMAGE_PATTERN)):
                raise DatasetNotFoundError(
                    f"{name}（{IMAGE_PATTERN}）が見つかりません: {directory}\n\n{guide}"
                )
        mask_dir = self.category_dir / "test_public" / "ground_truth" / "bad"
        if not any(mask_dir.glob(IMAGE_PATTERN)):
            raise DatasetNotFoundError(f"Ground Truth Maskが見つかりません: {mask_dir}\n\n{guide}")

    def load_samples(self) -> pd.DataFrame:
        """train / validation / test のsplitを持つSample Tableを返す。"""
        self.validate()
        logger.info("Dataset読み込み開始: MVTec AD 2 / %s (%s)", self.category, self.category_dir)
        try:
            raw = make_mvtec2_dataset(self.category_dir, test_type=TestType.PUBLIC)
        except Exception as e:  # MisMatchError（Mask不足）等をDatasetErrorとして報告する
            raise DatasetError(f"MVTec AD 2の読み込みに失敗しました: {e}") from e

        train = raw[raw["split"] == "train"]
        test_public = raw[raw["split"] == "test"].sort_values("image_path", ignore_index=True)
        validation, test = self._split_test_public(test_public)

        samples = pd.concat(
            [
                self._to_common(train, "train"),
                self._to_common(validation, "validation"),
                self._to_common(test, "test"),
            ],
            ignore_index=True,
        )
        validate_samples(samples)
        logger.info("Dataset読み込み完了: %d 件", len(samples))
        log_sample_counts(samples)
        return samples

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "dataset": "mvtec_ad2",
            "category": self.category,
            "root": str(self.root),
            "threshold_split": {
                "source": "test_public",
                "validation_ratio": self.threshold_split_ratio,
                "stratify": "label",
                "seed": self.seed,
            },
            "unused": ["validation/good", "test_private", "test_private_mixed"],
        }
        if self.category_dir.is_dir():
            info["mvtec_validation_good_count"] = len(
                list((self.category_dir / "validation" / "good").glob(IMAGE_PATTERN))
            )
        return info

    def _split_test_public(self, test_public: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        label_counts = test_public["label"].value_counts()
        if len(label_counts) < 2 or label_counts.min() < 2:
            raise DatasetError(
                "test_publicを閾値決定用と評価用に層化分割するには、good / bad がそれぞれ"
                f"2枚以上必要です（現在: {label_counts.to_dict()}）。"
            )
        try:
            validation, test = train_test_split(
                test_public,
                train_size=self.threshold_split_ratio,
                stratify=test_public["label"],
                random_state=self.seed,
            )
        except ValueError as e:
            raise DatasetError(
                f"test_publicの分割に失敗しました（threshold_split_ratio="
                f"{self.threshold_split_ratio}）: {e}"
            ) from e
        overlap = set(validation["image_path"]) & set(test["image_path"])
        if overlap:  # 閾値決定に使った画像を最終評価に使わないことを保証する
            raise DatasetError(f"validationとtestに同じ画像があります: {sorted(overlap)[:3]}")
        return validation, test

    @staticmethod
    def _to_common(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        is_defect = frame["label_index"] == 1
        common = pd.DataFrame(
            {
                "image_path": frame["image_path"].astype(str),
                "label": is_defect.map({True: LABEL_DEFECT, False: LABEL_GOOD}),
                "label_index": is_defect.astype(int),
                "mask_path": frame["mask_path"],
                "split": split,
                "defect_type": "",
                "object_id": "",
            }
        )
        return common[list(SAMPLE_COLUMNS)]
