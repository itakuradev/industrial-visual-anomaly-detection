"""Dataset Adapterの共通定義。

各Adapter（MVTec AD 2 / Walnut）は、Dataset固有の構造を次の列を持つSample Table
（pandas.DataFrame）へ変換する。学習・推論・評価はこのTableだけに依存する。

- ``image_path``: 画像の絶対パス
- ``label``: ``good`` / ``defect``
- ``label_index``: 0（good）/ 1（defect）
- ``mask_path``: Ground Truth Maskの絶対パス（なければ欠損値）
- ``split``: ``train``（学習）/ ``validation``（閾値決定）/ ``test``（最終評価）
- ``defect_type``: 欠陥種別（なければ空文字）
- ``object_id``: 個体ID（Data Leakage防止用。なければ空文字）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_COLUMNS = (
    "image_path",
    "label",
    "label_index",
    "mask_path",
    "split",
    "defect_type",
    "object_id",
)
LABEL_GOOD = "good"
LABEL_DEFECT = "defect"
LABEL_INDEX = {LABEL_GOOD: 0, LABEL_DEFECT: 1}
SPLITS = ("train", "validation", "test")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DatasetError(Exception):
    """Datasetの構成や内容の不備を表す。"""


class DatasetNotFoundError(DatasetError):
    """Datasetが配置されていないことを表す。メッセージに配置方法を含める。"""


class DatasetAdapter(Protocol):
    """Dataset固有の構造をSample Tableへ変換するAdapter。"""

    def validate(self) -> None:
        """Datasetの配置を検証し、不備があれば ``DatasetError`` を送出する。"""
        ...

    def load_samples(self) -> pd.DataFrame:
        """split列を含むSample Tableを返す。"""
        ...

    def describe(self) -> dict[str, Any]:
        """Run Metadataへ保存するDataset情報を返す。"""
        ...


def validate_samples(samples: pd.DataFrame) -> None:
    """全Datasetに共通する条件をSample Tableに対して検証する。"""
    missing_columns = [c for c in SAMPLE_COLUMNS if c not in samples.columns]
    if missing_columns:
        raise DatasetError(f"Sample Tableに必要な列がありません: {', '.join(missing_columns)}")
    if samples.empty:
        raise DatasetError("画像が0件です。Datasetの配置を確認してください。")

    unknown_splits = sorted(set(samples["split"]) - set(SPLITS))
    if unknown_splits:
        raise DatasetError(f"未知のsplitがあります: {unknown_splits}（使用可能: {list(SPLITS)}）")
    unknown_labels = sorted(set(samples["label"]) - set(LABEL_INDEX))
    if unknown_labels:
        raise DatasetError(
            f"未知のlabelがあります: {unknown_labels}（使用可能: {list(LABEL_INDEX)}）"
        )

    for split in ("train", "test"):
        if not (samples["split"] == split).any():
            raise DatasetError(f"split={split} の画像が0件です。")

    train_defects = samples[(samples["split"] == "train") & (samples["label"] == LABEL_DEFECT)]
    if not train_defects.empty:
        examples = ", ".join(train_defects["image_path"].head(3))
        raise DatasetError(
            f"Trainingに異常画像が {len(train_defects)} 件含まれています（例: {examples}）。"
            "PatchCoreの学習には正常画像のみを使用してください。"
        )

    duplicated = samples[samples["image_path"].duplicated(keep=False)]
    if not duplicated.empty:
        examples = ", ".join(duplicated["image_path"].drop_duplicates().head(3))
        raise DatasetError(f"同じ画像が複数回登録されています（例: {examples}）。")

    unsupported = samples[
        ~samples["image_path"].map(lambda p: Path(p).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    ]
    if not unsupported.empty:
        examples = ", ".join(unsupported["image_path"].head(3))
        raise DatasetError(
            f"対応していない画像形式です（例: {examples}）。"
            f"対応形式: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    paths = pd.concat([samples["image_path"], samples["mask_path"].dropna()])
    missing = paths[~paths.map(lambda p: Path(p).is_file())]
    if not missing.empty:
        examples = ", ".join(missing.head(3))
        raise DatasetError(f"ファイルが存在しません: {len(missing)} 件（例: {examples}）")


def count_samples(samples: pd.DataFrame) -> dict[str, dict[str, int]]:
    """split × label の件数を返す。"""
    counts = samples.groupby(["split", "label"]).size()
    return {
        split: {label: int(counts.get((split, label), 0)) for label in LABEL_INDEX}
        for split in SPLITS
    }


def log_sample_counts(samples: pd.DataFrame) -> None:
    for split, labels in count_samples(samples).items():
        logger.info(
            "Dataset件数 %-10s good=%d defect=%d",
            split,
            labels[LABEL_GOOD],
            labels[LABEL_DEFECT],
        )


def read_image(path: str | Path) -> np.ndarray:
    """画像をRGB（uint8, HxWx3）で読み込む。Windowsの非ASCIIパスにも対応する。"""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise DatasetError(
            f"対応していない画像形式です: {path}"
            f"（対応形式: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}）"
        )
    if not path.is_file():
        raise DatasetError(f"画像ファイルが存在しません: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"画像を読み込めません（破損または非対応の形式）: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Ground Truth Maskを0/1（uint8, HxW）で読み込む。"""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Maskファイルが存在しません: {path}")
    mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise DatasetError(f"Maskを読み込めません: {path}")
    return (mask > 0).astype(np.uint8)
