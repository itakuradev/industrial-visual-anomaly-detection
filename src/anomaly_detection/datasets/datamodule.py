"""Sample TableからAnomalibのDataset / DataModuleを作る共通層。

画像はOpenCVで読み込み、``ImagePreprocessor`` を読み込み時に適用する（元画像は変更しない）。
ModelのPreProcessorが持つResizeは、Anomalibの仕組みによりDatasetのaugmentationsへ追加される。
"""

from __future__ import annotations

import pandas as pd
import torch
from anomalib.data import ImageItem
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from torchvision.transforms.v2.functional import to_dtype, to_image
from torchvision.tv_tensors import Mask

from anomaly_detection.datasets.base import read_image, read_mask
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor


class PreprocessedDataset(AnomalibDataset):
    """OpenCV前処理を適用して ``ImageItem`` を返すDataset。"""

    def __init__(self, samples: pd.DataFrame, preprocessor: ImagePreprocessor) -> None:
        super().__init__(augmentations=None)
        self.preprocessor = preprocessor
        frame = samples.reset_index(drop=True).copy()
        # Maskが1件でもあればSegmentation（Pixel-level評価可能）として扱う
        frame.attrs["task"] = (
            "segmentation" if frame["mask_path"].notna().any() else "classification"
        )
        self.samples = frame

    def __getitem__(self, index: int) -> ImageItem:
        row = self.samples.iloc[index]
        image_np = self.preprocessor.process(read_image(row.image_path))
        image = to_dtype(to_image(image_np), torch.float32, scale=True)

        gt_mask = None
        mask_path = None if pd.isna(row.mask_path) else str(row.mask_path)
        if self.task == "segmentation":
            if mask_path is None:
                gt_mask = Mask(torch.zeros(image.shape[-2:], dtype=torch.uint8))
            else:
                mask_np = self.preprocessor.process_mask(read_mask(mask_path))
                gt_mask = Mask(torch.from_numpy(mask_np))

        if self.augmentations:
            if gt_mask is None:
                image = self.augmentations(image)
            else:
                image, gt_mask = self.augmentations(image, gt_mask)

        return ImageItem(
            image=image,
            gt_mask=gt_mask,
            gt_label=torch.tensor(int(row.label_index)),
            image_path=str(row.image_path),
            mask_path=mask_path,
        )


class SampleDataModule(AnomalibDataModule):
    """Sample Tableのsplit列（train / validation / test）からDataModuleを作る。

    推論時はAnomalibの仕様に合わせ、推論対象を ``test`` 側（predict_dataloader）に置く。
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        preprocessor: ImagePreprocessor,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__(
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
        )
        self.sample_table = samples
        self.preprocessor = preprocessor

    def _setup(self, _stage: str | None = None) -> None:
        split = self.sample_table["split"]
        self.train_data = PreprocessedDataset(
            self.sample_table[split == "train"], self.preprocessor
        )
        self.val_data = PreprocessedDataset(
            self.sample_table[split == "validation"], self.preprocessor
        )
        self.test_data = PreprocessedDataset(self.sample_table[split == "test"], self.preprocessor)

    def _create_test_split(self) -> None:
        # splitはSample Tableで確定済み。Anomalibの既定処理はtest_split_mode=NONEでも
        # testから正常画像を取り除くため、何もしない。
        pass

    @classmethod
    def for_prediction(
        cls,
        samples: pd.DataFrame,
        preprocessor: ImagePreprocessor,
        batch_size: int,
        num_workers: int,
    ) -> SampleDataModule:
        """任意のSampleを推論対象（predict_dataloader）とするDataModuleを作る。"""
        return cls(samples.assign(split="test"), preprocessor, batch_size, num_workers)
