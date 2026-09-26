"""PatchCoreの学習（正常画像からMemory Bankを構築）とModel Artifact保存。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from anomalib.models import Patchcore

from anomaly_detection.datasets.base import DatasetError
from anomaly_detection.datasets.datamodule import SampleDataModule
from anomaly_detection.models.patchcore import create_engine, create_patchcore, save_model
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.utils.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model: Patchcore
    model_path: Path | None
    num_train_images: int
    memory_bank_size: int
    duration_sec: float


def train_patchcore(
    config: AppConfig,
    samples: pd.DataFrame,
    device: str,
    output_dir: Path,
    pre_trained: bool = True,
) -> TrainingResult:
    """split=train の画像でPatchCoreを学習し、設定に応じて ``output_dir/model`` へ保存する。

    ``pre_trained=False`` はBackboneの重みをDownloadしないテスト用。
    """
    train = samples[samples["split"] == "train"]
    if train.empty:
        raise DatasetError("split=train の画像が0件のため学習できません。")

    preprocessor = ImagePreprocessor(config.preprocessing)
    logger.info("Preprocessing: %s", preprocessor.summary())
    model = create_patchcore(config.model, preprocessor.output_size, pre_trained=pre_trained)
    datamodule = SampleDataModule(
        train, preprocessor, config.runtime.batch_size, config.runtime.num_workers
    )
    engine = create_engine(device, output_dir)

    logger.info("Training開始: 学習画像 %d 件, device=%s", len(train), device)
    start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    duration = time.perf_counter() - start
    memory_bank_size = int(model.model.memory_bank.shape[0])
    logger.info("Training完了: %.1f 秒, Memory Bank %d 件", duration, memory_bank_size)

    model_path = save_model(engine, output_dir / "model") if config.output.save_model else None
    return TrainingResult(
        model=model,
        model_path=model_path,
        num_train_images=len(train),
        memory_bank_size=memory_bank_size,
        duration_sec=duration,
    )
