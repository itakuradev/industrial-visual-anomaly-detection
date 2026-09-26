"""Anomalib PatchCoreのWrapper（Model生成・保存・読み込み・Engine生成）。

閾値判定と評価は本パッケージ側で行うため、AnomalibのPostProcessor（スコア正規化・閾値）、
Evaluator、Visualizerは無効化し、PatchCoreの生スコアとAnomaly Mapをそのまま使う。
"""

from __future__ import annotations

import logging
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore

from anomaly_detection.utils.config import ModelConfig

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.ckpt"


class ModelNotFoundError(FileNotFoundError):
    """Model Artifactが存在しないことを表す。"""


class _IgnoreMissingCkptPathWarning(logging.Filter):
    """学習済みModelをEngine.predictへ直接渡すため、ckpt_path未指定の警告は誤解を招くので除外する。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return "ckpt_path is not provided" not in record.getMessage()


logging.getLogger("anomalib.engine.engine").addFilter(_IgnoreMissingCkptPathWarning())


def create_patchcore(
    config: ModelConfig, image_size: tuple[int, int], pre_trained: bool = True
) -> Patchcore:
    """設定からPatchCoreを生成する。``image_size`` は (height, width)。"""
    pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
    model = Patchcore(
        backbone=config.backbone,
        layers=list(config.layers),
        pre_trained=pre_trained,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=pre_processor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    logger.info(
        "PatchCore生成: backbone=%s layers=%s coreset_sampling_ratio=%s num_neighbors=%d "
        "image_size=%s",
        config.backbone,
        list(config.layers),
        config.coreset_sampling_ratio,
        config.num_neighbors,
        image_size,
    )
    return model


def create_engine(device: str, work_dir: Path) -> Engine:
    """学習・推論用のEngineを作る。Checkpointは ``save_model`` で明示的に保存する。"""
    return Engine(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=False,
        default_root_dir=work_dir,
        barebones=True,
    )


def save_model(engine: Engine, model_dir: Path) -> Path:
    """学習済みModel（Memory Bankを含むCheckpoint）を保存する。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILENAME
    engine.trainer.save_checkpoint(path)
    logger.info("Model保存先: %s", path)
    return path


def load_patchcore(path: str | Path) -> Patchcore:
    """保存したCheckpointからPatchCoreを読み込む。"""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILENAME
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model Fileが存在しません: {path}\n"
            "学習を実行してModelを保存するか、Runの model/ ディレクトリを指定してください。"
        )
    model = Patchcore.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    logger.info("Model読み込み: %s", path)
    return model
