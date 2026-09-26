"""パッケージ共通のLogging設定。

``anomaly_detection`` 配下の各モジュールは ``logging.getLogger(__name__)`` で取得したLoggerを使い、
出力先（Console / Log File）はここで一括設定する。
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

PACKAGE_LOGGER_NAME = "anomaly_detection"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# 原因が分かっており動作に影響しないライブラリの警告（Notebookの可読性のため非表示にする）
KNOWN_HARMLESS_WARNINGS = (
    r".*does not have many workers.*",  # num_workers=0 を意図的に既定としている
    r"Total length of `DataLoader` across ranks is zero.*",  # 学習時はvalidationを使わない
    r"Found \d+ module\(s\) in eval mode.*",  # PatchCoreのBackboneは学習しない
    r"`LightningModule.configure_optimizers` returned `None`.*",  # PatchCoreはOptimizer不要
    r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",  # Lightning内部
    r"Attribute 'pre_processor' is an instance of `nn.Module`.*",  # Anomalib内部
    r"Argument embeddings is deprecated.*",  # Anomalib内部
    r".*cache-system uses symlinks.*",  # WindowsのHugging Face Cache
    r"IProgress not found.*",  # ipywidgets未導入時もtqdmはテキストで進捗を表示する
    # Anomalib / timm import時の非推奨警告（本パッケージのコードは該当APIを使わない）
    r"`torch.jit.script` is deprecated.*",
    r"Importing from timm.models.layers is deprecated.*",
    r"The anomalib.models.components.dinov2 package is deprecated.*",
)
# Lightningは lightning.* と lightning_fabric.* の両方の名前空間でINFOログを出す
THIRD_PARTY_INFO_LOGGERS = (
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "lightning_fabric",
)


def setup_logging(
    level: int | str = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """ConsoleとLog Fileへの出力を設定する。

    Notebookでセルを再実行してもHandlerが重複しないよう、既存のHandlerは置き換える。
    LightningのINFOログ（Trainer設定の説明等）と既知の無害な警告は表示しない。
    """
    for name in THIRD_PARTY_INFO_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for pattern in KNOWN_HARMLESS_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
