"""パッケージ共通のLogging設定。

``anomaly_detection`` 配下の各モジュールは ``logging.getLogger(__name__)`` で取得したLoggerを使い、
出力先（Console / Log File）はここで一括設定する。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER_NAME = "anomaly_detection"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    level: int | str = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """ConsoleとLog Fileへの出力を設定する。

    Notebookでセルを再実行してもHandlerが重複しないよう、既存のHandlerは置き換える。
    """
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
