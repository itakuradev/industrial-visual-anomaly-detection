from __future__ import annotations

import logging
from pathlib import Path

from anomaly_detection.utils.logging import PACKAGE_LOGGER_NAME, setup_logging


def test_logs_to_file_without_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file=log_file)
    logger = setup_logging(log_file=log_file)
    assert len(logger.handlers) == 2

    logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test").info("dataset loaded")
    for handler in logger.handlers:
        handler.flush()
    assert "dataset loaded" in log_file.read_text(encoding="utf-8")

    setup_logging()  # File Handlerを閉じてtmp_pathを削除可能にする
