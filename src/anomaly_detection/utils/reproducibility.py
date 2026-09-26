"""Random Seedの設定と、再現性のための実行環境情報の収集。

完全なBit-level再現性は保証しない（GPUの非決定的な演算やライブラリ差があるため）。
"""

from __future__ import annotations

import logging
import platform
import sys
from importlib import metadata
from typing import Any

import torch
from lightning import seed_everything

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "anomalib",
    "lightning",
    "opencv-python-headless",
    "numpy",
    "scikit-learn",
)


def set_seed(seed: int) -> None:
    """Python random / NumPy / PyTorch / CUDA のSeedを設定する。"""
    seed_everything(seed, workers=True, verbose=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random Seed: %d", seed)


def collect_environment() -> dict[str, Any]:
    """Python・主要PackageのVersionとPlatformを返す。"""
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda_build": torch.version.cuda,
        "packages": packages,
    }
