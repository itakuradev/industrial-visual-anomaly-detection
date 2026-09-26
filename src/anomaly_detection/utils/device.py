"""実行Deviceの決定。GPUは必須とせず、利用できない場合はCPUへFallbackする。"""

from __future__ import annotations

import logging
import platform

import torch

from anomaly_detection.utils.config import DEVICES

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """設定値（auto / cuda / cpu）から実際に使うDevice（cuda / cpu）を決める。"""
    if requested not in DEVICES:
        raise ValueError(
            f"runtime.device は {', '.join(DEVICES)} のいずれかです（値: {requested!r}）。"
        )

    if requested == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        if requested == "cuda":
            logger.warning(
                "runtime.device=cuda が指定されましたが CUDA を利用できません"
                "（torch.cuda.is_available() が False）。GPUドライバーと PyTorch の CUDA 版が"
                "インストールされているか確認してください。CPU で実行を継続します。"
            )
        device = "cpu"

    info = describe_device(device)
    logger.info("使用Device: %s (%s)", info["device"], info["name"])
    return device


def describe_device(device: str) -> dict[str, str | None]:
    """Run Metadataへ保存するDevice情報を返す。"""
    if device == "cuda":
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }
    return {"device": "cpu", "name": platform.processor() or "CPU", "cuda_version": None}
