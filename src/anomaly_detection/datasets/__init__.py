"""Dataset Adapter（MVTec AD 2 / Walnut）と共通Sample Table。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anomaly_detection.datasets.base import DatasetAdapter
    from anomaly_detection.utils.config import AppConfig


def create_adapter(config: AppConfig) -> DatasetAdapter:
    """``dataset.type`` に対応するAdapterを返す。後続の共通PipelineはAdapterの種類に依存しない。"""
    if config.dataset.type == "mvtec_ad2":
        from anomaly_detection.datasets.mvtec import MVTecAD2Adapter

        return MVTecAD2Adapter.from_config(config)
    if config.dataset.type == "walnut":
        from anomaly_detection.datasets.walnut import WalnutAdapter

        return WalnutAdapter.from_config(config)
    raise NotImplementedError(f"dataset.type={config.dataset.type} のAdapterは未実装です。")
