"""Run単位の実行結果の保存・読み込みとExperiment Summary。

::

    runs/
    ├─ summary.csv
    └─ 20260926_163000_phase1/
       ├─ config.yaml        設定Snapshot
       ├─ metadata.json      環境・Device・Dataset・Threshold・学習情報
       ├─ metrics.json       評価指標（test）とThreshold決定データでの参考値
       ├─ samples.csv        split割り当てを含むSample Table
       ├─ predictions.csv    画像ごとのスコア・Threshold・判定
       ├─ anomaly_maps.npz   Anomaly Map（float16）とGT Mask
       ├─ run.log
       ├─ model/
       └─ images/heatmaps/, images/overlays/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.utils.config import AppConfig, load_config, save_config

PHASES = {"mvtec_ad2": "phase1", "walnut": "phase2"}
SUMMARY_FILENAME = "summary.csv"
SUMMARY_COLUMNS = (
    "run_id",
    "created_at",
    "phase",
    "dataset",
    "model",
    "threshold_mode",
    "threshold",
    "preprocessing_summary",
    "evaluation_split",
    "num_images",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "aupr",
    "pixel_auroc",
    "pixel_aupro",
    "fp",
    "fn",
    "device",
    "retrained",
    "parent_run_id",
)


class RunNotFoundError(FileNotFoundError):
    """Run Directoryや必要な保存ファイルが存在しないことを表す。"""


@dataclass(frozen=True)
class Run:
    run_id: str
    run_dir: Path

    @classmethod
    def create(cls, run_root: Path, phase: str, now: datetime | None = None) -> Run:
        """``<run_root>/<YYYYmmdd_HHMMSS>_<phase>`` を作る。同名があれば連番を付ける。"""
        base = f"{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}_{phase}"
        run_id, number = base, 1
        while (run_root / run_id).exists():
            number += 1
            run_id = f"{base}_{number}"
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True)
        return cls(run_id=run_id, run_dir=run_dir)

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def samples_path(self) -> Path:
        return self.run_dir / "samples.csv"

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.csv"

    @property
    def anomaly_maps_path(self) -> Path:
        return self.run_dir / "anomaly_maps.npz"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "model"

    @property
    def images_dir(self) -> Path:
        return self.run_dir / "images"

    def save_config(self, config: AppConfig) -> None:
        save_config(config, self.config_path)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        _write_json(self.metadata_path, metadata)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        _write_json(self.metrics_path, metrics)

    def save_samples(self, samples: pd.DataFrame) -> None:
        samples.to_csv(self.samples_path, index=False, encoding="utf-8")

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        predictions.to_csv(self.predictions_path, index=False, encoding="utf-8")

    def save_anomaly_maps(self, anomaly_maps: np.ndarray, gt_masks: np.ndarray | None) -> None:
        arrays = {"anomaly_maps": anomaly_maps.astype(np.float16)}
        if gt_masks is not None:
            arrays["gt_masks"] = gt_masks.astype(np.uint8)
        np.savez_compressed(self.anomaly_maps_path, **arrays)


@dataclass(frozen=True)
class LoadedRun:
    run: Run
    config: AppConfig
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    anomaly_maps: np.ndarray
    gt_masks: np.ndarray | None


def load_run(run_dir: str | Path) -> LoadedRun:
    """保存済みRunを読み込む。Anomaly Mapは ``predictions`` の行順に対応する。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunNotFoundError(f"Run Directoryが存在しません: {run_dir}")
    run = Run(run_id=run_dir.name, run_dir=run_dir)
    required = (run.config_path, run.metadata_path, run.predictions_path, run.anomaly_maps_path)
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise RunNotFoundError(
            f"Runに必要なファイルがありません: {', '.join(missing)}（{run_dir}）。"
            "output.save_predictions: true で実行したRunを指定してください。"
        )
    with np.load(run.anomaly_maps_path) as arrays:
        anomaly_maps = arrays["anomaly_maps"].astype(np.float32)
        gt_masks = arrays["gt_masks"] if "gt_masks" in arrays else None
    predictions = pd.read_csv(
        run.predictions_path, keep_default_na=False, dtype={"defect_type": str, "object_id": str}
    )
    return LoadedRun(
        run=run,
        config=load_config(run.config_path),
        metadata=_read_json(run.metadata_path),
        metrics=_read_json(run.metrics_path) if run.metrics_path.is_file() else {},
        predictions=predictions,
        anomaly_maps=anomaly_maps,
        gt_masks=gt_masks,
    )


def append_summary(run_root: Path, row: dict[str, Any]) -> Path:
    """``runs/summary.csv`` へ1 Run分の主要結果を追記する。"""
    path = run_root / SUMMARY_FILENAME
    new_row = pd.DataFrame([{column: row.get(column) for column in SUMMARY_COLUMNS}])
    existing = load_summary(run_root)
    summary = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    summary.to_csv(path, index=False, encoding="utf-8")
    return path


def load_summary(run_root: Path) -> pd.DataFrame:
    path = run_root / SUMMARY_FILENAME
    if not path.is_file():
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    return pd.read_csv(path, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
