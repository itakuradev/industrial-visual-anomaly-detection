"""Phase 1 / Phase 2共通のExperiment Pipeline。

``run_experiment``: Config読込 → Device決定 → Seed設定 → Dataset読込 → 学習 → Model保存 →
推論 → Threshold決定（validation） → 評価（test） → 可視化 → Run保存 → Summary追記

``reevaluate``: 保存済みRunのスコアへ手動Thresholdを適用し、再学習せずに再評価する。
結果は新しいRunとして保存し、``retrained: false`` と元Runを記録する。
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.datasets import create_adapter
from anomaly_detection.datasets.base import count_samples
from anomaly_detection.evaluation.metrics import evaluate_split
from anomaly_detection.evaluation.threshold import (
    ThresholdResult,
    resolve_threshold,
    with_predictions,
)
from anomaly_detection.experiments.run_manager import PHASES, Run, append_summary, load_run
from anomaly_detection.inference.predictor import PREDICTION_COLUMNS, predict
from anomaly_detection.preprocessing.pipeline import ImagePreprocessor
from anomaly_detection.training.trainer import train_patchcore
from anomaly_detection.utils.config import AppConfig, ThresholdConfig, load_config
from anomaly_detection.utils.device import describe_device, resolve_device
from anomaly_detection.utils.logging import setup_logging
from anomaly_detection.utils.reproducibility import collect_environment, set_seed
from anomaly_detection.visualization.anomaly_visualizer import HeatmapScale, save_visualizations

logger = logging.getLogger(__name__)

EVALUATION_SPLIT = "test"
MANUAL_THRESHOLD_NOTE = (
    "manualのThresholdをtestの結果を見ながら決めた場合、このRunのtest指標は独立した評価ではない。"
)


@dataclass(frozen=True)
class ExperimentResult:
    run: Run
    config: AppConfig
    predictions: pd.DataFrame  # threshold / prediction列を含む。行順はanomaly_mapsと一致
    anomaly_maps: np.ndarray
    gt_masks: np.ndarray | None
    threshold: ThresholdResult
    metrics: dict[str, Any]
    heatmap_scale: HeatmapScale

    @property
    def preprocessor(self) -> ImagePreprocessor:
        return ImagePreprocessor(self.config.preprocessing)


def run_experiment(config: str | Path | AppConfig, pre_trained: bool = True) -> ExperimentResult:
    """学習から評価・保存までを1 Runとして実行する。

    ``pre_trained=False`` はBackboneの重みをDownloadしないテスト用。
    """
    if not isinstance(config, AppConfig):
        config = load_config(config)
    adapter = create_adapter(config)
    adapter.validate()  # Datasetが未配置ならRunを作らずに終了する
    run = Run.create(config.resolve_path(config.output.run_root), PHASES[config.dataset.type])
    setup_logging(log_file=run.log_path)
    try:
        logger.info("Execution Start: run_id=%s", run.run_id)
        logger.info("Config Path: %s", config.config_path)
        logger.info("Run Directory: %s", run.run_dir)
        run.save_config(config)
        device = resolve_device(config.runtime.device)
        set_seed(config.project.seed)

        samples = adapter.load_samples()
        run.save_samples(samples)

        training = train_patchcore(config, samples, device, run.run_dir, pre_trained=pre_trained)
        targets = samples[samples["split"] != "train"].reset_index(drop=True)
        prediction = predict(training.model, targets, config, device, run.run_dir)
        threshold = resolve_threshold(config.threshold, prediction.table)

        metadata = {
            "device": describe_device(device),
            "dataset": {**adapter.describe(), "counts": count_samples(samples)},
            "training": {
                "num_train_images": training.num_train_images,
                "memory_bank_size": training.memory_bank_size,
                "duration_sec": round(training.duration_sec, 2),
            },
            "model_path": str(training.model_path) if training.model_path else None,
            "retrained": True,
            "parent_run_id": None,
        }
        return _evaluate_and_save(
            run,
            config,
            prediction.table,
            prediction.anomaly_maps,
            prediction.gt_masks,
            threshold,
            metadata,
        )
    except Exception:
        logger.exception("実行中にエラーが発生しました (run_id=%s)", run.run_id)
        raise
    finally:
        setup_logging()  # Run Log Fileを閉じる


def reevaluate(run_dir: str | Path, threshold: float) -> ExperimentResult:
    """保存済みRunのスコアに手動Thresholdを適用して再評価する（再学習なし）。"""
    parent = load_run(run_dir)
    config = dataclasses.replace(
        parent.config, threshold=ThresholdConfig(mode="manual", manual_value=float(threshold))
    )
    run = Run.create(parent.run.run_dir.parent, PHASES[config.dataset.type])
    setup_logging(log_file=run.log_path)
    try:
        logger.info("Execution Start: run_id=%s", run.run_id)
        logger.info(
            "再評価（再学習なし）: parent_run_id=%s, threshold=%s", parent.run.run_id, threshold
        )
        logger.info("Run Directory: %s", run.run_dir)
        run.save_config(config)
        threshold_result = resolve_threshold(config.threshold, parent.predictions)
        metadata = {
            "device": parent.metadata.get("device"),
            "dataset": parent.metadata.get("dataset"),
            "training": None,
            "model_path": parent.metadata.get("model_path"),
            "retrained": False,
            "parent_run_id": parent.run.run_id,
        }
        return _evaluate_and_save(
            run,
            config,
            parent.predictions[list(PREDICTION_COLUMNS)],
            parent.anomaly_maps,
            parent.gt_masks,
            threshold_result,
            metadata,
        )
    except Exception:
        logger.exception("再評価中にエラーが発生しました (run_id=%s)", run.run_id)
        raise
    finally:
        setup_logging()


def _evaluate_and_save(
    run: Run,
    config: AppConfig,
    table: pd.DataFrame,
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray | None,
    threshold: ThresholdResult,
    metadata: dict[str, Any],
) -> ExperimentResult:
    predictions = with_predictions(table, threshold.value)
    metrics = _compute_metrics(predictions, anomaly_maps, gt_masks, threshold)
    preprocessor = ImagePreprocessor(config.preprocessing)
    scale = HeatmapScale.from_maps(anomaly_maps)

    if config.output.save_predictions:
        run.save_predictions(predictions)
        run.save_anomaly_maps(anomaly_maps, gt_masks)
    save_visualizations(
        predictions,
        anomaly_maps,
        preprocessor,
        scale,
        run.images_dir,
        save_heatmaps=config.output.save_heatmaps,
        save_overlays=config.output.save_overlays,
    )

    created_at = datetime.now().isoformat(timespec="seconds")
    run.save_metrics(metrics)
    run.save_metadata(
        {
            "run_id": run.run_id,
            "created_at": created_at,
            "phase": PHASES[config.dataset.type],
            "config_path": str(config.config_path),
            "seed": config.project.seed,
            "environment": collect_environment(),
            **metadata,
            "preprocessing": {
                "summary": preprocessor.summary(),
                "steps": preprocessor.enabled_steps(),
            },
            "threshold": threshold.to_dict(),
            "heatmap_scale": dataclasses.asdict(scale),
        }
    )

    image, pixel = metrics["image"], metrics["pixel"] or {}
    summary_path = append_summary(
        run.run_dir.parent,
        {
            "run_id": run.run_id,
            "created_at": created_at,
            "phase": PHASES[config.dataset.type],
            "dataset": config.dataset.type,
            "model": config.model.name,
            "threshold_mode": threshold.mode,
            "threshold": threshold.value,
            "preprocessing_summary": preprocessor.summary(),
            "evaluation_split": EVALUATION_SPLIT,
            "num_images": image["num_images"],
            **{k: image[k] for k in ("accuracy", "precision", "recall", "f1", "auroc", "aupr")},
            "pixel_auroc": pixel.get("pixel_auroc"),
            "pixel_aupro": pixel.get("pixel_aupro"),
            "fp": image["fp"],
            "fn": image["fn"],
            "device": (metadata.get("device") or {}).get("device"),
            "retrained": metadata["retrained"],
            "parent_run_id": metadata["parent_run_id"],
        },
    )
    logger.info("Summary追記: %s", summary_path)
    logger.info("Run完了: %s", run.run_dir)
    return ExperimentResult(
        run=run,
        config=config,
        predictions=predictions,
        anomaly_maps=anomaly_maps,
        gt_masks=gt_masks,
        threshold=threshold,
        metrics=metrics,
        heatmap_scale=scale,
    )


def _compute_metrics(
    predictions: pd.DataFrame,
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray | None,
    threshold: ThresholdResult,
) -> dict[str, Any]:
    """最終評価はtestのみ。validationの値はThreshold決定に使ったデータでの参考値として分ける。"""
    evaluation = evaluate_split(predictions, EVALUATION_SPLIT, anomaly_maps, gt_masks)
    image = evaluation["image"]
    logger.info(
        "Evaluation (%s, %d件): recall=%.4f precision=%.4f f1=%.4f accuracy=%.4f "
        "auroc=%s aupr=%s FN=%d FP=%d",
        EVALUATION_SPLIT,
        image["num_images"],
        image["recall"],
        image["precision"],
        image["f1"],
        image["accuracy"],
        _fmt(image["auroc"]),
        _fmt(image["aupr"]),
        image["fn"],
        image["fp"],
    )
    if evaluation["pixel"]:
        logger.info(
            "Evaluation Pixel-level: pixel_auroc=%s pixel_aupro=%s",
            _fmt(evaluation["pixel"]["pixel_auroc"]),
            _fmt(evaluation["pixel"]["pixel_aupro"]),
        )

    metrics: dict[str, Any] = {
        "evaluation_split": EVALUATION_SPLIT,
        "threshold": threshold.to_dict(),
        "image": image,
        "pixel": evaluation["pixel"],
        "notes": [MANUAL_THRESHOLD_NOTE] if threshold.mode == "manual" else [],
    }
    if (predictions["split"] == "validation").any():
        metrics["threshold_selection_reference"] = {
            "split": "validation",
            "note": "Threshold決定に使用したデータでの参考値。独立した評価ではない。",
            "image": evaluate_split(predictions, "validation")["image"],
        }
    return metrics


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
