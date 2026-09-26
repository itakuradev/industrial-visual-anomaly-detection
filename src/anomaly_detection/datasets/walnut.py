"""独自くるみDataset Adapter（Phase 2）とManifestの作成・検証。

Directory構造（``dataset.root``）::

    data/walnut/
    ├─ raw/
    │  ├─ good/    正常（明確な欠け・割れがない）
    │  ├─ crack/   割れ
    │  └─ chip/    欠け
    └─ manifest.csv

ファイル名は ``<object_id>_<連番>.jpg``（例: ``G001_1.jpg``）。``_`` を含まない場合は
ファイル名（拡張子なし）全体をobject_idとする。同じ個体を複数角度で撮影した画像は
同じobject_idにし、Train / Validation / Testに跨がらないようobject_id単位で分割する。
object_idはDataset全体で一意にする（good / crack / chip で重複させない）。

元画像（``raw/``）は読み込みのみ行い、変更しない。
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.datasets.base import (
    LABEL_DEFECT,
    LABEL_GOOD,
    LABEL_INDEX,
    SAMPLE_COLUMNS,
    SPLITS,
    SUPPORTED_IMAGE_EXTENSIONS,
    DatasetError,
    DatasetNotFoundError,
    log_sample_counts,
    validate_samples,
)
from anomaly_detection.utils.config import AppConfig, ManifestSplitConfig, SplitRatios

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ("image_path", "label", "defect_type", "split", "object_id")
DEFECT_TYPES = ("crack", "chip")
RAW_CLASSES = {
    "good": (LABEL_GOOD, ""),
    "crack": (LABEL_DEFECT, "crack"),
    "chip": (LABEL_DEFECT, "chip"),
}

PLACEMENT_GUIDE = """\
くるみDatasetを次の構造で配置してください。

{root}/
├─ raw/
│  ├─ good/    正常（明確な欠け・割れがない）
│  ├─ crack/   割れ
│  └─ chip/    欠け
└─ manifest.csv

- ファイル名: <object_id>_<連番>.jpg（例: G001_1.jpg, G001_2.jpg, C001_1.jpg）
  同じ個体を複数角度で撮影した画像は同じobject_idにする。object_idはDataset全体で一意にする
- 対応形式: {extensions}（iPhoneのHEICはJPEGへ変換してから配置する）
- manifest.csv は画像配置後に次で作成できる（split比率は設定の dataset.split）:
    from anomaly_detection.datasets.walnut import WalnutAdapter
    WalnutAdapter.from_config(config).build_manifest()"""


class ManifestError(DatasetError):
    """Manifestの列・値・ファイル・分割の不備を表す。"""


def object_id_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def scan_raw_images(root: Path) -> pd.DataFrame:
    """``raw/good|crack|chip`` の画像を列挙する（image_pathはrootからの相対パス）。"""
    rows, unsupported = [], []
    for class_dir, (label, defect_type) in RAW_CLASSES.items():
        directory = root / "raw" / class_dir
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                unsupported.append(path.relative_to(root).as_posix())
                continue
            rows.append(
                {
                    "image_path": path.relative_to(root).as_posix(),
                    "label": label,
                    "defect_type": defect_type,
                    "object_id": object_id_from_filename(path),
                }
            )
    if unsupported:
        raise ManifestError(
            f"対応していない形式のファイルがあります: {len(unsupported)} 件"
            f"（例: {unsupported[:3]}）。"
            f"対応形式: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}（HEICはJPEGへ変換してください）"
        )
    images = pd.DataFrame(rows, columns=["image_path", "label", "defect_type", "object_id"])
    if images.empty:
        raise DatasetNotFoundError(
            f"raw/ に画像がありません: {root / 'raw'}\n\n"
            + PLACEMENT_GUIDE.format(root=root, extensions=", ".join(SUPPORTED_IMAGE_EXTENSIONS))
        )
    classes_per_object = images.groupby("object_id")["defect_type"].nunique()
    conflicts = classes_per_object[classes_per_object > 1].index.tolist()
    if conflicts:
        raise ManifestError(
            f"同じobject_idが複数のクラス（good / crack / chip）にあります: {conflicts[:5]}。"
            "object_idはDataset全体で一意にしてください（例: 正常 G001、割れ C001、欠け K001）。"
        )
    return images


def allocate(n: int, ratios: SplitRatios) -> dict[str, int]:
    """n個体を比率に従って各splitへ割り当てる件数（最大剰余法）。"""
    targets = {split: n * getattr(ratios, split) for split in SPLITS}
    counts = {split: int(np.floor(value)) for split, value in targets.items()}
    remainders = sorted(SPLITS, key=lambda split: targets[split] - counts[split], reverse=True)
    for split in remainders[: n - sum(counts.values())]:
        counts[split] += 1
    return counts


def assign_splits(images: pd.DataFrame, split: ManifestSplitConfig, seed: int) -> pd.DataFrame:
    """object_id単位でsplitを割り当てる。異常はdefect_typeごとに比率を保つ。"""
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for (label, _defect_type), group in images.groupby(["label", "defect_type"], sort=True):
        ratios = split.good if label == LABEL_GOOD else split.defect
        object_ids = sorted(group["object_id"].unique())
        shuffled = [object_ids[i] for i in rng.permutation(len(object_ids))]
        start = 0
        for split_name, count in allocate(len(shuffled), ratios).items():
            for object_id in shuffled[start : start + count]:
                assignment[object_id] = split_name
            start += count
    manifest = images.assign(split=images["object_id"].map(assignment))
    return manifest[list(MANIFEST_COLUMNS)]


def load_manifest(path: Path) -> pd.DataFrame:
    try:
        manifest = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ManifestError(f"Manifestを読み込めません: {path}\n{e}") from e
    return manifest.apply(lambda column: column.str.strip())


def find_object_leaks(manifest: pd.DataFrame) -> pd.DataFrame:
    """複数のsplitに跨がっているobject_idの一覧（空ならLeakageなし）。"""
    splits = manifest.groupby("object_id")["split"].agg(lambda s: sorted(set(s)))
    leaks = splits[splits.map(len) > 1]
    return pd.DataFrame({"object_id": leaks.index, "splits": leaks.to_numpy()})


def validate_manifest(manifest: pd.DataFrame, root: Path) -> None:
    """Manifestの必須列・値・ファイル存在・Train混入・object_id Leakageを検証する。"""
    missing_columns = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing_columns:
        raise ManifestError(
            f"Manifestに必要な列がありません: {', '.join(missing_columns)}"
            f"（必要な列: {', '.join(MANIFEST_COLUMNS)}）"
        )
    if manifest.empty:
        raise ManifestError("Manifestに画像が登録されていません。")

    def fail(mask: pd.Series, message: str) -> None:
        if mask.any():
            rows = manifest.loc[mask, "image_path"].head(3).tolist()
            raise ManifestError(f"{message}: {int(mask.sum())} 件（例: {rows}）")

    fail(
        ~manifest["label"].isin(list(LABEL_INDEX)), "labelは good / defect のいずれかにしてください"
    )
    fail(
        (manifest["label"] == LABEL_GOOD) & (manifest["defect_type"] != ""),
        "label=good の defect_type は空にしてください",
    )
    fail(
        (manifest["label"] == LABEL_DEFECT) & ~manifest["defect_type"].isin(DEFECT_TYPES),
        f"label=defect の defect_type は {' / '.join(DEFECT_TYPES)} のいずれかにしてください",
    )
    fail(~manifest["split"].isin(SPLITS), f"splitは {' / '.join(SPLITS)} のいずれかにしてください")
    fail(manifest["object_id"] == "", "object_idが空です")
    fail(
        (manifest["split"] == "train") & (manifest["label"] == LABEL_DEFECT),
        "Trainingに異常画像が含まれています（PatchCoreの学習には正常画像のみを使用します）",
    )
    fail(~manifest["image_path"].map(lambda p: (root / p).is_file()), "画像ファイルが存在しません")

    leaks = find_object_leaks(manifest)
    if not leaks.empty:
        examples = leaks.head(3).to_dict("records")
        raise ManifestError(
            f"同じobject_idが複数のsplitに跨がっています: {len(leaks)} 件（例: {examples}）。"
            "同一個体の画像は同じsplitにしてください（Data Leakage防止）。"
        )


def count_by_defect_type(samples: pd.DataFrame) -> pd.DataFrame:
    """split × (good / crack / chip) の件数表。"""
    kind = samples["defect_type"].where(samples["label"] == LABEL_DEFECT, LABEL_GOOD)
    table = pd.crosstab(samples["split"], kind).reindex(
        index=list(SPLITS), columns=[LABEL_GOOD, *DEFECT_TYPES], fill_value=0
    )
    table.loc["total"] = table.sum()
    return table


class WalnutAdapter:
    """Manifestを共通Sample Tableへ変換する。Maskは作成しないためImage-level評価のみ。"""

    def __init__(
        self, root: Path, manifest_path: Path, split: ManifestSplitConfig, seed: int
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.seed = seed

    @classmethod
    def from_config(cls, config: AppConfig) -> WalnutAdapter:
        dataset = config.dataset
        return cls(
            root=config.resolve_path(dataset.root),
            manifest_path=config.resolve_path(dataset.manifest),
            split=dataset.split,
            seed=config.project.seed,
        )

    @property
    def guide(self) -> str:
        return PLACEMENT_GUIDE.format(
            root=self.root, extensions=", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        )

    def validate(self) -> None:
        if not self.root.is_dir():
            raise DatasetNotFoundError(f"Dataset Rootが存在しません: {self.root}\n\n{self.guide}")
        if not self.manifest_path.is_file():
            raise DatasetNotFoundError(
                f"Manifestが存在しません: {self.manifest_path}\n\n{self.guide}"
            )

    def build_manifest(self, overwrite: bool = False) -> pd.DataFrame:
        """``raw/`` の画像からManifestを作成して保存する。

        手作業で修正したManifestを失わないよう、
        既存ファイルは ``overwrite=True`` の時だけ上書きする。
        """
        if self.manifest_path.exists() and not overwrite:
            raise ManifestError(
                f"Manifestは既に存在します: {self.manifest_path}。"
                "作り直す場合は build_manifest(overwrite=True) を指定してください。"
            )
        manifest = assign_splits(scan_raw_images(self.root), self.split, self.seed)
        validate_manifest(manifest, self.root)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(self.manifest_path, index=False, encoding="utf-8")
        logger.info("Manifest作成: %s (%d 件)", self.manifest_path, len(manifest))
        return manifest

    def load_samples(self) -> pd.DataFrame:
        self.validate()
        logger.info("Dataset読み込み開始: Walnut (%s)", self.manifest_path)
        manifest = load_manifest(self.manifest_path)
        validate_manifest(manifest, self.root)
        samples = pd.DataFrame(
            {
                "image_path": manifest["image_path"].map(lambda p: str((self.root / p).resolve())),
                "label": manifest["label"],
                "label_index": manifest["label"].map(LABEL_INDEX).astype(int),
                "mask_path": None,
                "split": manifest["split"],
                "defect_type": manifest["defect_type"],
                "object_id": manifest["object_id"],
            }
        )[list(SAMPLE_COLUMNS)]
        validate_samples(samples)
        logger.info(
            "Dataset読み込み完了: %d 件, object_id %d 個",
            len(samples),
            samples["object_id"].nunique(),
        )
        log_sample_counts(samples)
        return samples

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "dataset": "walnut",
            "root": str(self.root),
            "manifest": str(self.manifest_path),
            "manifest_split": dataclasses.asdict(self.split),
            "seed": self.seed,
            "pixel_level_evaluation": False,
        }
        if self.manifest_path.is_file():
            manifest = load_manifest(self.manifest_path)
            if "object_id" in manifest.columns:
                info["num_objects"] = int(manifest["object_id"].nunique())
        return info
