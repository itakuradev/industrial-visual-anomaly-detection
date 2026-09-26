"""YAML設定の読み込みと検証。

設定値はdataclassへ変換し、必須項目の不足・未知のキー・不正値を ``ConfigError`` として報告する。
相対パスはプロジェクトルート（``pyproject.toml`` を含むディレクトリ）を基準に解決するため、
Notebookのカレントディレクトリに依存しない。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DATASET_TYPES = ("mvtec_ad2", "walnut")
MODEL_NAMES = ("patchcore",)
THRESHOLD_MODES = ("auto", "manual")
DEVICES = ("auto", "cuda", "cpu")
DENOISE_METHODS = ("gaussian", "median")


class ConfigError(ValueError):
    """設定ファイルの不存在・書式不正・値不正を表す。"""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class SplitRatios:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class ManifestSplitConfig:
    """独自DatasetのManifest作成時に使うsplit比率（object_id単位で割り当てる）。"""

    good: SplitRatios
    defect: SplitRatios  # 異常画像は学習に使わないため train は 0


@dataclass(frozen=True)
class DatasetConfig:
    type: str
    root: str
    category: str | None = None
    manifest: str | None = None
    threshold_split_ratio: float | None = None
    split: ManifestSplitConfig | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backbone: str
    layers: tuple[str, ...]
    coreset_sampling_ratio: float
    num_neighbors: int


@dataclass(frozen=True)
class ResizeConfig:
    enabled: bool
    width: int
    height: int


@dataclass(frozen=True)
class CropConfig:
    enabled: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class ContrastConfig:
    enabled: bool
    alpha: float = 1.0
    beta: float = 0.0


@dataclass(frozen=True)
class ClaheConfig:
    enabled: bool
    clip_limit: float = 2.0
    tile_grid_size: int = 8


@dataclass(frozen=True)
class DenoiseConfig:
    enabled: bool
    method: str = "gaussian"
    kernel_size: int = 3


@dataclass(frozen=True)
class NormalizeConfig:
    enabled: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    resize: ResizeConfig
    crop: CropConfig
    contrast: ContrastConfig
    clahe: ClaheConfig
    denoise: DenoiseConfig
    normalize: NormalizeConfig


@dataclass(frozen=True)
class ThresholdConfig:
    mode: str
    manual_value: float | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class OutputConfig:
    run_root: str
    save_model: bool
    save_predictions: bool
    save_heatmaps: bool
    save_overlays: bool


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    threshold: ThresholdConfig
    runtime: RuntimeConfig
    output: OutputConfig
    config_path: Path
    root_dir: Path

    def resolve_path(self, path: str | Path) -> Path:
        """設定内の相対パスをプロジェクトルート基準の絶対パスへ変換する。"""
        path = Path(path)
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        """設定値を辞書として返す（パスは設定ファイルに書かれた値のまま、未設定の項目は除く）。"""
        data = dataclasses.asdict(self)
        data.pop("config_path")
        data.pop("root_dir")
        data["model"]["layers"] = list(data["model"]["layers"])
        return _drop_none(data)


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _drop_none(value) if isinstance(value, dict) else value
        for key, value in data.items()
        if value is not None
    }


def find_project_root(start: Path) -> Path:
    """``start`` から親方向へ ``pyproject.toml`` を探し、見つからなければ ``start`` を返す。"""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def load_config(path: str | Path) -> AppConfig:
    """YAML設定ファイルを読み込み、検証済みの ``AppConfig`` を返す。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path.resolve()}")
    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"設定ファイルのYAML書式が不正です: {path}\n{e}") from e
    return parse_config(raw, config_path=path.resolve(), root_dir=find_project_root(path.parent))


def save_config(config: AppConfig, path: str | Path) -> None:
    """設定のSnapshotをYAMLとして保存する。"""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)


def parse_config(raw: Any, config_path: Path, root_dir: Path) -> AppConfig:
    """YAMLから読み込んだ辞書を検証し ``AppConfig`` へ変換する。"""
    top = _Section(raw, "")
    config = AppConfig(
        project=_parse_project(top.section("project")),
        dataset=_parse_dataset(top.section("dataset")),
        model=_parse_model(top.section("model")),
        preprocessing=_parse_preprocessing(top.section("preprocessing")),
        threshold=_parse_threshold(top.section("threshold")),
        runtime=_parse_runtime(top.section("runtime")),
        output=_parse_output(top.section("output")),
        config_path=config_path,
        root_dir=root_dir,
    )
    top.finish()
    return config


class _Section:
    """設定の1セクションを読み取り、使用しなかったキーを未知のキーとして検出する。"""

    def __init__(self, data: Any, name: str) -> None:
        if not isinstance(data, dict):
            label = name or "設定ファイルのトップレベル"
            raise ConfigError(f"{label} はキーと値の組（mapping）である必要があります。")
        self.data = data
        self.name = name
        self.used: set[str] = set()

    def _key(self, key: str) -> str:
        return f"{self.name}.{key}" if self.name else key

    def _get(self, key: str, required: bool) -> Any:
        self.used.add(key)
        if key not in self.data or self.data[key] is None:
            if required:
                raise ConfigError(f"必須項目 {self._key(key)} が設定されていません。")
            return None
        return self.data[key]

    def section(self, key: str) -> _Section:
        return _Section(self._get(key, required=True), self._key(key))

    def str_(
        self, key: str, choices: tuple[str, ...] | None = None, required: bool = True
    ) -> str | None:
        value = self._get(key, required)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ConfigError(
                f"{self._key(key)} は空でない文字列である必要があります（値: {value!r}）。"
            )
        if choices is not None and value not in choices:
            raise ConfigError(
                f"{self._key(key)} は {', '.join(choices)} のいずれかである必要があります"
                f"（値: {value!r}）。"
            )
        return value

    def bool_(self, key: str) -> bool:
        value = self._get(key, required=True)
        if not isinstance(value, bool):
            raise ConfigError(
                f"{self._key(key)} は true / false である必要があります（値: {value!r}）。"
            )
        return value

    def int_(self, key: str, minimum: int | None = None, default: int | None = None) -> int:
        value = self._get(key, required=default is None)
        if value is None:
            return default  # type: ignore[return-value]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._key(key)} は整数である必要があります（値: {value!r}）。")
        if minimum is not None and value < minimum:
            raise ConfigError(
                f"{self._key(key)} は {minimum} 以上である必要があります（値: {value}）。"
            )
        return value

    def float_(
        self,
        key: str,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
        default: float | None = None,
        required: bool = True,
    ) -> float | None:
        value = self._get(key, required=required and default is None)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{self._key(key)} は数値である必要があります（値: {value!r}）。")
        value = float(value)
        if minimum is not None and (value <= minimum if exclusive_minimum else value < minimum):
            op = "より大きい" if exclusive_minimum else "以上である"
            raise ConfigError(f"{self._key(key)} は {minimum} {op}必要があります（値: {value}）。")
        if maximum is not None and value > maximum:
            raise ConfigError(
                f"{self._key(key)} は {maximum} 以下である必要があります（値: {value}）。"
            )
        return value

    def str_list(self, key: str) -> tuple[str, ...]:
        value = self._get(key, required=True)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(v, str) and v for v in value)
        ):
            raise ConfigError(
                f"{self._key(key)} は空でない文字列のリストである必要があります（値: {value!r}）。"
            )
        return tuple(value)

    def finish(self) -> None:
        unknown = sorted(set(self.data) - self.used)
        if unknown:
            keys = ", ".join(self._key(k) for k in unknown)
            raise ConfigError(f"未知の設定項目があります: {keys}（綴りを確認してください）。")


def _parse_project(s: _Section) -> ProjectConfig:
    config = ProjectConfig(name=s.str_("name"), seed=s.int_("seed", minimum=0))
    s.finish()
    return config


def _parse_dataset(s: _Section) -> DatasetConfig:
    dataset_type = s.str_("type", choices=DATASET_TYPES)
    root = s.str_("root")
    if dataset_type == "mvtec_ad2":
        config = DatasetConfig(
            type=dataset_type,
            root=root,
            category=s.str_("category"),
            threshold_split_ratio=s.float_(
                "threshold_split_ratio", minimum=0.0, maximum=1.0, exclusive_minimum=True
            ),
        )
        if config.threshold_split_ratio == 1.0:
            raise ConfigError(
                "dataset.threshold_split_ratio は 1 未満である必要があります"
                "（最終評価用のデータが残らないため）。"
            )
    else:
        config = DatasetConfig(
            type=dataset_type,
            root=root,
            manifest=s.str_("manifest"),
            split=_parse_manifest_split(s.section("split")),
        )
    s.finish()
    return config


def _parse_manifest_split(s: _Section) -> ManifestSplitConfig:
    ratios = {}
    for label in ("good", "defect"):
        r = s.section(label)
        values = {
            name: r.float_(name, minimum=0.0, maximum=1.0, default=0.0)
            for name in ("train", "validation", "test")
        }
        r.finish()
        if abs(sum(values.values()) - 1.0) > 1e-6:
            raise ConfigError(
                f"dataset.split.{label} の train / validation / test の合計は"
                f" 1 である必要があります（値: {values}）。"
            )
        ratios[label] = SplitRatios(**values)
    if ratios["defect"].train != 0.0:
        raise ConfigError(
            "dataset.split.defect.train は 0 である必要があります"
            "（PatchCoreの学習には正常画像のみを使用します）。"
        )
    s.finish()
    return ManifestSplitConfig(good=ratios["good"], defect=ratios["defect"])


def _parse_model(s: _Section) -> ModelConfig:
    config = ModelConfig(
        name=s.str_("name", choices=MODEL_NAMES),
        backbone=s.str_("backbone"),
        layers=s.str_list("layers"),
        coreset_sampling_ratio=s.float_(
            "coreset_sampling_ratio", minimum=0.0, maximum=1.0, exclusive_minimum=True
        ),
        num_neighbors=s.int_("num_neighbors", minimum=1),
    )
    s.finish()
    return config


def _parse_preprocessing(s: _Section) -> PreprocessingConfig:
    r = s.section("resize")
    resize = ResizeConfig(
        enabled=r.bool_("enabled"),
        width=r.int_("width", minimum=1),
        height=r.int_("height", minimum=1),
    )
    r.finish()

    c = s.section("crop")
    crop = CropConfig(
        enabled=c.bool_("enabled"),
        x=c.int_("x", minimum=0, default=0),
        y=c.int_("y", minimum=0, default=0),
        width=c.int_("width", minimum=0, default=0),
        height=c.int_("height", minimum=0, default=0),
    )
    if crop.enabled and (crop.width == 0 or crop.height == 0):
        raise ConfigError(
            "preprocessing.crop.enabled が true の場合は crop.width と crop.height を"
            " 1 以上で設定してください。"
        )
    c.finish()

    ct = s.section("contrast")
    contrast = ContrastConfig(
        enabled=ct.bool_("enabled"),
        alpha=ct.float_("alpha", minimum=0.0, exclusive_minimum=True, default=1.0),
        beta=ct.float_("beta", default=0.0),
    )
    ct.finish()

    cl = s.section("clahe")
    clahe = ClaheConfig(
        enabled=cl.bool_("enabled"),
        clip_limit=cl.float_("clip_limit", minimum=0.0, exclusive_minimum=True, default=2.0),
        tile_grid_size=cl.int_("tile_grid_size", minimum=1, default=8),
    )
    cl.finish()

    d = s.section("denoise")
    denoise = DenoiseConfig(
        enabled=d.bool_("enabled"),
        method=d.str_("method", choices=DENOISE_METHODS, required=False) or "gaussian",
        kernel_size=d.int_("kernel_size", minimum=1, default=3),
    )
    if denoise.kernel_size % 2 == 0:
        raise ConfigError(
            "preprocessing.denoise.kernel_size は奇数である必要があります"
            f"（値: {denoise.kernel_size}）。"
        )
    d.finish()

    n = s.section("normalize")
    normalize = NormalizeConfig(enabled=n.bool_("enabled"))
    n.finish()

    s.finish()
    return PreprocessingConfig(
        resize=resize,
        crop=crop,
        contrast=contrast,
        clahe=clahe,
        denoise=denoise,
        normalize=normalize,
    )


def _parse_threshold(s: _Section) -> ThresholdConfig:
    config = ThresholdConfig(
        mode=s.str_("mode", choices=THRESHOLD_MODES),
        manual_value=s.float_("manual_value", required=False),
    )
    if config.mode == "manual" and config.manual_value is None:
        raise ConfigError(
            "threshold.mode が manual の場合は threshold.manual_value（PatchCoreの生スコア単位）"
            "を設定してください。"
        )
    s.finish()
    return config


def _parse_runtime(s: _Section) -> RuntimeConfig:
    config = RuntimeConfig(
        device=s.str_("device", choices=DEVICES),
        batch_size=s.int_("batch_size", minimum=1),
        num_workers=s.int_("num_workers", minimum=0),
    )
    s.finish()
    return config


def _parse_output(s: _Section) -> OutputConfig:
    config = OutputConfig(
        run_root=s.str_("run_root"),
        save_model=s.bool_("save_model"),
        save_predictions=s.bool_("save_predictions"),
        save_heatmaps=s.bool_("save_heatmaps"),
        save_overlays=s.bool_("save_overlays"),
    )
    s.finish()
    return config
