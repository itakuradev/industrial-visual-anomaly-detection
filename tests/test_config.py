from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from anomaly_detection.utils.config import ConfigError, load_config, save_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG = PROJECT_ROOT / "configs" / "phase1_mvtec_vial.yaml"
PHASE2_CONFIG = PROJECT_ROOT / "configs" / "phase2_walnut.yaml"


@pytest.fixture
def base_raw() -> dict[str, Any]:
    with PHASE1_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_config(tmp_path: Path, raw: Any) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def modified(raw: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    *parents, last = dotted_key.split(".")
    target = raw
    for key in parents:
        target = target[key]
    target[last] = value
    return raw


def removed(raw: dict[str, Any], dotted_key: str) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    *parents, last = dotted_key.split(".")
    target = raw
    for key in parents:
        target = target[key]
    del target[last]
    return raw


class TestValidConfig:
    def test_phase1_config_loads(self) -> None:
        config = load_config(PHASE1_CONFIG)
        assert config.project.seed == 42
        assert config.dataset.type == "mvtec_ad2"
        assert config.dataset.category == "vial"
        assert config.dataset.threshold_split_ratio == 0.5
        assert config.model.layers == ("layer2", "layer3")
        assert config.preprocessing.resize.width == 256
        assert config.threshold.mode == "auto"
        assert config.threshold.manual_value is None
        assert config.runtime.device == "auto"

    def test_phase2_config_loads(self) -> None:
        config = load_config(PHASE2_CONFIG)
        assert config.dataset.type == "walnut"
        assert config.dataset.manifest == "data/walnut/manifest.csv"

    def test_relative_paths_resolve_from_project_root(self) -> None:
        config = load_config(PHASE1_CONFIG)
        assert config.root_dir == PROJECT_ROOT
        assert config.resolve_path(config.dataset.root) == PROJECT_ROOT / "data" / "mvtec_ad2"

    def test_project_root_found_from_nested_config(self, tmp_path: Path, base_raw) -> None:
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config = load_config(write_config(config_dir, base_raw))
        assert config.root_dir == tmp_path.resolve()

    def test_manual_threshold(self, tmp_path: Path, base_raw) -> None:
        raw = modified(
            modified(base_raw, "threshold.mode", "manual"), "threshold.manual_value", 3.2
        )
        config = load_config(write_config(tmp_path, raw))
        assert config.threshold.mode == "manual"
        assert config.threshold.manual_value == 3.2

    def test_snapshot_round_trip(self, tmp_path: Path) -> None:
        config = load_config(PHASE1_CONFIG)
        snapshot = tmp_path / "snapshot.yaml"
        save_config(config, snapshot)
        reloaded = load_config(snapshot)
        assert reloaded.to_dict() == config.to_dict()


class TestInvalidConfig:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="見つかりません"):
            load_config(tmp_path / "missing.yaml")

    def test_broken_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("project: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML書式"):
            load_config(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    @pytest.mark.parametrize(
        "key",
        [
            "project",
            "project.seed",
            "dataset.root",
            "dataset.category",
            "model.backbone",
            "preprocessing.resize",
            "threshold.mode",
            "runtime.device",
            "output.run_root",
        ],
    )
    def test_missing_required(self, tmp_path: Path, base_raw, key: str) -> None:
        with pytest.raises(ConfigError, match=f"必須項目 {key} "):
            load_config(write_config(tmp_path, removed(base_raw, key)))

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("project.seed", -1, "project.seed"),
            ("project.seed", "42", "project.seed"),
            ("dataset.type", "mvtec", "dataset.type"),
            ("dataset.threshold_split_ratio", 0, "dataset.threshold_split_ratio"),
            ("dataset.threshold_split_ratio", 1.0, "dataset.threshold_split_ratio"),
            ("model.name", "padim", "model.name"),
            ("model.layers", [], "model.layers"),
            ("model.coreset_sampling_ratio", 0, "model.coreset_sampling_ratio"),
            ("model.coreset_sampling_ratio", 1.5, "model.coreset_sampling_ratio"),
            ("model.num_neighbors", 0, "model.num_neighbors"),
            ("preprocessing.resize.width", 0, "preprocessing.resize.width"),
            ("preprocessing.resize.enabled", "yes", "preprocessing.resize.enabled"),
            ("preprocessing.clahe.clip_limit", -1, "preprocessing.clahe.clip_limit"),
            ("preprocessing.denoise.kernel_size", 4, "preprocessing.denoise.kernel_size"),
            ("preprocessing.denoise.method", "bilateral", "preprocessing.denoise.method"),
            ("threshold.mode", "adaptive", "threshold.mode"),
            ("runtime.device", "gpu", "runtime.device"),
            ("runtime.batch_size", 0, "runtime.batch_size"),
        ],
    )
    def test_invalid_value(self, tmp_path: Path, base_raw, key: str, value: Any, message: str):
        with pytest.raises(ConfigError, match=message):
            load_config(write_config(tmp_path, modified(base_raw, key, value)))

    def test_manual_threshold_requires_value(self, tmp_path: Path, base_raw) -> None:
        raw = modified(base_raw, "threshold.mode", "manual")
        with pytest.raises(ConfigError, match="manual_value"):
            load_config(write_config(tmp_path, raw))

    def test_crop_enabled_requires_size(self, tmp_path: Path, base_raw) -> None:
        raw = modified(base_raw, "preprocessing.crop.enabled", True)
        with pytest.raises(ConfigError, match="crop.width"):
            load_config(write_config(tmp_path, raw))

    def test_unknown_key(self, tmp_path: Path, base_raw) -> None:
        raw = modified(base_raw, "preprocessing.clahe.cliplimit", 2.0)
        with pytest.raises(ConfigError, match="preprocessing.clahe.cliplimit"):
            load_config(write_config(tmp_path, raw))

    def test_walnut_requires_manifest(self, tmp_path: Path) -> None:
        with PHASE2_CONFIG.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        with pytest.raises(ConfigError, match="dataset.manifest"):
            load_config(write_config(tmp_path, removed(raw, "dataset.manifest")))
