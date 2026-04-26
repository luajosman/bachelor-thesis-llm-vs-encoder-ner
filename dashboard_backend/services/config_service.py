"""Config discovery and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dashboard_backend.models.schemas import ConfigDetail, ConfigSummary
from dashboard_backend.utils.paths import CONFIGS_DIR, as_display_path
from dashboard_backend.utils.yaml_json import load_yaml_file


def _normalize_model_type(raw_model_type: str | None) -> str:
    return "encoder" if str(raw_model_type).lower() == "encoder" else "llm"


def _normalize_regime(config: dict[str, Any]) -> str:
    model_type = _normalize_model_type(str(config.get("model_type")))
    if model_type == "encoder":
        return "encoder"

    mode = str(config.get("mode", "")).lower()
    experiment_name = str(config.get("experiment_name", "")).lower()
    if mode == "zeroshot" or "zeroshot" in experiment_name:
        return "zeroshot"
    return "lora"


def _config_to_summary(path: Path, config: dict[str, Any]) -> ConfigSummary:
    experiment_name = str(config.get("experiment_name", path.stem))
    model_name = str(config.get("model_name", ""))
    dataset = str(config.get("dataset", "multinerd"))
    output_dir = str(config.get("output_dir", f"results/{dataset}/{experiment_name}"))
    return ConfigSummary(
        id=path.stem,
        name=path.name,
        path=as_display_path(path),
        experiment_name=experiment_name,
        model_name=model_name,
        model_type=_normalize_model_type(str(config.get("model_type"))),
        regime=_normalize_regime(config),
        dataset=dataset,
        output_dir=output_dir,
    )


def list_configs() -> list[ConfigSummary]:
    """Return all YAML experiment configs from the ML project."""
    configs: list[ConfigSummary] = []
    if not CONFIGS_DIR.exists():
        return configs

    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        raw = load_yaml_file(path)
        configs.append(_config_to_summary(path, raw))
    return configs


def get_config(config_id: str) -> ConfigDetail | None:
    """Return one config by file stem id."""
    path = CONFIGS_DIR / f"{config_id}.yaml"
    if not path.exists():
        return None
    raw = load_yaml_file(path)
    summary = _config_to_summary(path, raw)
    return ConfigDetail(**summary.model_dump(), config=raw)


def get_config_by_experiment_name(experiment_name: str) -> ConfigSummary | None:
    """Resolve a config based on its ``experiment_name`` field."""
    for config in list_configs():
        if config.experiment_name == experiment_name:
            return config
    return None
