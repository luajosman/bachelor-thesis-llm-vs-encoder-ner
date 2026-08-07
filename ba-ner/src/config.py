"""
Final experiment configuration registry.

The project intentionally supports one benchmark setup:
MultiNERD English with two DeBERTa encoders, three Qwen3.5 zero-shot runs,
and three Qwen3.5 LoRA/QLoRA runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


DATASET_NAME = "multinerd"
DATASET_LANGUAGE = "en"
RESULTS_ROOT = "results"

ENCODER_MODELS = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
}

QWEN_MODELS = {
    "qwen35-08b": "Qwen/Qwen3.5-0.8B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-27b": "Qwen/Qwen3.5-27B",
}


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    config_path: str
    experiment_name: str
    model_name: str
    model_type: str
    regime: str

    @property
    def output_dir(self) -> str:
        return f"{RESULTS_ROOT}/{DATASET_NAME}/{self.experiment_name}"


FINAL_EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "deberta_base": ExperimentSpec(
        key="deberta_base",
        config_path="configs/deberta_base.yaml",
        experiment_name="deberta-v3-base",
        model_name=ENCODER_MODELS["deberta-v3-base"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_large": ExperimentSpec(
        key="deberta_large",
        config_path="configs/deberta_large.yaml",
        experiment_name="deberta-v3-large",
        model_name=ENCODER_MODELS["deberta-v3-large"],
        model_type="encoder",
        regime="encoder",
    ),
    "qwen35_08b_zs": ExperimentSpec(
        key="qwen35_08b_zs",
        config_path="configs/qwen35_08b_zeroshot.yaml",
        experiment_name="qwen35-08b-zeroshot",
        model_name=QWEN_MODELS["qwen35-08b"],
        model_type="decoder",
        regime="llm_zeroshot",
    ),
    "qwen35_4b_zs": ExperimentSpec(
        key="qwen35_4b_zs",
        config_path="configs/qwen35_4b_zeroshot.yaml",
        experiment_name="qwen35-4b-zeroshot",
        model_name=QWEN_MODELS["qwen35-4b"],
        model_type="decoder",
        regime="llm_zeroshot",
    ),
    "qwen35_27b_zs": ExperimentSpec(
        key="qwen35_27b_zs",
        config_path="configs/qwen35_27b_zeroshot.yaml",
        experiment_name="qwen35-27b-zeroshot",
        model_name=QWEN_MODELS["qwen35-27b"],
        model_type="decoder",
        regime="llm_zeroshot",
    ),
    "qwen35_08b": ExperimentSpec(
        key="qwen35_08b",
        config_path="configs/qwen35_08b.yaml",
        experiment_name="qwen35-08b-qlora",
        model_name=QWEN_MODELS["qwen35-08b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_4b": ExperimentSpec(
        key="qwen35_4b",
        config_path="configs/qwen35_4b.yaml",
        experiment_name="qwen35-4b-qlora",
        model_name=QWEN_MODELS["qwen35-4b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_27b": ExperimentSpec(
        key="qwen35_27b",
        config_path="configs/qwen35_27b.yaml",
        experiment_name="qwen35-27b-qlora",
        model_name=QWEN_MODELS["qwen35-27b"],
        model_type="decoder",
        regime="llm_lora",
    ),
}


# The historical eight-experiment matrix above remains unchanged.  The seed
# study is registered separately so existing callers of ``FINAL_EXPERIMENTS``
# keep their original behaviour (especially the deterministic zero-shot runs).
SEED_STUDY_EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "deberta_base_canonical_seed42": ExperimentSpec(
        key="deberta_base_canonical_seed42",
        config_path="configs/deberta_base_canonical.yaml",
        experiment_name="deberta-v3-base-canonical",
        model_name=ENCODER_MODELS["deberta-v3-base"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_base_seed123": ExperimentSpec(
        key="deberta_base_seed123",
        config_path="configs/deberta_base_seed123.yaml",
        experiment_name="deberta-v3-base-canonical-seed123",
        model_name=ENCODER_MODELS["deberta-v3-base"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_base_seed456": ExperimentSpec(
        key="deberta_base_seed456",
        config_path="configs/deberta_base_seed456.yaml",
        experiment_name="deberta-v3-base-canonical-seed456",
        model_name=ENCODER_MODELS["deberta-v3-base"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_large_canonical_seed42": ExperimentSpec(
        key="deberta_large_canonical_seed42",
        config_path="configs/deberta_large_canonical.yaml",
        experiment_name="deberta-v3-large-canonical",
        model_name=ENCODER_MODELS["deberta-v3-large"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_large_seed123": ExperimentSpec(
        key="deberta_large_seed123",
        config_path="configs/deberta_large_seed123.yaml",
        experiment_name="deberta-v3-large-canonical-seed123",
        model_name=ENCODER_MODELS["deberta-v3-large"],
        model_type="encoder",
        regime="encoder",
    ),
    "deberta_large_seed456": ExperimentSpec(
        key="deberta_large_seed456",
        config_path="configs/deberta_large_seed456.yaml",
        experiment_name="deberta-v3-large-canonical-seed456",
        model_name=ENCODER_MODELS["deberta-v3-large"],
        model_type="encoder",
        regime="encoder",
    ),
    "qwen35_08b_canonical_seed42": ExperimentSpec(
        key="qwen35_08b_canonical_seed42",
        config_path="configs/qwen35_08b_canonical.yaml",
        experiment_name="qwen35-08b-qlora-canonical",
        model_name=QWEN_MODELS["qwen35-08b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_08b_seed123": ExperimentSpec(
        key="qwen35_08b_seed123",
        config_path="configs/qwen35_08b_seed123.yaml",
        experiment_name="qwen35-08b-qlora-canonical-seed123",
        model_name=QWEN_MODELS["qwen35-08b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_08b_seed456": ExperimentSpec(
        key="qwen35_08b_seed456",
        config_path="configs/qwen35_08b_seed456.yaml",
        experiment_name="qwen35-08b-qlora-canonical-seed456",
        model_name=QWEN_MODELS["qwen35-08b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_4b_canonical_seed42": ExperimentSpec(
        key="qwen35_4b_canonical_seed42",
        config_path="configs/qwen35_4b_canonical.yaml",
        experiment_name="qwen35-4b-qlora-canonical",
        model_name=QWEN_MODELS["qwen35-4b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_4b_seed123": ExperimentSpec(
        key="qwen35_4b_seed123",
        config_path="configs/qwen35_4b_seed123.yaml",
        experiment_name="qwen35-4b-qlora-canonical-seed123",
        model_name=QWEN_MODELS["qwen35-4b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_4b_seed456": ExperimentSpec(
        key="qwen35_4b_seed456",
        config_path="configs/qwen35_4b_seed456.yaml",
        experiment_name="qwen35-4b-qlora-canonical-seed456",
        model_name=QWEN_MODELS["qwen35-4b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_27b_3ep_seed42": ExperimentSpec(
        key="qwen35_27b_3ep_seed42",
        config_path="configs/qwen35_27b_3ep.yaml",
        experiment_name="qwen35-27b-qlora-3ep",
        model_name=QWEN_MODELS["qwen35-27b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_27b_3ep_seed123": ExperimentSpec(
        key="qwen35_27b_3ep_seed123",
        config_path="configs/qwen35_27b_3ep_seed123.yaml",
        experiment_name="qwen35-27b-qlora-3ep-seed123",
        model_name=QWEN_MODELS["qwen35-27b"],
        model_type="decoder",
        regime="llm_lora",
    ),
    "qwen35_27b_3ep_seed456": ExperimentSpec(
        key="qwen35_27b_3ep_seed456",
        config_path="configs/qwen35_27b_3ep_seed456.yaml",
        experiment_name="qwen35-27b-qlora-3ep-seed456",
        model_name=QWEN_MODELS["qwen35-27b"],
        model_type="decoder",
        regime="llm_lora",
    ),
}

ALL_EXPERIMENTS: Dict[str, ExperimentSpec] = {
    **FINAL_EXPERIMENTS,
    **SEED_STUDY_EXPERIMENTS,
}


def final_config_paths(
    *,
    model_type: Optional[str] = None,
    regime: Optional[str] = None,
) -> list[str]:
    specs = _filter_specs(FINAL_EXPERIMENTS.values(), model_type=model_type, regime=regime)
    return [spec.config_path for spec in specs]


def experiment_by_config_path(config_path: str | Path) -> ExperimentSpec:
    normalized = Path(config_path).as_posix()
    for spec in ALL_EXPERIMENTS.values():
        if normalized == spec.config_path or normalized.endswith("/" + spec.config_path):
            return spec
    raise ValueError(f"Unsupported experiment config: {config_path}")


def load_experiment_config(
    config_path: str | Path,
    *,
    expected_model_type: Optional[str] = None,
    expected_regime: Optional[str] = None,
) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_experiment_config(
        cfg,
        config_path=config_path,
        expected_model_type=expected_model_type,
        expected_regime=expected_regime,
    )
    return cfg


def validate_experiment_config(
    cfg: Dict[str, Any],
    *,
    config_path: str | Path | None = None,
    expected_model_type: Optional[str] = None,
    expected_regime: Optional[str] = None,
) -> ExperimentSpec:
    missing = [
        field
        for field in ("experiment_name", "model_name", "model_type", "dataset", "dataset_language", "output_dir")
        if field not in cfg
    ]
    if missing:
        raise ValueError(f"Config is missing required field(s): {', '.join(missing)}")

    spec = (
        experiment_by_config_path(config_path)
        if config_path is not None
        else _experiment_by_name(str(cfg["experiment_name"]))
    )

    if cfg["experiment_name"] != spec.experiment_name:
        raise ValueError(f"Config experiment_name must be {spec.experiment_name!r}")
    if cfg["model_name"] != spec.model_name:
        raise ValueError(f"{spec.experiment_name}: model_name must be {spec.model_name!r}")
    if cfg["model_type"] != spec.model_type:
        raise ValueError(f"{spec.experiment_name}: model_type must be {spec.model_type!r}")
    if cfg["dataset"] != DATASET_NAME:
        raise ValueError(f"{spec.experiment_name}: dataset must be {DATASET_NAME!r}")
    if cfg["dataset_language"] != DATASET_LANGUAGE:
        raise ValueError(f"{spec.experiment_name}: dataset_language must be {DATASET_LANGUAGE!r}")
    if Path(str(cfg["output_dir"])).as_posix() != spec.output_dir:
        raise ValueError(f"{spec.experiment_name}: output_dir must be {spec.output_dir!r}")

    mode = str(cfg.get("mode", "")).lower()
    if spec.regime == "encoder":
        if mode:
            raise ValueError(f"{spec.experiment_name}: encoder configs must not set mode")
    elif spec.regime == "llm_lora":
        if mode != "lora":
            raise ValueError(f"{spec.experiment_name}: decoder LoRA configs must set mode: lora")
    elif spec.regime == "llm_zeroshot":
        if mode != "zeroshot":
            raise ValueError(f"{spec.experiment_name}: zero-shot configs must set mode: zeroshot")

    if expected_model_type and spec.model_type != expected_model_type:
        raise ValueError(f"{spec.experiment_name}: expected model_type {expected_model_type!r}")
    if expected_regime and spec.regime != expected_regime:
        raise ValueError(f"{spec.experiment_name}: expected regime {expected_regime!r}")

    return spec


def output_dir_from_config(cfg: Dict[str, Any]) -> Path:
    validate_experiment_config(cfg)
    configured_output = Path(str(cfg["output_dir"]))
    results_root = os.environ.get("BA_NER_RESULTS_ROOT")
    if not results_root:
        return configured_output
    return Path(results_root) / configured_output.relative_to(RESULTS_ROOT)


def _experiment_by_name(experiment_name: str) -> ExperimentSpec:
    for spec in ALL_EXPERIMENTS.values():
        if spec.experiment_name == experiment_name:
            return spec
    raise ValueError(f"Unsupported experiment name: {experiment_name}")


def _filter_specs(
    specs: Iterable[ExperimentSpec],
    *,
    model_type: Optional[str],
    regime: Optional[str],
) -> list[ExperimentSpec]:
    filtered = list(specs)
    if model_type is not None:
        filtered = [spec for spec in filtered if spec.model_type == model_type]
    if regime is not None:
        filtered = [spec for spec in filtered if spec.regime == regime]
    return filtered
