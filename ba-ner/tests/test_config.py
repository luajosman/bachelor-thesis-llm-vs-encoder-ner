from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    DATASET_LANGUAGE,
    DATASET_NAME,
    FINAL_EXPERIMENTS,
    final_config_paths,
    load_experiment_config,
    output_dir_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_registry_contains_exactly_eight_experiments() -> None:
    assert len(FINAL_EXPERIMENTS) == 8
    assert final_config_paths(model_type="encoder") == [
        "configs/deberta_base.yaml",
        "configs/deberta_large.yaml",
    ]
    assert len(final_config_paths(regime="llm_zeroshot")) == 3
    assert len(final_config_paths(regime="llm_lora")) == 3


def test_all_final_configs_validate_against_registry() -> None:
    seen_output_dirs = set()
    for spec in FINAL_EXPERIMENTS.values():
        cfg = load_experiment_config(PROJECT_ROOT / spec.config_path)
        assert cfg["dataset"] == DATASET_NAME
        assert cfg["dataset_language"] == DATASET_LANGUAGE
        assert cfg["experiment_name"] == spec.experiment_name
        assert cfg["model_name"] == spec.model_name
        assert str(output_dir_from_config(cfg)) == spec.output_dir
        seen_output_dirs.add(cfg["output_dir"])

    assert len(seen_output_dirs) == 8


def test_decoder_training_rejects_zero_shot_config() -> None:
    with pytest.raises(ValueError, match="expected regime"):
        load_experiment_config(
            PROJECT_ROOT / "configs/qwen35_4b_zeroshot.yaml",
            expected_model_type="decoder",
            expected_regime="llm_lora",
        )
