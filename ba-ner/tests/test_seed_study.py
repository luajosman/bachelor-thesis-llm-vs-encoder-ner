from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config import load_experiment_config
from src.seed_study import (
    ConfigMismatchError,
    OutputCollisionError,
    ResumeValidationError,
    RunLockedError,
    full_run_config_hash,
    managed_run_phase,
    recursive_config_diff,
    scientific_config_hash,
    validate_manifest,
    validate_resume,
    validate_seed_equivalence,
)
from src.seed_provenance import (
    DATASET_REVISION,
    MODEL_REVISIONS,
    reference_provenance_issues,
    scientific_contract,
)


def test_manifest_contains_exactly_fifteen_new_runs_and_valid_groups() -> None:
    report = validate_manifest()

    assert report["valid"] is True
    assert report["new_run_count"] == 15
    assert report["historical_output_count"] == 5
    assert set(report["groups"]) == {
        "deberta-v3-base",
        "deberta-v3-large",
        "qwen35-08b-qlora",
        "qwen35-4b-qlora",
        "qwen35-27b-qlora-3ep",
    }
    for group in report["groups"].values():
        assert len(set(group["scientific_hashes"].values())) == 1
        assert len(set(group["full_hashes"].values())) == 3


def test_recursive_diff_detects_nested_list_missing_extra_and_type_changes() -> None:
    reference = {
        "nested": {"learning_rate": 0.1, "items": [1, 2]},
        "missing": True,
    }
    candidate = {
        "nested": {"learning_rate": "0.1", "items": [1, 3, 4]},
        "extra": None,
    }

    differences = recursive_config_diff(reference, candidate)

    assert {(difference.path, difference.kind) for difference in differences} == {
        ("missing", "missing"),
        ("extra", "additional"),
        ("nested.learning_rate", "type_mismatch"),
        ("nested.items", "list_length"),
        ("nested.items[1]", "value"),
    }


def test_only_seed_and_operational_identity_differences_are_allowed() -> None:
    reference = {"seed": 42, "experiment_name": "a", "output_dir": "x", "lr": 0.1}
    candidate = {"seed": 123, "experiment_name": "b", "output_dir": "y", "lr": 0.1}

    differences = validate_seed_equivalence(reference, candidate)

    assert {difference.path for difference in differences} == {
        "seed", "experiment_name", "output_dir"
    }


def test_unapproved_scientific_difference_blocks_run() -> None:
    with pytest.raises(ConfigMismatchError, match="learning_rate"):
        validate_seed_equivalence(
            {"seed": 42, "training": {"learning_rate": 1e-4}},
            {"seed": 123, "training": {"learning_rate": 2e-4}},
        )


def test_hashes_are_deterministic_and_ignore_key_order() -> None:
    first = {"b": [2, 1], "a": 1, "seed": 42, "output_dir": "one"}
    second = {"output_dir": "one", "seed": 42, "a": 1, "b": [2, 1]}
    different_run = {**first, "seed": 123, "output_dir": "two"}

    assert scientific_config_hash(first) == scientific_config_hash(second)
    assert full_run_config_hash(first) == full_run_config_hash(second)
    assert scientific_config_hash(first) == scientific_config_hash(different_run)
    assert full_run_config_hash(first) != full_run_config_hash(different_run)


def test_scientific_contract_pins_revisions_prompt_parser_and_code() -> None:
    config = load_experiment_config("configs/qwen35_4b.yaml")

    contract = scientific_contract(config)

    assert contract["dataset_revision"] == DATASET_REVISION
    assert contract["model_revision"] == MODEL_REVISIONS[config["model_name"]]
    assert contract["prompt"]["thinking_enabled"] is False
    assert contract["prompt"]["do_sample"] is False
    assert len(contract["prompt"]["prompt_sha256"]) == 64
    assert len(contract["prompt"]["parser_sha256"]) == 64
    assert "src/decoder/train.py" in contract["code_files"]


def test_manifest_materializes_and_validates_both_hashes_per_canonical_run() -> None:
    manifest = yaml.safe_load(Path("configs/seed_study_manifest.yaml").read_text())

    for group in manifest["groups"]:
        for run in group["canonical_runs"]:
            assert len(run["scientific_config_hash"]) == 64
            assert len(run["full_run_config_hash"]) == 64


def test_all_fresh_canonical_groups_have_verified_provenance() -> None:
    manifest = yaml.safe_load(Path("configs/seed_study_manifest.yaml").read_text())

    issues = reference_provenance_issues(manifest)

    assert issues == []


@pytest.mark.parametrize(
    ("historical_path", "canonical_path"),
    [
        ("configs/deberta_base.yaml", "configs/deberta_base_canonical.yaml"),
        ("configs/deberta_large.yaml", "configs/deberta_large_canonical.yaml"),
        ("configs/qwen35_08b.yaml", "configs/qwen35_08b_canonical.yaml"),
        ("configs/qwen35_4b.yaml", "configs/qwen35_4b_canonical.yaml"),
    ],
)
def test_fresh_seed42_configs_are_scientifically_equivalent(
    historical_path: str,
    canonical_path: str,
) -> None:
    historical = load_experiment_config(historical_path)
    canonical = load_experiment_config(canonical_path)

    differences = validate_seed_equivalence(historical, canonical)

    assert historical["output_dir"] != canonical["output_dir"]
    assert {difference.path for difference in differences} == {
        "experiment_name", "output_dir"
    }
    assert "resume_from_checkpoint" not in canonical


def test_qwen27b_three_epoch_config_is_fresh_and_only_changes_epochs_and_identity() -> None:
    original = load_experiment_config("configs/qwen35_27b.yaml")
    canonical = load_experiment_config("configs/qwen35_27b_3ep.yaml")

    differences = validate_seed_equivalence(
        original,
        canonical,
        additionally_allowed_fields={"num_train_epochs"},
    )

    assert original["num_train_epochs"] == 2
    assert canonical["num_train_epochs"] == 3
    assert {difference.path for difference in differences} == {
        "experiment_name", "num_train_epochs", "output_dir"
    }
    assert "resume_from_checkpoint" not in canonical


def test_managed_run_writes_atomic_provenance_and_prevents_duplicate_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setenv("BA_NER_RESULTS_ROOT", str(results_root))
    config = "configs/deberta_base_seed123.yaml"

    with managed_run_phase(config, "training") as context:
        assert context is not None
        assert (context.output_dir / "config_source.yaml").is_file()
        assert (context.output_dir / "config_resolved.yaml").is_file()
        assert (context.output_dir / "run_metadata.yaml").is_file()
        assert (context.output_dir / ".run.lock").is_file()
        with pytest.raises(RunLockedError):
            with managed_run_phase(config, "training"):
                pass

    status = json.loads((context.output_dir / "status.json").read_text())
    assert status["status"] == "TRAINING_COMPLETED"
    assert not (context.output_dir / ".run.lock").exists()


def test_same_run_resume_requires_explicit_flag_and_matching_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BA_NER_RESULTS_ROOT", str(tmp_path / "results"))
    config = "configs/qwen35_08b_seed123.yaml"
    with managed_run_phase(config, "training") as context:
        assert context is not None
    checkpoint = context.output_dir / "checkpoint-500"
    checkpoint.mkdir()

    with pytest.raises(ResumeValidationError, match="explicit"):
        with managed_run_phase(config, "training"):
            pass

    monkeypatch.setenv("BA_NER_ALLOW_RESUME", "1")
    with managed_run_phase(config, "training") as resumed:
        assert resumed is not None
        assert resumed.resumed is True
        assert resumed.resume_checkpoint == str(checkpoint)


def test_cross_seed_resume_is_blocked(tmp_path: Path) -> None:
    cfg_123 = load_experiment_config("configs/qwen35_4b_seed123.yaml")
    cfg_456 = load_experiment_config("configs/qwen35_4b_seed456.yaml")
    output = tmp_path / "qwen-seed123"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    metadata = {
        "model_name": cfg_123["model_name"],
        "seed": cfg_123["seed"],
        "num_train_epochs": cfg_123["num_train_epochs"],
        "scientific_config_hash": scientific_config_hash(cfg_123),
        "full_run_config_hash": full_run_config_hash(cfg_123),
        "resolved_output_dir": str(output),
    }

    with pytest.raises(ResumeValidationError, match="seed"):
        validate_resume(
            expected_config=cfg_456,
            metadata=metadata,
            output_dir=output,
            checkpoint=checkpoint,
        )


def test_two_epoch_to_three_epoch_cross_variant_resume_is_blocked(tmp_path: Path) -> None:
    historical = load_experiment_config("configs/qwen35_27b.yaml")
    canonical = load_experiment_config("configs/qwen35_27b_3ep.yaml")
    output = tmp_path / "qwen27-3ep"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    metadata = {
        "model_name": historical["model_name"],
        "seed": historical["seed"],
        "num_train_epochs": historical["num_train_epochs"],
        "scientific_config_hash": scientific_config_hash(historical),
        "full_run_config_hash": full_run_config_hash(historical),
        "resolved_output_dir": str(output),
    }

    with pytest.raises(ResumeValidationError, match="num_train_epochs"):
        validate_resume(
            expected_config=canonical,
            metadata=metadata,
            output_dir=output,
            checkpoint=checkpoint,
        )


def test_existing_reference_and_historical_runs_are_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BA_NER_RESULTS_ROOT", str(tmp_path / "results"))

    with pytest.raises(OutputCollisionError, match="read-only"):
        with managed_run_phase("configs/deberta_base.yaml", "training"):
            pass
    with pytest.raises(OutputCollisionError, match="read-only"):
        with managed_run_phase("configs/deberta_large.yaml", "training"):
            pass
    with pytest.raises(OutputCollisionError, match="read-only"):
        with managed_run_phase("configs/qwen35_27b.yaml", "training"):
            pass
    with pytest.raises(OutputCollisionError, match="read-only"):
        with managed_run_phase("configs/qwen35_08b.yaml", "training"):
            pass
    with pytest.raises(OutputCollisionError, match="read-only"):
        with managed_run_phase("configs/qwen35_4b.yaml", "training"):
            pass
