from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.run_seed_matrix as seed_matrix

from scripts.run_seed_matrix import (
    build_plan,
    command_preview,
    render_dry_run,
    select_new_runs,
)


def test_complete_matrix_contains_exactly_fifteen_new_non_zeroshot_runs() -> None:
    runs = select_new_runs()

    assert len(runs) == 15
    assert all(run.regime in {"encoder", "llm_lora"} for run in runs)
    assert all("zeroshot" not in run.group_key for run in runs)


def test_family_seed_and_model_filters_select_expected_runs() -> None:
    assert len(select_new_runs(encoder_only=True)) == 6
    assert len(select_new_runs(decoder_only=True)) == 9
    assert len(select_new_runs(seeds=[123])) == 5
    assert len(select_new_runs(seeds=[456])) == 5
    assert len(select_new_runs(seeds=[42])) == 5
    qwen08 = select_new_runs(models=["qwen35_08b"])
    assert {(run.seed, run.status) for run in qwen08} == {
        (42, "planned_canonical"),
        (123, "planned_canonical"),
        (456, "planned_canonical"),
    }
    qwen27 = select_new_runs(models=["qwen35_27b_3ep"])
    assert {(run.seed, run.variant) for run in qwen27} == {
        (42, "3ep"), (123, "3ep"), (456, "3ep")
    }


def test_exact_config_filter_does_not_expand_to_model_seed_cross_product() -> None:
    runs = select_new_runs(config_paths=[
        "configs/qwen35_4b_seed123.yaml",
        "configs/qwen35_27b_3ep.yaml",
    ])

    assert [run.config_path for run in runs] == [
        "configs/qwen35_4b_seed123.yaml",
        "configs/qwen35_27b_3ep.yaml",
    ]


def test_compute_preflight_receives_exact_selected_run_configs() -> None:
    runs = select_new_runs()

    command = seed_matrix._compute_preflight_command(runs)
    selected = {
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--config"
    }

    assert selected == {run.config_path for run in runs}
    assert len(selected) == 15
    script_index = command.index("scripts/cluster/preflight.sh")
    assert command[script_index + 1:] == [
        value
        for run in runs
        for value in ("--config", run.config_path)
    ]


def test_slurm_commands_have_unique_seed_variant_phase_names_and_logs() -> None:
    run = select_new_runs(models=["qwen35_27b_3ep"], seeds=[42])[0]

    train = command_preview(run, "training")
    infer = command_preview(run, "inference", dependency="123")
    evaluate = command_preview(run, "evaluation", dependency="456")

    assert "--job-name=ner-qwen27b-3ep-s42-train" in train
    assert any("seed-42/training-%j.out" in value for value in train)
    assert "--dependency=afterok:123" in infer
    assert "--dependency=afterok:456" in evaluate
    assert train[-1] == "configs/qwen35_27b_3ep.yaml"
    assert infer[-1].endswith("qwen35-27b-qlora-3ep/best_lora_adapter")


def test_full_dry_run_reports_required_zero_conflict_summary() -> None:
    rendered = render_dry_run(build_plan(select_new_runs()))

    assert "Planned new training runs: 15" in rendered
    assert "New canonical DeBERTa-v3-base runs: 3" in rendered
    assert "New canonical DeBERTa-v3-large runs: 3" in rendered
    assert "New canonical Qwen3.5-0.8B runs: 3" in rendered
    assert "New canonical Qwen3.5-4B runs: 3" in rendered
    assert "Historical Qwen3.5-0.8B runs preserved: 1" in rendered
    assert "New canonical Qwen3.5-27B 3ep runs: 3" in rendered
    assert "Historical Qwen3.5-27B 2ep runs preserved: 1" in rendered
    assert "Historical DeBERTa runs preserved: 2" in rendered
    assert "Duplicated zero-shot runs: 0" in rendered
    assert "Scientific config mismatches within canonical seed groups: 0" in rendered
    assert "num_train_epochs: 2 -> 3" in rendered


def test_submission_freeze_rejects_a_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        seed_matrix,
        "_git_repository_state",
        lambda: ("abc123", " M src/decoder/train.py"),
    )

    with pytest.raises(seed_matrix.SeedStudyError, match="clean committed worktree"):
        seed_matrix.require_submission_repository_freeze()


def test_submission_freeze_exports_the_exact_clean_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BA_NER_EXPECTED_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        seed_matrix,
        "_git_repository_state",
        lambda: ("abc123", ""),
    )

    assert seed_matrix.require_submission_repository_freeze() == "abc123"
    assert "BA_NER_EXPECTED_GIT_COMMIT=abc123" in seed_matrix._base_exports()
    run = select_new_runs(models=["deberta_base"], seeds=[42])[0]
    assert any(
        "BA_NER_EXPECTED_GIT_COMMIT=abc123" in value
        for value in command_preview(run, "training")
    )


def test_partial_submission_failure_is_persisted_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = select_new_runs(models=["qwen35_27b_3ep"], seeds=[42])[0]
    calls = 0

    def fake_submit(command):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise seed_matrix.SeedStudyError("synthetic submission failure")
        return "9001"

    monkeypatch.setattr(seed_matrix, "_submit", fake_submit)
    monkeypatch.setattr(seed_matrix, "resolve_results_dir", lambda: tmp_path / "results")
    monkeypatch.setattr(seed_matrix, "_log_dir", lambda run: tmp_path / "logs" / str(run.seed))
    monkeypatch.setattr(seed_matrix, "inspect_output_path", lambda path: "FREE")

    with pytest.raises(seed_matrix.SeedStudyError, match="synthetic"):
        seed_matrix.submit_pipelines(
            [run],
            phases=("training", "inference", "evaluation"),
            skip_cluster_preflight=True,
        )

    registry_path = tmp_path / "results/seed_studies/multinerd/submission_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert registry["submission_status"] == "PARTIAL_SUBMISSION_FAILED"
    assert registry["jobs"][0]["job_id"] == "9001"


def test_aggregation_submission_uses_afterany_for_partial_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = select_new_runs(models=["qwen35_27b_3ep"], seeds=[42])[0]
    commands = []

    def fake_submit(command):
        commands.append(command)
        return str(9100 + len(commands))

    monkeypatch.setattr(seed_matrix, "_submit", fake_submit)
    monkeypatch.setattr(seed_matrix, "resolve_results_dir", lambda: tmp_path / "results")
    monkeypatch.setattr(seed_matrix, "_log_dir", lambda run: tmp_path / "logs" / str(run.seed))
    monkeypatch.setattr(seed_matrix, "inspect_output_path", lambda path: "FREE")

    registry = seed_matrix.submit_pipelines(
        [run],
        phases=("training", "inference", "evaluation"),
        skip_cluster_preflight=True,
    )

    aggregate = registry["jobs"][-1]
    assert aggregate["phase"] == "aggregation"
    assert aggregate["dependency"].startswith("afterany:")
    assert any(value.startswith("--dependency=afterany:") for value in commands[-1])
