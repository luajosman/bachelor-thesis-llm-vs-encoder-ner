from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.evaluate.monitor_training import (
    _encoder_validation_metrics,
    _inference_time_cells,
    collect_final_results,
    FinalResultSnapshot,
    JobState,
    MonitorConfig,
    ModelSnapshot,
    ModelSpec,
    Progress,
    choose_job,
    collect_snapshot,
    estimate_remaining,
    filter_specs,
    load_config,
    parse_inference_progress,
    parse_progress,
    parse_train_metrics,
    render_html,
    render_markdown,
    write_atomic,
)


def _decoder_spec() -> ModelSpec:
    return ModelSpec(
        key="qwen-test",
        label="Qwen test",
        kind="decoder",
        job_ids=(10, 11),
        total_steps=300,
        epochs=3,
        eval_seconds_low=60,
        eval_seconds_high=120,
        restart_buffer_seconds=30,
    )


def test_parse_progress_uses_latest_tqdm_sample() -> None:
    text = (
        "\r 10%|x| 100/1000 [00:10<01:30, 2.00it/s]"
        "\r 25%|x| 250/1000 [01:00<03:00, 1.50s/it]"
    )

    progress = parse_progress(text)

    assert progress == Progress(step=250, total=1000, seconds_per_step=1.0)
    assert progress.percent == pytest.approx(25.0)


def test_parse_train_metrics_skips_unrelated_dicts() -> None:
    text = "\n".join([
        "{'eval_loss': 0.4}",
        "not a dict",
        "{'loss': '0.25', 'mean_token_accuracy': 0.95, 'step': 25}",
    ])

    assert parse_train_metrics(text) == {
        "loss": "0.25",
        "mean_token_accuracy": 0.95,
        "step": 25,
    }


def test_parse_inference_progress_uses_latest_sample() -> None:
    text = "\n".join([
        "INFERENCE_PROGRESS 100/1000 elapsed=20.000s",
        "INFERENCE_PROGRESS 250/1000 elapsed=45.500s",
    ])

    progress = parse_inference_progress(text)

    assert progress is not None
    assert progress.completed == 250
    assert progress.total == 1000
    assert progress.percent == pytest.approx(25.0)
    assert progress.remaining_seconds == pytest.approx(136.5)


def test_pending_inference_distinguishes_unknown_eta_from_time_limit() -> None:
    result = FinalResultSnapshot(
        experiment_name="qwen-test",
        regime="llm_zeroshot",
        status="Inference queued",
        metrics={},
        job=JobState(123, "PENDING", "00:00"),
        time_limit_seconds=3 * 86400,
    )

    assert _inference_time_cells(
        result,
        datetime.now().astimezone(),
    ) == (
        "0s",
        "not started",
        "unknown (limit: 3d 00h)",
        "unknown until start",
        "waiting for scheduler",
    )


def test_choose_job_prefers_running_resume() -> None:
    spec = _decoder_spec()
    jobs = {
        10: JobState(10, "FAILED", "01:00", "-", "1:0"),
        11: JobState(11, "RUNNING", "00:05", "node1"),
    }

    assert choose_job(spec, jobs).job_id == 11


def test_choose_job_renders_new_run_without_placeholder_id_as_planned() -> None:
    spec = ModelSpec(
        key="qwen-new",
        label="Qwen new",
        kind="decoder",
        job_ids=(),
        variant="3ep",
        seed=123,
    )

    assert choose_job(spec, {}) == JobState(0, "PLANNED")


def test_monitor_filters_canonical_historical_variant_and_seed() -> None:
    specs = (
        ModelSpec("qwen-2ep", "Qwen 2ep", "decoder", (1,), variant="2ep", seed=42, canonical=False, historical=True),
        ModelSpec("qwen-3ep", "Qwen 3ep", "decoder", (), variant="3ep", seed=42),
        ModelSpec("qwen-3ep-seed123", "Qwen 3ep", "decoder", (), variant="3ep", seed=123),
    )

    assert [spec.key for spec in filter_specs(specs, historical=True)] == ["qwen-2ep"]
    assert [spec.key for spec in filter_specs(specs, canonical=True, seed=123)] == [
        "qwen-3ep-seed123"
    ]
    assert len(filter_specs(specs, variant="3ep")) == 2


def test_estimate_remaining_includes_pending_evals_and_restart_buffer() -> None:
    snapshot = ModelSnapshot(
        spec=_decoder_spec(),
        job=JobState(11, "RUNNING"),
        progress=Progress(step=100, total=300, seconds_per_step=2.0),
        train_metrics={},
        dev_metrics={"epoch_results": [{"epoch": 1}]},
        checkpoint_step=100,
        checkpoint_time=None,
        results={},
        alert=None,
    )

    assert estimate_remaining(snapshot) == (520.0, 670.0)


def test_encoder_validation_metrics_recovers_best_checkpoint_row(tmp_path) -> None:
    state_path = tmp_path / "checkpoint-200" / "trainer_state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps({
            "log_history": [
                {
                    "epoch": 1.0,
                    "eval_precision": 0.85,
                    "eval_recall": 0.90,
                    "eval_f1": 0.874,
                },
                {
                    "epoch": 2.0,
                    "eval_precision": 0.92,
                    "eval_recall": 0.94,
                    "eval_f1": 0.9299,
                },
            ],
        }),
        encoding="utf-8",
    )

    metrics = _encoder_validation_metrics(tmp_path, best_validation_f1=0.93)

    assert metrics == {
        "best_f1": 0.9299,
        "best_precision": 0.92,
        "best_recall": 0.94,
        "best_epoch": 2.0,
    }


def test_render_markdown_contains_requested_live_sections() -> None:
    snapshot = ModelSnapshot(
        spec=_decoder_spec(),
        job=JobState(11, "RUNNING", "00:05", "node1"),
        progress=Progress(step=100, total=300, seconds_per_step=2.0),
        train_metrics={
            "loss": 0.25,
            "mean_token_accuracy": 0.95,
            "learning_rate": 0.0002,
            "grad_norm": 0.4,
        },
        dev_metrics={
            "best_f1": 0.9,
            "best_epoch": 1,
            "epoch_results": [{
                "epoch": 1,
                "dev_precision": 0.88,
                "dev_recall": 0.92,
                "parse_failure_rate": 0.0,
            }],
        },
        checkpoint_step=100,
        checkpoint_time=None,
        results={},
        alert=None,
        eta_low_seconds=520.0,
        eta_high_seconds=670.0,
    )

    rendered = render_markdown(
        [snapshot],
        datetime.now().astimezone(),
        refresh_seconds=300,
        summary_job=JobState(99, "PENDING"),
        final_results=[
            FinalResultSnapshot(
                experiment_name="qwen-test",
                regime="llm_zeroshot",
                status="Training",
                metrics={},
            ),
        ],
    )

    assert "**Data refresh:** every 5 minutes" in rendered
    assert "**Page refresh:** every 15 seconds" in rendered
    assert "## Live Estimate" in rendered
    assert "## Live Training Metrics" in rendered
    assert "## Validation Results" in rendered
    assert "## Canonical Seed Study Runs" in rendered
    assert "## All Final Experiments" in rendered
    assert "## Inference Time Estimates" in rendered
    assert "## Recovery Checkpoints" in rendered
    assert "Qwen test | RUNNING" in rendered
    assert "**All models** | Parallel" in rendered
    assert "90.00%" in rendered
    assert "Final comparison job: `99` (PENDING)" in rendered
    assert "`results/multinerd/qwen-test`" in rendered
    assert "| qwen-test | LLM zero-shot | Training | - | N/A |" in rendered


def test_completed_decoder_without_log_progress_renders_complete() -> None:
    snapshot = ModelSnapshot(
        spec=_decoder_spec(),
        job=JobState(11, "COMPLETED", "00:00:01"),
        progress=None,
        train_metrics={},
        dev_metrics={"best_f1": 0.9, "best_epoch": 2},
        checkpoint_step=300,
        checkpoint_time=None,
        results={"best_dev_f1": 0.9, "train_runtime_seconds": 600.0},
        alert=None,
    )
    snapshot.eta_low_seconds, snapshot.eta_high_seconds = estimate_remaining(snapshot)

    rendered = render_markdown(
        [snapshot],
        datetime.now().astimezone(),
        refresh_seconds=300,
        summary_job=None,
    )

    assert (
        "| Qwen test | COMPLETED | 100.0% (300/300) | 3.00/3 | "
        "complete | complete |"
    ) in rendered
    assert "| Qwen test | 11 | - | 00:10:00 | 2.00 s |" in rendered


def test_collect_final_results_includes_all_experiments_and_metrics(tmp_path) -> None:
    result_dir = tmp_path / "deberta-v3-base"
    result_dir.mkdir()
    (result_dir / "results.yaml").write_text(
        "best_validation_f1: 0.9\n",
        encoding="utf-8",
    )
    (result_dir / "inference_metrics.yaml").write_text(
        "\n".join([
            "test_f1: 0.88",
            "test_precision: 0.87",
            "test_recall: 0.89",
        ]),
        encoding="utf-8",
    )

    results = collect_final_results(tmp_path)

    assert len(results) == 8
    assert results[0].experiment_name == "deberta-v3-base"
    assert results[0].status == "Complete"
    assert results[0].metrics["test_f1"] == 0.88
    assert {result.status for result in results[1:]} == {"Waiting"}


def test_collect_final_results_shows_inference_job_state(tmp_path) -> None:
    results = collect_final_results(
        tmp_path,
        {"qwen35-08b-zeroshot": (123,)},
        jobs={123: JobState(123, "RUNNING", "00:01", "node1")},
    )

    qwen = next(
        result
        for result in results
        if result.experiment_name == "qwen35-08b-zeroshot"
    )
    assert qwen.status == "Inference running"
    assert qwen.job == JobState(123, "RUNNING", "00:01", "node1")


def test_render_markdown_includes_encoder_validation_details() -> None:
    spec = ModelSpec(
        key="deberta-test",
        label="DeBERTa test",
        kind="encoder",
        job_ids=(20,),
    )
    snapshot = ModelSnapshot(
        spec=spec,
        job=JobState(20, "COMPLETED"),
        progress=None,
        train_metrics={},
        dev_metrics={
            "best_f1": 0.9263,
            "best_precision": 0.9199,
            "best_recall": 0.9328,
            "best_epoch": 5.0,
        },
        checkpoint_step=100,
        checkpoint_time=None,
        results={"best_validation_f1": 0.9263, "num_train_epochs": 5},
        alert=None,
        eta_low_seconds=0.0,
        eta_high_seconds=0.0,
    )

    rendered = render_markdown(
        [snapshot],
        datetime.now().astimezone(),
        refresh_seconds=300,
        summary_job=None,
    )

    assert "| DeBERTa test | 92.63% | 91.99% | 93.28% | 5 | N/A |" in rendered


def test_render_html_auto_refreshes_and_renders_dashboard_tables() -> None:
    markdown = "\n".join([
        "# Training Live Monitor",
        "",
        "**Updated:** 2026-07-25 19:00:00 CEST",
        "",
        "## Live Estimate",
        "",
        "| Model | Status |",
        "|---|---|",
        "| Qwen <test> | **RUNNING** |",
    ])

    rendered = render_html(markdown, browser_refresh_seconds=30)

    assert '<meta http-equiv="refresh" content="30">' in rendered
    assert 'id="reload-countdown">30</span>' in rendered
    assert "<h1>Training Live Monitor</h1>" in rendered
    assert "<table>" in rendered
    assert "<strong>RUNNING</strong>" in rendered
    assert "Qwen &lt;test&gt;" in rendered


def test_load_config_supports_separate_refresh_intervals(tmp_path) -> None:
    config = tmp_path / "monitor.yaml"
    config.write_text(
        "\n".join([
            "refresh_seconds: 30",
            "scheduler_refresh_seconds: 60",
            "browser_refresh_seconds: 15",
            "models:",
            "  - key: test",
            "    label: Test",
            "    kind: decoder",
            "    job_ids: [1]",
        ]),
        encoding="utf-8",
    )

    settings = load_config(config)

    assert settings == MonitorConfig(
        specs=(ModelSpec(
            key="test",
            label="Test",
            kind="decoder",
            job_ids=(1,),
        ),),
        refresh_seconds=30,
        scheduler_refresh_seconds=60,
        browser_refresh_seconds=15,
        summary_job_id=None,
        result_job_ids={},
        result_time_limits_seconds={},
    )


def test_repository_monitor_tracks_all_fresh_and_historical_groups() -> None:
    settings = load_config(Path("configs/training_monitor.yaml"))
    by_key = {spec.key: spec for spec in settings.specs}

    assert len(by_key) == len(settings.specs)
    historical_keys = {
        "deberta-v3-base",
        "deberta-v3-large",
        "qwen35-08b-qlora",
        "qwen35-4b-qlora",
        "qwen35-27b-qlora",
    }
    assert all(
        by_key[key].historical and not by_key[key].canonical
        for key in historical_keys
    )

    canonical_groups = (
        "deberta-v3-base-canonical",
        "deberta-v3-large-canonical",
        "qwen35-08b-qlora-canonical",
        "qwen35-4b-qlora-canonical",
        "qwen35-27b-qlora-3ep",
    )
    for base in canonical_groups:
        keys = (base, f"{base}-seed123", f"{base}-seed456")
        assert {by_key[key].seed for key in keys} == {42, 123, 456}
        assert all(
            by_key[key].canonical and not by_key[key].historical
            for key in keys
        )


def test_write_atomic_makes_dashboard_browser_readable(tmp_path) -> None:
    output = tmp_path / "training_monitor.html"

    write_atomic(output, "<p>live</p>")

    assert output.read_text(encoding="utf-8") == "<p>live</p>"
    assert output.stat().st_mode & 0o777 == 0o644


def test_corrupt_status_file_does_not_break_snapshot_collection(tmp_path) -> None:
    result_dir = tmp_path / "qwen-test"
    result_dir.mkdir()
    (result_dir / "status.json").write_text("{broken", encoding="utf-8")

    snapshot = collect_snapshot(_decoder_spec(), {}, tmp_path)

    assert snapshot.status_data == {}
    assert snapshot.job.state == "UNKNOWN"


def test_seed_table_renders_extended_runtime_fields_and_stale_state() -> None:
    snapshot = ModelSnapshot(
        spec=_decoder_spec(),
        job=JobState(11, "RUNNING", "00:05", "node1"),
        progress=Progress(step=100, total=300, seconds_per_step=2.0),
        train_metrics={},
        dev_metrics={},
        checkpoint_step=100,
        checkpoint_time=None,
        results={},
        alert=None,
        status_data={
            "status": "RUNNING",
            "phase": "training",
            "slurm_array_id": "77",
            "node": "node1",
            "start_time": "2026-08-07T10:00:00+00:00",
            "last_update": "2020-01-01T00:00:00+00:00",
            "elapsed_seconds": 300,
            "eta_seconds": 400,
            "progress_percent": 33.3,
            "step": 100,
            "max_steps": 300,
            "best_step": 90,
            "best_epoch": 1,
            "peak_vram_bytes": 1024 * 1024,
        },
        run_metadata={"runtime": {"cuda": {"devices": ["H100"]}}},
    )

    rendered = render_markdown(
        [snapshot],
        datetime.now().astimezone(),
        refresh_seconds=30,
        summary_job=None,
    )

    assert "| Job | Array | Node | GPU | Started | Elapsed |" in rendered
    assert "STALE (RUNNING)" in rendered
    assert "| 11 | 77 | node1 | H100 |" in rendered
    assert "100/300 | 33.3%" in rendered
    assert "1.0 MiB" in rendered
