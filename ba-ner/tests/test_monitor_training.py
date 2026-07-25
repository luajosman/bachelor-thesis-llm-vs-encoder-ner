from __future__ import annotations

from datetime import datetime

import pytest

from src.evaluate.monitor_training import (
    JobState,
    ModelSnapshot,
    ModelSpec,
    Progress,
    choose_job,
    estimate_remaining,
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


def test_choose_job_prefers_running_resume() -> None:
    spec = _decoder_spec()
    jobs = {
        10: JobState(10, "FAILED", "01:00", "-", "1:0"),
        11: JobState(11, "RUNNING", "00:05", "node1"),
    }

    assert choose_job(spec, jobs).job_id == 11


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
    )

    assert "**Refresh:** every 5 minutes" in rendered
    assert "## Live Estimate" in rendered
    assert "## Live Training Metrics" in rendered
    assert "## Validation Results" in rendered
    assert "## Recovery Checkpoints" in rendered
    assert "Qwen test | RUNNING" in rendered
    assert "**All models** | Parallel" in rendered
    assert "90.00%" in rendered
    assert "Final summary job: `99` (PENDING)" in rendered


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


def test_write_atomic_makes_dashboard_browser_readable(tmp_path) -> None:
    output = tmp_path / "training_monitor.html"

    write_atomic(output, "<p>live</p>")

    assert output.read_text(encoding="utf-8") == "<p>live</p>"
    assert output.stat().st_mode & 0o777 == 0o644
