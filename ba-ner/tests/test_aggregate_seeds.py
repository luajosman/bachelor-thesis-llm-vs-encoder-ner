from __future__ import annotations

import statistics
from pathlib import Path

import pytest
import yaml

from src.evaluate.aggregate_seeds import aggregate_seed_study


GROUP_OUTPUTS = {
    "deberta-v3-base": [
        "deberta-v3-base-canonical",
        "deberta-v3-base-canonical-seed123",
        "deberta-v3-base-canonical-seed456",
    ],
    "qwen35-27b-qlora-3ep": [
        "qwen35-27b-qlora-3ep",
        "qwen35-27b-qlora-3ep-seed123",
        "qwen35-27b-qlora-3ep-seed456",
    ],
}


def _write_run(results_root: Path, experiment: str, value: float) -> None:
    output = results_root / "multinerd" / experiment
    output.mkdir(parents=True)
    (output / "results.yaml").write_text(
        yaml.safe_dump({
            "train_runtime_seconds": 100.0 + value,
            "trainable_params": 10,
        }),
        encoding="utf-8",
    )
    (output / "inference_metrics.yaml").write_text(
        yaml.safe_dump({
            "test_precision": value,
            "test_recall": value,
            "test_f1": value,
            "latency_ms_mean": 10.0,
            "latency_ms_p95": 12.0,
            "vram_peak_mb": 1000.0,
            "total_params": 100,
        }),
        encoding="utf-8",
    )


def test_aggregation_uses_mean_sample_std_min_max_and_writes_outputs(tmp_path: Path) -> None:
    values = [0.8, 0.9, 1.0]
    for experiment, value in zip(GROUP_OUTPUTS["deberta-v3-base"], values):
        _write_run(tmp_path, experiment, value)
    _write_run(tmp_path, "deberta-v3-base", 0.1)

    report = aggregate_seed_study(tmp_path, group_key="deberta-v3-base")
    summary = report["groups"]["deberta-v3-base"]
    f1 = summary["metrics"]["f1"]

    assert summary["complete"] is True
    assert summary["successful_runs"] == 3
    assert f1["mean"] == pytest.approx(0.9)
    assert f1["std"] == pytest.approx(statistics.stdev(values))
    assert f1["min"] == 0.8
    assert f1["max"] == 1.0
    assert f1["ddof"] == 1
    assert len(summary["historical_runs"]) == 1
    assert summary["historical_runs"][0]["metrics"]["f1"] == 0.1
    aggregate = tmp_path / "seed_studies/multinerd/deberta-v3-base/aggregate"
    assert (aggregate / "seed_summary.yaml").is_file()
    assert (aggregate / "seed_summary.csv").is_file()
    assert (aggregate / "seed_metrics.json").is_file()
    assert (aggregate / "missing_or_failed_runs.yaml").is_file()


def test_missing_seed_is_visible_and_single_run_std_is_null(tmp_path: Path) -> None:
    _write_run(tmp_path, "deberta-v3-base-canonical", 0.8)

    report = aggregate_seed_study(tmp_path, group_key="deberta-v3-base")
    summary = report["groups"]["deberta-v3-base"]

    assert summary["complete"] is False
    assert summary["partial_aggregation"] is True
    assert summary["missing_seeds"] == [123, 456]
    assert summary["metrics"]["f1"]["std"] is None


def test_failed_seed_is_not_silently_counted_as_missing(tmp_path: Path) -> None:
    _write_run(tmp_path, "deberta-v3-base-canonical", 0.8)
    failed = tmp_path / "multinerd/deberta-v3-base-canonical-seed123"
    failed.mkdir(parents=True)
    (failed / "status.json").write_text('{"status": "FAILED"}\n', encoding="utf-8")

    report = aggregate_seed_study(tmp_path, group_key="deberta-v3-base")
    summary = report["groups"]["deberta-v3-base"]

    assert summary["failed_seeds"] == [123]
    assert summary["missing_seeds"] == [456]
    assert summary["partial_aggregation"] is True


def test_historical_qwen27b_two_epoch_run_is_never_aggregated(tmp_path: Path) -> None:
    for experiment, value in zip(
        GROUP_OUTPUTS["qwen35-27b-qlora-3ep"],
        [0.8, 0.9, 1.0],
    ):
        _write_run(tmp_path, experiment, value)
    _write_run(tmp_path, "qwen35-27b-qlora", 0.1)

    report = aggregate_seed_study(tmp_path, group_key="qwen35-27b-qlora-3ep")
    summary = report["groups"]["qwen35-27b-qlora-3ep"]

    assert summary["metrics"]["f1"]["mean"] == pytest.approx(0.9)
    assert len(summary["historical_runs"]) == 1
    assert summary["historical_runs"][0]["included_in_primary_aggregate"] is False
    assert summary["historical_runs"][0]["metrics"]["f1"] == 0.1
