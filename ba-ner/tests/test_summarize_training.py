from pathlib import Path

import pytest
import yaml

from src.evaluate.summarize_training import EXPERIMENTS, summarize_training


def _write_result(root: Path, experiment: str, metric_name: str) -> None:
    experiment_dir = root / "multinerd" / experiment
    experiment_dir.mkdir(parents=True)
    with (experiment_dir / "results.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model_name": f"test/{experiment}",
                "regime": "encoder" if experiment.startswith("deberta") else "llm_lora",
                metric_name: 0.75,
                "train_runtime_seconds": 12.5,
            },
            handle,
        )

    artifact = experiment_dir / EXPERIMENTS[experiment][1]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")


def test_summarize_training_validates_all_five_models(tmp_path):
    for experiment, (metric_name, _) in EXPERIMENTS.items():
        _write_result(tmp_path, experiment, metric_name)

    summary = summarize_training(tmp_path)

    assert summary["complete"] is True
    assert summary["model_count"] == 5
    assert {model["experiment_name"] for model in summary["models"]} == set(EXPERIMENTS)


def test_summarize_training_reports_missing_results(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        summarize_training(tmp_path)
