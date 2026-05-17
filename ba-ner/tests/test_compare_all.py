from __future__ import annotations

import yaml

from src.evaluate.compare_all import load_all_results


def test_load_all_results_reads_multinerd_layout(tmp_path) -> None:
    exp_dir = tmp_path / "results" / "multinerd" / "deberta-v3-base"
    exp_dir.mkdir(parents=True)
    with open(exp_dir / "inference_metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment_name": "deberta-v3-base",
                "dataset": "multinerd",
                "regime": "encoder",
                "test_f1": 0.91,
            },
            f,
        )

    results = load_all_results(str(tmp_path / "results"))

    assert len(results) == 1
    assert results[0]["experiment_name"] == "deberta-v3-base"
    assert results[0]["dataset"] == "multinerd"
    assert results[0]["test_f1"] == 0.91


def test_load_all_results_returns_empty_when_multinerd_root_is_missing(tmp_path) -> None:
    assert load_all_results(str(tmp_path / "results")) == []
