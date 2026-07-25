"""Validate and summarize the five final MultiNERD training runs."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


EXPERIMENTS = {
    "deberta-v3-base": ("best_validation_f1", "best_model/model.safetensors"),
    "deberta-v3-large": ("best_validation_f1", "best_model/model.safetensors"),
    "qwen35-08b-qlora": ("best_dev_f1", "best_lora_adapter/adapter_model.safetensors"),
    "qwen35-4b-qlora": ("best_dev_f1", "best_lora_adapter/adapter_model.safetensors"),
    "qwen35-27b-qlora": ("best_dev_f1", "best_lora_adapter/adapter_model.safetensors"),
}


def summarize_training(results_dir: Path) -> Dict[str, Any]:
    dataset_dir = results_dir / "multinerd"
    models = []
    errors = []

    for experiment, (metric_name, artifact_path) in EXPERIMENTS.items():
        experiment_dir = dataset_dir / experiment
        results_file = experiment_dir / "results.yaml"
        artifact_file = experiment_dir / artifact_path

        if not results_file.is_file():
            errors.append(f"{experiment}: missing {results_file}")
            continue

        with results_file.open(encoding="utf-8") as handle:
            result = yaml.safe_load(handle) or {}

        metric = result.get(metric_name)
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            errors.append(f"{experiment}: invalid {metric_name}={metric!r}")
            continue
        if not artifact_file.is_file() or artifact_file.stat().st_size == 0:
            errors.append(f"{experiment}: missing artifact {artifact_file}")
            continue

        runtime = result.get("train_runtime_seconds")
        if not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)):
            errors.append(f"{experiment}: invalid train_runtime_seconds={runtime!r}")
            continue

        models.append(
            {
                "experiment_name": experiment,
                "model_name": result.get("model_name"),
                "regime": result.get("regime"),
                "metric_name": metric_name,
                "metric_value": float(metric),
                "train_runtime_seconds": float(runtime),
                "results_file": str(results_file),
                "artifact_file": str(artifact_file),
                "artifact_size_bytes": artifact_file.stat().st_size,
            }
        )

    if errors:
        raise RuntimeError("Training result validation failed:\n- " + "\n- ".join(errors))

    return {
        "complete": True,
        "dataset": "multinerd",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    summary = summarize_training(args.results_dir)
    output_file = args.results_dir / "multinerd" / "training_summary.yaml"
    with output_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"Validated {summary['model_count']} training results: {output_file}")


if __name__ == "__main__":
    main()
