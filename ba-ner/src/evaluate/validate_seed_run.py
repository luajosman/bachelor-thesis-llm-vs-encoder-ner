"""Recompute and validate strict test metrics from one managed run's predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.config import load_experiment_config, output_dir_from_config
from src.evaluate.metrics import compute_ner_metrics, compute_per_entity_metrics
from src.seed_study import SeedStudyError, atomic_write_yaml, managed_run_phase


def evaluate_saved_run(config_path: str) -> Dict[str, Any]:
    with managed_run_phase(config_path, "evaluation") as context:
        if context is None:
            raise SeedStudyError("Evaluation validation is only defined for managed seed-study runs")
        cfg = load_experiment_config(config_path)
        output_dir = output_dir_from_config(cfg)
        prediction_path = output_dir / "test_predictions.json"
        inference_path = output_dir / "inference_metrics.yaml"
        if not prediction_path.is_file() or not inference_path.is_file():
            raise SeedStudyError(
                f"Inference artifacts are incomplete: {prediction_path}, {inference_path}"
            )

        with prediction_path.open(encoding="utf-8") as handle:
            predictions = json.load(handle)
        if not isinstance(predictions, list) or not predictions:
            raise SeedStudyError(f"No test predictions in {prediction_path}")

        y_true: List[List[str]] = []
        y_pred: List[List[str]] = []
        for index, sample in enumerate(predictions):
            if not isinstance(sample, dict):
                raise SeedStudyError(f"Prediction {index} is not an object")
            gold = sample.get("gold", sample.get("gold_bio"))
            pred = sample.get("pred", sample.get("pred_bio"))
            if not isinstance(gold, list) or not isinstance(pred, list) or len(gold) != len(pred):
                raise SeedStudyError(f"Prediction {index} has invalid BIO sequences")
            y_true.append([str(value) for value in gold])
            y_pred.append([str(value) for value in pred])

        metrics = compute_ner_metrics(y_true, y_pred)
        per_entity = compute_per_entity_metrics(y_true, y_pred)
        with inference_path.open(encoding="utf-8") as handle:
            inference = yaml.safe_load(handle) or {}
        for name in ("precision", "recall", "f1"):
            stored = inference.get(f"test_{name}", inference.get(name))
            if not isinstance(stored, (int, float)) or not math.isclose(
                float(stored), float(metrics[name]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise SeedStudyError(
                    f"Stored test_{name}={stored!r} does not match recomputed {metrics[name]!r}"
                )

        checkpoint_key = "best_model" if cfg["model_type"] == "encoder" else "best_lora_adapter"
        expected_checkpoint = output_dir / checkpoint_key
        checkpoint_used = inference.get("checkpoint_used")
        if checkpoint_used is None or Path(str(checkpoint_used)).resolve() != expected_checkpoint.resolve():
            raise SeedStudyError(
                f"Inference did not record the required best checkpoint: {expected_checkpoint}"
            )

        result: Dict[str, Any] = {
            "experiment_name": cfg["experiment_name"],
            "model_name": cfg["model_name"],
            "seed": int(cfg["seed"]),
            "variant": context.descriptor.variant,
            "canonical": True,
            "included_in_primary_seed_aggregation": True,
            "evaluation": "strict_entity_level_iob2",
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "per_entity": per_entity,
            "test_sample_count": len(predictions),
            "checkpoint_used": str(expected_checkpoint),
            "checkpoint_selection": (
                "highest_validation_f1"
                if cfg["model_type"] == "encoder"
                else "highest_generative_validation_f1"
            ),
            "scientific_config_hash": context.scientific_config_hash,
            "full_run_config_hash": context.full_run_config_hash,
            "prediction_path": str(prediction_path),
            "inference_metrics_path": str(inference_path),
        }
        atomic_write_yaml(output_dir / "evaluation_metrics.yaml", result)
        context.update_metadata(
            evaluation="strict_entity_level_iob2",
            checkpoint_selection=result["checkpoint_selection"],
            checkpoint_used=str(expected_checkpoint),
            test_precision=metrics["precision"],
            test_recall=metrics["recall"],
            strict_entity_micro_f1=metrics["f1"],
            per_entity_metrics=per_entity,
            test_sample_count=len(predictions),
            test_predictions_path=str(prediction_path),
            evaluation_metrics_path=str(output_dir / "evaluation_metrics.yaml"),
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = evaluate_saved_run(args.config)
    print(
        f"Validated {result['experiment_name']} seed {result['seed']}: "
        f"F1={result['test_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
