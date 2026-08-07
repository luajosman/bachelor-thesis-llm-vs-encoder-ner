"""Aggregate canonical seed-study results with sample standard deviation (ddof=1)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from src.config import load_experiment_config
from src.seed_study import (
    DEFAULT_MANIFEST_PATH,
    PROJECT_ROOT,
    full_run_config_hash,
    load_manifest,
    scientific_config_hash,
)


SCALAR_METRICS = (
    "precision",
    "recall",
    "f1",
    "training_runtime_seconds",
    "inference_latency_ms_mean",
    "inference_latency_ms_p50",
    "inference_latency_ms_p95",
    "peak_vram_mb",
    "total_params",
    "trainable_params",
    "json_validity_rate",
    "parse_failure_rate",
    "schema_compliance_rate",
    "span_alignment_rate",
    "unknown_entity_type_rate",
    "invalid_offset_rate",
    "text_token_mismatch_rate",
    "overlap_rate",
)

ZERO_SHOT_EXPERIMENTS = (
    "qwen35-08b-zeroshot",
    "qwen35-4b-zeroshot",
    "qwen35-27b-zeroshot",
)


def resolve_results_dir(value: Optional[str | Path] = None) -> Path:
    if value is not None:
        return Path(value)
    configured = os.environ.get("BA_NER_RESULTS_ROOT")
    if configured:
        return Path(configured)
    scratch = Path(f"/netscratch/{os.environ.get('USER', 'losman')}/ba-ner/results")
    return scratch if scratch.is_dir() else PROJECT_ROOT / "results"


def aggregate_seed_study(
    results_dir: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    group_key: Optional[str] = None,
) -> Dict[str, Any]:
    results_root = Path(results_dir)
    manifest = load_manifest(manifest_path)
    selected_groups = [
        group for group in manifest["groups"]
        if group_key is None or group["key"] == group_key
    ]
    if group_key is not None and not selected_groups:
        raise ValueError(f"Unknown seed-study group: {group_key}")

    summaries: Dict[str, Any] = {}
    long_rows: list[Dict[str, Any]] = []
    for group in selected_groups:
        summary, rows = _aggregate_group(results_root, group)
        summaries[str(group["key"])] = summary
        long_rows.extend(rows)
        _write_group_outputs(results_root, str(group["key"]), summary, rows)

    comparison_rows = [_comparison_row(summary) for summary in summaries.values()]
    if group_key is None:
        zero_shot_rows, zero_shot_long = _zero_shot_rows(results_root)
        comparison_rows.extend(zero_shot_rows)
        long_rows.extend(zero_shot_long)
        comparison_dir = results_root / "seed_studies" / "multinerd"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        _atomic_yaml(comparison_dir / "model_comparison.yaml", {
            "ddof": 1,
            "seed_groups": summaries,
            "comparison": comparison_rows,
            "zero_shot_seed_policy": "not_applicable",
        })
        _atomic_csv(comparison_dir / "model_comparison.csv", comparison_rows)
        _atomic_csv(comparison_dir / "runs_long.csv", long_rows)

    return {
        "ddof": 1,
        "groups": summaries,
        "comparison": comparison_rows,
        "runs": long_rows,
    }


def _aggregate_group(results_root: Path, group: Mapping[str, Any]) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    rows: list[Dict[str, Any]] = []
    missing_seeds: list[int] = []
    failed_seeds: list[int] = []
    scientific_hashes: Dict[int, str] = {}

    for run in group["canonical_runs"]:
        seed = int(run["seed"])
        cfg = load_experiment_config(PROJECT_ROOT / str(run["config"]))
        scientific_hashes[seed] = scientific_config_hash(cfg)
        output_dir = _resolve_logical_output(results_root, str(run["output_dir"]))
        training = _load_yaml(output_dir / "results.yaml")
        inference = _load_yaml(output_dir / "inference_metrics.yaml")
        evaluation = _load_yaml(output_dir / "evaluation_metrics.yaml")
        status_data = _load_json(output_dir / "status.json")
        metrics = _normalize_metrics(training, inference, evaluation)

        if not _has_primary_metrics(metrics):
            status = str(status_data.get("status", "MISSING")).upper()
            if "FAILED" in status or status in {"CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                failed_seeds.append(seed)
            else:
                missing_seeds.append(seed)
        else:
            status = "COMPLETED" if run.get("source") == "existing" else str(
                status_data.get("status", "COMPLETED")
            ).upper()

        row: Dict[str, Any] = {
            "model": group["label"],
            "model_name": group["model_name"],
            "model_family": group["model_family"],
            "regime": group["regime"],
            "variant": group.get("variant", "default"),
            "max_epochs": int(group["max_epochs"]),
            "seed": seed,
            "canonical": True,
            "historical": False,
            "included_in_primary_aggregation": True,
            "status": status,
            "output_path": str(output_dir),
            "scientific_config_hash": scientific_hashes[seed],
            "full_run_config_hash": full_run_config_hash(cfg),
            **metrics,
        }
        rows.append(row)

    successful = [row for row in rows if _has_primary_metrics(row)]
    metric_summary = {
        metric: _summary([row.get(metric) for row in successful])
        for metric in SCALAR_METRICS
        if any(_number(row.get(metric)) is not None for row in successful)
    }
    if len(set(scientific_hashes.values())) != 1:
        raise RuntimeError(f"Scientific hashes differ in {group['key']}: {scientific_hashes}")

    historical: list[Dict[str, Any]] = []
    for run in group.get("historical_runs", []):
        output_dir = _resolve_logical_output(results_root, str(run["output_dir"]))
        historical_cfg = load_experiment_config(PROJECT_ROOT / str(run["config"]))
        metrics = _normalize_metrics(
            _load_yaml(output_dir / "results.yaml"),
            _load_yaml(output_dir / "inference_metrics.yaml"),
            {},
        )
        historical_row = {
            "id": run.get("id"),
            "seed": int(run["seed"]),
            "max_epochs": int(run["max_epochs"]),
            "included_in_primary_aggregate": False,
            "role": str(run.get("role", "historical_exploratory")),
            "output_path": str(output_dir),
            "metrics": metrics or None,
        }
        historical.append(historical_row)
        rows.append({
            "model": group["label"],
            "model_name": group["model_name"],
            "model_family": group["model_family"],
            "regime": group["regime"],
            "variant": str(run.get("variant", "historical")),
            "max_epochs": int(run["max_epochs"]),
            "seed": int(run["seed"]),
            "canonical": False,
            "historical": True,
            "included_in_primary_aggregation": False,
            "status": str(run.get("status", "unknown")).upper(),
            "output_path": str(output_dir),
            "scientific_config_hash": scientific_config_hash(historical_cfg),
            "full_run_config_hash": full_run_config_hash(historical_cfg),
            **metrics,
        })

    summary: Dict[str, Any] = {
        "group": group["key"],
        "model": group["label"],
        "model_name": group["model_name"],
        "model_family": group["model_family"],
        "regime": group["regime"],
        "variant": group.get("variant", "default"),
        "max_epochs": int(group["max_epochs"]),
        "canonical": True,
        "expected_seeds": [42, 123, 456],
        "successful_seeds": [int(row["seed"]) for row in successful],
        "successful_runs": len(successful),
        "expected_runs": 3,
        "complete": len(successful) == 3,
        "partial_aggregation": len(successful) != 3,
        "missing_seeds": sorted(missing_seeds),
        "failed_seeds": sorted(failed_seeds),
        "standard_deviation": "sample",
        "ddof": 1,
        "scientific_config_hash": next(iter(scientific_hashes.values())),
        "scientific_config_hashes": scientific_hashes,
        "metrics": metric_summary,
        "runs": rows,
        "historical_runs": historical,
    }
    return summary, rows


def _normalize_metrics(
    training: Mapping[str, Any],
    inference: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = {**training, **inference, **evaluation}
    sample_count = _number(merged.get("test_sample_count"))
    if sample_count is None:
        ok = _number(merged.get("parse_ok"))
        failed = _number(merged.get("parse_failed"))
        if ok is not None and failed is not None:
            sample_count = ok + failed

    metrics: Dict[str, Any] = {
        "precision": _first_number(merged, "test_precision", "precision"),
        "recall": _first_number(merged, "test_recall", "recall"),
        "f1": _first_number(merged, "test_f1", "f1"),
        "training_runtime_seconds": _first_number(merged, "train_runtime_seconds"),
        "inference_latency_ms_mean": _first_number(merged, "latency_ms_mean"),
        "inference_latency_ms_p50": _first_number(merged, "latency_ms_p50"),
        "inference_latency_ms_p95": _first_number(merged, "latency_ms_p95"),
        "peak_vram_mb": _first_number(merged, "vram_peak_mb"),
        "total_params": _first_number(merged, "total_params"),
        "trainable_params": _first_number(merged, "trainable_params"),
        "parse_failure_rate": _first_number(merged, "parse_failure_rate"),
    }
    if metrics["parse_failure_rate"] is not None:
        metrics["json_validity_rate"] = 1.0 - metrics["parse_failure_rate"]
    if sample_count and sample_count > 0:
        wrong_schema = _first_number(merged, "parse_wrong_schema") or 0.0
        invalid_items = _first_number(merged, "parse_invalid_items") or 0.0
        missing_fields = _first_number(merged, "parse_missing_fields") or 0.0
        invalid_offsets = _first_number(merged, "parse_invalid_offsets") or 0.0
        mismatches = _first_number(merged, "parse_text_mismatches") or 0.0
        unknown = _first_number(merged, "parse_unknown_types") or 0.0
        overlaps = _first_number(merged, "parse_overlaps") or 0.0
        metrics.update({
            "schema_compliance_rate": max(0.0, 1.0 - (wrong_schema + invalid_items + missing_fields) / sample_count),
            "span_alignment_rate": max(0.0, 1.0 - (invalid_offsets + mismatches) / sample_count),
            "unknown_entity_type_rate": unknown / sample_count,
            "invalid_offset_rate": invalid_offsets / sample_count,
            "text_token_mismatch_rate": mismatches / sample_count,
            "overlap_rate": overlaps / sample_count,
        })
    return {key: value for key, value in metrics.items() if value is not None}


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    numeric = [value for value in (_number(item) for item in values) if value is not None]
    return {
        "values": numeric,
        "mean": statistics.fmean(numeric) if numeric else None,
        "std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "count": len(numeric),
        "ddof": 1,
    }


def _comparison_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = summary["metrics"]
    row: Dict[str, Any] = {
        "model": summary["model"],
        "model_family": summary["model_family"],
        "regime": summary["regime"],
        "variant": summary["variant"],
        "max_epochs": summary["max_epochs"],
        "canonical": True,
        "expected_seeds": "42,123,456",
        "successful_runs": summary["successful_runs"],
        "expected_runs": summary["expected_runs"],
        "successful_runs_over_expected": f"{summary['successful_runs']}/{summary['expected_runs']}",
        "scientific_config_hash": summary["scientific_config_hash"],
        "complete": summary["complete"],
    }
    for metric, label in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("training_runtime_seconds", "training_runtime_seconds"),
        ("inference_latency_ms_mean", "inference_latency_ms_mean"),
        ("peak_vram_mb", "peak_vram_mb"),
        ("json_validity_rate", "json_validity_rate"),
        ("schema_compliance_rate", "schema_compliance_rate"),
    ):
        stats = metrics.get(metric, {})
        row[f"{label}_mean"] = stats.get("mean")
        row[f"{label}_std"] = stats.get("std")
        if metric == "f1":
            row["f1_min"] = stats.get("min")
            row["f1_max"] = stats.get("max")
    return row


def _zero_shot_rows(results_root: Path) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    comparison: list[Dict[str, Any]] = []
    long_rows: list[Dict[str, Any]] = []
    for experiment in ZERO_SHOT_EXPERIMENTS:
        output_dir = results_root / "multinerd" / experiment
        inference = _load_yaml(output_dir / "inference_metrics.yaml")
        if not inference:
            continue
        metrics = _normalize_metrics({}, inference, {})
        comparison.append({
            "model": experiment,
            "model_family": "decoder",
            "regime": "llm_zeroshot",
            "variant": "zeroshot",
            "max_epochs": 0,
            "canonical": True,
            "expected_seeds": "not_applicable",
            "successful_runs": 1,
            "expected_runs": 1,
            "successful_runs_over_expected": "1/1",
            "precision_mean": metrics.get("precision"),
            "precision_std": None,
            "recall_mean": metrics.get("recall"),
            "recall_std": None,
            "f1_mean": metrics.get("f1"),
            "f1_std": None,
            "f1_min": metrics.get("f1"),
            "f1_max": metrics.get("f1"),
            "scientific_config_hash": None,
            "complete": True,
        })
        long_rows.append({
            "model": experiment,
            "model_family": "decoder",
            "regime": "llm_zeroshot",
            "variant": "zeroshot",
            "max_epochs": 0,
            "seed": "not_applicable",
            "canonical": True,
            "historical": False,
            "included_in_primary_aggregation": True,
            "status": "COMPLETED",
            "output_path": str(output_dir),
            "scientific_config_hash": None,
            "full_run_config_hash": None,
            **metrics,
        })
    return comparison, long_rows


def _write_group_outputs(
    results_root: Path,
    group_key: str,
    summary: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> None:
    output_dir = results_root / "seed_studies" / "multinerd" / group_key / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_yaml(output_dir.parent / "manifest.yaml", {
        "group": summary["group"],
        "model_name": summary["model_name"],
        "variant": summary["variant"],
        "max_epochs": summary["max_epochs"],
        "expected_canonical_seeds": summary["expected_seeds"],
        "scientific_config_hash": summary["scientific_config_hash"],
        "canonical_runs": [row for row in rows if row.get("canonical") is True],
        "historical_runs": summary["historical_runs"],
        "complete": summary["complete"],
    })
    _atomic_yaml(output_dir / "seed_summary.yaml", summary)
    _atomic_json(output_dir / "seed_metrics.json", summary)
    _atomic_csv(output_dir / "seed_summary.csv", rows)
    _atomic_yaml(output_dir / "missing_or_failed_runs.yaml", {
        "complete": summary["complete"],
        "missing_seeds": summary["missing_seeds"],
        "failed_seeds": summary["failed_seeds"],
        "successful_runs": summary["successful_runs"],
        "expected_runs": summary["expected_runs"],
    })


def _resolve_logical_output(results_root: Path, logical: str) -> Path:
    path = Path(logical)
    return results_root.joinpath(*path.parts[1:]) if path.parts and path.parts[0] == "results" else path


def _has_primary_metrics(values: Mapping[str, Any]) -> bool:
    return all(_number(values.get(name)) is not None for name in ("precision", "recall", "f1"))


def _first_number(values: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = _number(values.get(name))
        if value is not None:
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError, TypeError):
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_yaml(path: Path, value: Any) -> None:
    _atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--group")
    args = parser.parse_args()
    report = aggregate_seed_study(
        resolve_results_dir(args.results_dir),
        manifest_path=args.manifest,
        group_key=args.group,
    )
    for key, summary in report["groups"].items():
        print(
            f"{key}: {summary['successful_runs']}/{summary['expected_runs']} successful; "
            f"missing={summary['missing_seeds']} failed={summary['failed_seeds']}"
        )


if __name__ == "__main__":
    main()
