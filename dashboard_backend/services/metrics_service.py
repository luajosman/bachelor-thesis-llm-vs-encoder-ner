"""Metrics aggregation and comparison utilities."""

from __future__ import annotations

from typing import Any

from dashboard_backend.models.schemas import ComparisonRow, ExperimentMetrics, MetricsSummary, PerEntityComparisonItem, PerEntityComparisonResponse, PerEntityMetricsResponse
from dashboard_backend.services.experiment_service import get_experiment, list_experiments
from dashboard_backend.services.prediction_service import list_predictions
from dashboard_backend.utils.paths import DEFAULT_DATASET, experiment_dir
from dashboard_backend.utils.yaml_json import load_yaml_file


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_parse_failure_rate(experiment_id: str, dataset: str = DEFAULT_DATASET) -> float | None:
    samples = list_predictions(experiment_id, dataset)
    if not samples:
        return None
    statuses = [sample.parse_status for sample in samples if sample.parse_status is not None]
    if not statuses:
        return None
    failures = sum(1 for status in statuses if status == "failed")
    return failures / len(statuses)


def get_experiment_metrics(experiment_id: str, dataset: str = DEFAULT_DATASET) -> ExperimentMetrics:
    """Normalize training and inference metrics for one experiment."""
    exp_dir = experiment_dir(experiment_id, dataset)
    train_data = load_yaml_file(exp_dir / "results.yaml")
    inference_data = load_yaml_file(exp_dir / "inference_metrics.yaml")
    merged = {**train_data, **inference_data}

    parse_failure_rate = _to_float(merged.get("parse_failure_rate"))
    if parse_failure_rate is None:
        parse_failure_rate = _derive_parse_failure_rate(experiment_id, dataset)

    return ExperimentMetrics(
        test_f1=_to_float(merged.get("test_f1", merged.get("f1"))),
        test_precision=_to_float(merged.get("test_precision", merged.get("precision"))),
        test_recall=_to_float(merged.get("test_recall", merged.get("recall"))),
        latency_ms_mean=_to_float(merged.get("latency_ms_mean")),
        latency_ms_p95=_to_float(merged.get("latency_ms_p95")),
        vram_peak_mb=_to_float(merged.get("vram_peak_mb")),
        total_params=_to_int(merged.get("total_params")),
        trainable_params=_to_int(merged.get("trainable_params")),
        train_runtime_seconds=_to_float(merged.get("train_runtime_seconds")),
        best_dev_f1=_to_float(merged.get("best_dev_f1")),
        best_epoch=_to_int(merged.get("best_epoch")),
        parse_failure_rate=parse_failure_rate,
    )


def get_metrics_summary(dataset: str = DEFAULT_DATASET) -> MetricsSummary:
    """Compute dashboard overview cards from all known experiments."""
    experiments = list_experiments(dataset)
    rows = get_metrics_comparison(dataset)

    best = max((row for row in rows if row.metrics.test_f1 is not None), key=lambda row: row.metrics.test_f1 or -1.0, default=None)
    best_encoder = max((row for row in rows if row.regime == "encoder" and row.metrics.test_f1 is not None), key=lambda row: row.metrics.test_f1 or -1.0, default=None)
    best_llm = max((row for row in rows if row.regime in {"zeroshot", "lora"} and row.metrics.test_f1 is not None), key=lambda row: row.metrics.test_f1 or -1.0, default=None)
    fastest = min((row for row in rows if row.metrics.latency_ms_mean is not None), key=lambda row: row.metrics.latency_ms_mean or 0.0, default=None)
    lowest_vram = min((row for row in rows if row.metrics.vram_peak_mb is not None), key=lambda row: row.metrics.vram_peak_mb or 0.0, default=None)

    return MetricsSummary(
        best_f1=best.metrics.test_f1 if best else None,
        best_experiment_id=best.experiment_id if best else None,
        best_encoder_id=best_encoder.experiment_id if best_encoder else None,
        best_llm_id=best_llm.experiment_id if best_llm else None,
        fastest_experiment_id=fastest.experiment_id if fastest else None,
        lowest_vram_experiment_id=lowest_vram.experiment_id if lowest_vram else None,
        completed_count=sum(1 for exp in experiments if exp.status == "complete"),
        missing_count=sum(1 for exp in experiments if exp.status in {"not_started", "failed"}),
        running_count=0,
    )


def get_metrics_comparison(dataset: str = DEFAULT_DATASET) -> list[ComparisonRow]:
    """Return normalized comparison rows for tables and charts."""
    rows: list[ComparisonRow] = []
    for experiment in list_experiments(dataset):
        rows.append(
            ComparisonRow(
                experiment_id=experiment.id,
                experiment_name=experiment.experiment_name,
                model_name=experiment.model_name,
                model_type=experiment.model_type,
                regime=experiment.regime,
                dataset=experiment.dataset,
                status=experiment.status,
                metrics=get_experiment_metrics(experiment.id, dataset),
            )
        )
    return rows


def get_per_entity_metrics(experiment_id: str, dataset: str = DEFAULT_DATASET) -> PerEntityMetricsResponse:
    """Compute per-entity metrics from normalized prediction samples."""
    try:
        from src.data.dataset_loader import get_dataset_info
        from src.evaluate.metrics import compute_per_entity_metrics
    except ModuleNotFoundError:
        return PerEntityMetricsResponse(experiment_id=experiment_id, entity_types=[], metrics={})

    samples = list_predictions(experiment_id, dataset)
    gold = [sample.gold_bio for sample in samples]
    pred = [sample.pred_bio for sample in samples]
    entity_types = get_dataset_info(dataset).entity_types if gold else []
    metrics = compute_per_entity_metrics(gold, pred) if gold and pred else {}
    return PerEntityMetricsResponse(experiment_id=experiment_id, entity_types=entity_types, metrics=metrics)


def get_per_entity_comparison(dataset: str = DEFAULT_DATASET) -> PerEntityComparisonResponse:
    """Compute per-entity metrics for all experiments that have predictions."""
    try:
        from src.data.dataset_loader import get_dataset_info
    except ModuleNotFoundError:
        return PerEntityComparisonResponse(entity_types=[], experiments=[])

    items: list[PerEntityComparisonItem] = []
    entity_types = get_dataset_info(dataset).entity_types

    for experiment in list_experiments(dataset):
        if not experiment.has_predictions:
            continue
        per_entity = get_per_entity_metrics(experiment.id, dataset)
        items.append(
            PerEntityComparisonItem(
                experiment_id=experiment.id,
                experiment_name=experiment.experiment_name,
                regime=experiment.regime,
                dataset=experiment.dataset,
                metrics=per_entity.metrics,
            )
        )

    return PerEntityComparisonResponse(entity_types=entity_types, experiments=items)
