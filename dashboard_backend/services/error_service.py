"""Error analysis built on top of normalized predictions."""

from __future__ import annotations

from collections import defaultdict

from dashboard_backend.models.schemas import ErrorAnalysisResponse, ErrorComparisonResponse
from dashboard_backend.services.experiment_service import list_experiments
from dashboard_backend.services.metrics_service import get_metrics_comparison
from dashboard_backend.services.prediction_service import list_predictions
from dashboard_backend.utils.paths import DEFAULT_DATASET


def _group_examples(raw_examples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for example in raw_examples:
        grouped[str(example.get("error", "other"))].append(example)
    return dict(grouped)


def get_experiment_errors(experiment_id: str, dataset: str = DEFAULT_DATASET) -> ErrorAnalysisResponse:
    """Return normalized error statistics for one experiment."""
    try:
        from src.data.dataset_loader import get_dataset_info
        from src.evaluate.error_analysis import analyze_decoder_errors, analyze_encoder_errors
    except ModuleNotFoundError as exc:
        return ErrorAnalysisResponse(
            experiment_id=experiment_id,
            summary={"dependency_error": str(exc)},
            examples={},
        )

    experiments = {item.id: item for item in list_experiments(dataset)}
    experiment = experiments.get(experiment_id)
    if experiment is None:
        return ErrorAnalysisResponse(experiment_id=experiment_id, summary={}, examples={})

    samples = list_predictions(experiment_id, dataset)
    if not samples:
        return ErrorAnalysisResponse(experiment_id=experiment_id, summary={}, examples={})

    if experiment.regime == "encoder":
        stats = analyze_encoder_errors(
            tokens_list=[sample.tokens for sample in samples],
            gold_tags=[sample.gold_bio for sample in samples],
            pred_tags=[sample.pred_bio for sample in samples],
        )
        summary = {
            "missed_entities": stats.missed_entities,
            "hallucinated_entities": stats.hallucinated_entities,
            "boundary_errors": stats.boundary_errors,
            "type_errors": stats.type_errors,
            "total_gold": stats.total_gold,
            "total_pred": stats.total_pred,
        }
        return ErrorAnalysisResponse(
            experiment_id=experiment_id,
            summary=summary,
            examples=_group_examples(stats.examples),
        )

    stats = analyze_decoder_errors(
        gold_entities=[sample.gold_entities for sample in samples],
        pred_entities=[sample.pred_entities for sample in samples],
        raw_outputs=[sample.raw_output or "" for sample in samples],
        parse_statuses=[sample.parse_status or "unknown" for sample in samples],
        tokens_list=[sample.tokens for sample in samples],
        valid_types=frozenset(get_dataset_info(dataset).entity_types),
    )
    summary = {
        "json_parse_failures": stats.json_parse_failures,
        "incomplete_json": stats.incomplete_json,
        "wrong_schema": stats.wrong_schema,
        "missing_fields": stats.missing_fields,
        "unknown_entity_types": stats.unknown_entity_types,
        "span_mismatches": stats.span_mismatches,
        "total_samples": stats.total_samples,
    }
    return ErrorAnalysisResponse(
        experiment_id=experiment_id,
        summary=summary,
        examples=_group_examples(stats.examples),
    )


def compare_error_analysis(
    *,
    dataset: str = DEFAULT_DATASET,
    encoder_id: str | None = None,
    llm_id: str | None = None,
) -> ErrorComparisonResponse:
    """Compare one encoder experiment against one LLM experiment."""
    rows = get_metrics_comparison(dataset)
    if encoder_id is None:
        best_encoder = max((row for row in rows if row.regime == "encoder" and row.metrics.test_f1 is not None), key=lambda row: row.metrics.test_f1 or -1.0, default=None)
        encoder_id = best_encoder.experiment_id if best_encoder else None
    if llm_id is None:
        best_llm = max((row for row in rows if row.regime in {"zeroshot", "lora"} and row.metrics.test_f1 is not None), key=lambda row: row.metrics.test_f1 or -1.0, default=None)
        llm_id = best_llm.experiment_id if best_llm else None

    return ErrorComparisonResponse(
        encoder_id=encoder_id,
        llm_id=llm_id,
        encoder=get_experiment_errors(encoder_id, dataset) if encoder_id else None,
        llm=get_experiment_errors(llm_id, dataset) if llm_id else None,
    )
