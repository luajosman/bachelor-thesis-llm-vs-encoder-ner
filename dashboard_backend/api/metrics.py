"""Metrics and comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from dashboard_backend.models.schemas import ComparisonRow, ExperimentMetrics, MetricsSummary, PerEntityComparisonResponse
from dashboard_backend.services.metrics_service import get_experiment_metrics, get_metrics_comparison, get_metrics_summary, get_per_entity_comparison


router = APIRouter(tags=["metrics"])


@router.get("/experiments/{experiment_id}/metrics", response_model=ExperimentMetrics)
async def experiment_metrics(experiment_id: str, dataset: str = Query(default="multinerd")) -> ExperimentMetrics:
    return get_experiment_metrics(experiment_id, dataset)


@router.get("/metrics/summary", response_model=MetricsSummary)
async def metrics_summary(dataset: str = Query(default="multinerd")) -> MetricsSummary:
    return get_metrics_summary(dataset)


@router.get("/metrics/comparison", response_model=list[ComparisonRow])
async def metrics_comparison(dataset: str = Query(default="multinerd")) -> list[ComparisonRow]:
    return get_metrics_comparison(dataset)


@router.get("/metrics/per-entity-comparison", response_model=PerEntityComparisonResponse)
async def per_entity_comparison(dataset: str = Query(default="multinerd")) -> PerEntityComparisonResponse:
    return get_per_entity_comparison(dataset)
