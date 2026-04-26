"""Experiment discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard_backend.models.schemas import ExperimentDetail, ExperimentSummary, PerEntityMetricsResponse
from dashboard_backend.services.experiment_service import get_experiment, list_experiments
from dashboard_backend.services.metrics_service import get_per_entity_metrics


router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=list[ExperimentSummary])
async def experiments(dataset: str = Query(default="multinerd")) -> list[ExperimentSummary]:
    return list_experiments(dataset)


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def experiment_detail(experiment_id: str, dataset: str = Query(default="multinerd")) -> ExperimentDetail:
    experiment = get_experiment(experiment_id, dataset)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.get("/experiments/{experiment_id}/per-entity", response_model=PerEntityMetricsResponse)
async def experiment_per_entity(experiment_id: str, dataset: str = Query(default="multinerd")) -> PerEntityMetricsResponse:
    return get_per_entity_metrics(experiment_id, dataset)
