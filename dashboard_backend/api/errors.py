"""Error analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from dashboard_backend.models.schemas import ErrorAnalysisResponse, ErrorComparisonResponse
from dashboard_backend.services.error_service import compare_error_analysis, get_experiment_errors


router = APIRouter(tags=["errors"])


@router.get("/experiments/{experiment_id}/errors", response_model=ErrorAnalysisResponse)
async def experiment_errors(experiment_id: str, dataset: str = Query(default="multinerd")) -> ErrorAnalysisResponse:
    return get_experiment_errors(experiment_id, dataset)


@router.get("/error-analysis/compare", response_model=ErrorComparisonResponse)
async def error_compare(
    dataset: str = Query(default="multinerd"),
    encoder_id: str | None = Query(default=None),
    llm_id: str | None = Query(default=None),
) -> ErrorComparisonResponse:
    return compare_error_analysis(dataset=dataset, encoder_id=encoder_id, llm_id=llm_id)
