"""Prediction exploration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard_backend.models.schemas import PredictionSample
from dashboard_backend.services.prediction_service import get_prediction, list_predictions


router = APIRouter(tags=["predictions"])


@router.get("/experiments/{experiment_id}/predictions", response_model=list[PredictionSample])
async def predictions(experiment_id: str, dataset: str = Query(default="multinerd")) -> list[PredictionSample]:
    return list_predictions(experiment_id, dataset)


@router.get("/experiments/{experiment_id}/predictions/{sample_id}", response_model=PredictionSample)
async def prediction(experiment_id: str, sample_id: str, dataset: str = Query(default="multinerd")) -> PredictionSample:
    item = get_prediction(experiment_id, sample_id, dataset)
    if item is None:
        raise HTTPException(status_code=404, detail="Prediction sample not found")
    return item
