"""Config discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dashboard_backend.models.schemas import ConfigDetail, ConfigSummary
from dashboard_backend.services.config_service import get_config, list_configs


router = APIRouter(tags=["configs"])


@router.get("/configs", response_model=list[ConfigSummary])
async def configs() -> list[ConfigSummary]:
    return list_configs()


@router.get("/configs/{config_id}", response_model=ConfigDetail)
async def config_detail(config_id: str) -> ConfigDetail:
    config = get_config(config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return config
