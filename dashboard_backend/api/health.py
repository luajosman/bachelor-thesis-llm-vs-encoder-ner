"""Health and system information endpoints."""

from __future__ import annotations

import platform

from fastapi import APIRouter

from dashboard_backend.models.schemas import SystemInfo


router = APIRouter(tags=["health"])


def _get_system_info() -> SystemInfo:
    try:
        import torch
    except Exception:
        return SystemInfo(
            python_version=platform.python_version(),
            cuda_available=False,
            gpu_name=None,
            vram_total_mb=None,
        )

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    vram_total_mb = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram_total_mb = props.total_memory / (1024 * 1024)

    return SystemInfo(
        python_version=platform.python_version(),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/system", response_model=SystemInfo)
async def system_info() -> SystemInfo:
    return _get_system_info()
