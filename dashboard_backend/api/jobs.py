"""Safe job control endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket

from dashboard_backend.models.schemas import Job, JobCreateRequest, JobLogsResponse
from dashboard_backend.services.job_service import job_service


router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=Job)
async def create_job(request: JobCreateRequest) -> Job:
    try:
        return await job_service.create_job(request.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs", response_model=list[Job])
async def jobs() -> list[Job]:
    return job_service.list_jobs()


@router.get("/jobs/{job_id}", response_model=Job)
async def job_detail(job_id: str) -> Job:
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
async def job_logs(job_id: str) -> JobLogsResponse:
    logs = job_service.get_logs(job_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return logs


@router.post("/jobs/{job_id}/cancel", response_model=Job)
async def cancel_job(job_id: str) -> Job:
    job = await job_service.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.websocket("/jobs/{job_id}/stream")
async def stream_job_logs(websocket: WebSocket, job_id: str) -> None:
    await job_service.stream_logs(websocket, job_id)
