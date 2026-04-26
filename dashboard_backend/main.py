"""Application entrypoint for the filesystem-backed NER dashboard."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard_backend.api.artifacts import router as artifacts_router
from dashboard_backend.api.configs import router as configs_router
from dashboard_backend.api.errors import router as errors_router
from dashboard_backend.api.experiments import router as experiments_router
from dashboard_backend.api.health import router as health_router
from dashboard_backend.api.jobs import router as jobs_router
from dashboard_backend.api.metrics import router as metrics_router
from dashboard_backend.api.predictions import router as predictions_router
from dashboard_backend.utils.paths import ensure_runtime_directories


ensure_runtime_directories()

app = FastAPI(
    title="BA-NER Dashboard Backend",
    version="0.1.0",
    description="Filesystem-backed experiment dashboard for encoder vs. LLM NER comparisons.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(configs_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")
app.include_router(errors_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(artifacts_router, prefix="/api")
