"""Artifact listing, viewing and download endpoints."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from dashboard_backend.models.schemas import ArtifactContent, ArtifactSummary
from dashboard_backend.services.experiment_service import list_artifacts
from dashboard_backend.utils.paths import DEFAULT_DATASET, RESULTS_DIR, as_display_path, experiment_dir
from dashboard_backend.utils.yaml_json import load_json_file, load_yaml_file, read_text_file


router = APIRouter(tags=["artifacts"])


def _build_artifact_summary(path: Path) -> ArtifactSummary:
    mime_type, _ = mimetypes.guess_type(path.name)
    return ArtifactSummary(
        name=path.name,
        path=as_display_path(path),
        kind="directory" if path.is_dir() else "file",
        mime_type=mime_type,
        size_bytes=path.stat().st_size if path.exists() and path.is_file() else None,
        exists=path.exists(),
    )


def _artifact_content(path: Path) -> ArtifactContent:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    summary = _build_artifact_summary(path)
    if path.is_dir():
        children = [_build_artifact_summary(item) for item in sorted(path.iterdir())]
        return ArtifactContent(artifact=summary, content_type="directory", children=children)

    suffix = path.suffix.lower()
    if suffix == ".json":
        return ArtifactContent(artifact=summary, content_type="json", content=load_json_file(path, default={}))
    if suffix in {".yaml", ".yml"}:
        return ArtifactContent(artifact=summary, content_type="yaml", content=load_yaml_file(path))
    if suffix in {".txt", ".log", ".tex", ".md"}:
        return ArtifactContent(artifact=summary, content_type="text", content=read_text_file(path))
    return ArtifactContent(artifact=summary, content_type="binary", content=None)


@router.get("/experiments/{experiment_id}/artifacts", response_model=list[ArtifactSummary])
async def experiment_artifacts(experiment_id: str, dataset: str = Query(default=DEFAULT_DATASET)) -> list[ArtifactSummary]:
    return list_artifacts(experiment_id, dataset)


@router.get("/experiments/{experiment_id}/artifacts/{name}", response_model=ArtifactContent)
async def experiment_artifact(experiment_id: str, name: str, dataset: str = Query(default=DEFAULT_DATASET)) -> ArtifactContent:
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Nested artifact paths are not allowed")
    return _artifact_content(experiment_dir(experiment_id, dataset) / name)


@router.get("/download/comparison-table")
async def download_comparison_table() -> FileResponse:
    path = RESULTS_DIR / "comparison_table.tex"
    if not path.exists():
        raise HTTPException(status_code=404, detail="comparison_table.tex not found")
    return FileResponse(path=path, filename=path.name, media_type="text/plain")


@router.get("/download/predictions/{experiment_id}")
async def download_predictions(experiment_id: str, dataset: str = Query(default=DEFAULT_DATASET)) -> FileResponse:
    path = experiment_dir(experiment_id, dataset) / "test_predictions.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="test_predictions.json not found")
    return FileResponse(path=path, filename=path.name, media_type="application/json")
