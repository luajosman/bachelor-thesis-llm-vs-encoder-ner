"""Typed API contracts for the dashboard backend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ActionName = Literal[
    "run_all",
    "encoder_only",
    "zeroshot_only",
    "lora_only",
    "deberta_base",
    "deberta_large",
    "qwen35_08b_zs",
    "qwen35_08b_lora",
    "qwen35_4b_zs",
    "qwen35_4b_lora",
    "qwen35_27b_zs",
    "qwen35_27b_lora",
    "compare_only",
]

ExperimentStatus = Literal["not_started", "trained", "inferred", "complete", "failed"]
ExperimentModelType = Literal["encoder", "llm"]
ExperimentRegime = Literal["encoder", "zeroshot", "lora"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class ExperimentSummary(BaseModel):
    id: str
    experiment_name: str
    model_name: str
    model_type: ExperimentModelType
    regime: ExperimentRegime
    dataset: str
    status: ExperimentStatus
    output_dir: str
    has_results: bool
    has_inference_metrics: bool
    has_predictions: bool
    has_best_model: bool
    has_best_lora_adapter: bool


class ExperimentMetrics(BaseModel):
    test_f1: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    latency_ms_mean: float | None = None
    latency_ms_p95: float | None = None
    vram_peak_mb: float | None = None
    total_params: int | None = None
    trainable_params: int | None = None
    train_runtime_seconds: float | None = None
    best_dev_f1: float | None = None
    best_epoch: int | None = None
    parse_failure_rate: float | None = None


class PredictionSample(BaseModel):
    sample_id: str
    tokens: list[str]
    gold_bio: list[str]
    pred_bio: list[str]
    gold_entities: list[dict[str, Any]]
    pred_entities: list[dict[str, Any]]
    raw_output: str | None = None
    parse_status: str | None = None


class Job(BaseModel):
    job_id: str
    action: str
    status: JobStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration: float | None = None
    command_label: str
    exit_code: int | None = None
    log_path: str


class SystemInfo(BaseModel):
    python_version: str
    cuda_available: bool
    gpu_name: str | None = None
    vram_total_mb: float | None = None


class ConfigSummary(BaseModel):
    id: str
    name: str
    path: str
    experiment_name: str
    model_name: str
    model_type: ExperimentModelType
    regime: ExperimentRegime
    dataset: str
    output_dir: str


class ConfigDetail(ConfigSummary):
    config: dict[str, Any]


class ArtifactSummary(BaseModel):
    name: str
    path: str
    kind: Literal["file", "directory"]
    mime_type: str | None = None
    size_bytes: int | None = None
    exists: bool = True


class ArtifactContent(BaseModel):
    artifact: ArtifactSummary
    content_type: Literal["json", "yaml", "text", "directory", "binary"]
    content: Any | None = None
    children: list[ArtifactSummary] = Field(default_factory=list)


class ExperimentDetail(BaseModel):
    summary: ExperimentSummary
    config: ConfigSummary | None = None
    artifacts: list[ArtifactSummary] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    model_type: ExperimentModelType
    regime: ExperimentRegime
    dataset: str
    status: ExperimentStatus
    metrics: ExperimentMetrics


class MetricsSummary(BaseModel):
    best_f1: float | None = None
    best_experiment_id: str | None = None
    best_encoder_id: str | None = None
    best_llm_id: str | None = None
    fastest_experiment_id: str | None = None
    lowest_vram_experiment_id: str | None = None
    completed_count: int = 0
    missing_count: int = 0
    running_count: int = 0


class PerEntityMetricsResponse(BaseModel):
    experiment_id: str
    entity_types: list[str]
    metrics: dict[str, dict[str, float]]


class PerEntityComparisonItem(BaseModel):
    experiment_id: str
    experiment_name: str
    regime: ExperimentRegime
    dataset: str
    metrics: dict[str, dict[str, float]]


class PerEntityComparisonResponse(BaseModel):
    entity_types: list[str]
    experiments: list[PerEntityComparisonItem]


class ErrorAnalysisResponse(BaseModel):
    experiment_id: str
    summary: dict[str, Any]
    examples: dict[str, list[dict[str, Any]]]


class ErrorComparisonResponse(BaseModel):
    encoder_id: str | None = None
    llm_id: str | None = None
    encoder: ErrorAnalysisResponse | None = None
    llm: ErrorAnalysisResponse | None = None


class JobCreateRequest(BaseModel):
    action: ActionName


class JobLogsResponse(BaseModel):
    job: Job
    log: str

