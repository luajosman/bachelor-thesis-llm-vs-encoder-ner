"""Experiment discovery and artifact/status normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dashboard_backend.models.schemas import ArtifactSummary, ExperimentDetail, ExperimentSummary
from dashboard_backend.services.config_service import get_config_by_experiment_name, list_configs
from dashboard_backend.utils.paths import DEFAULT_DATASET, RESULTS_DIR, as_display_path, experiment_dir
from dashboard_backend.utils.yaml_json import load_yaml_file


def _normalize_model_type(value: str | None) -> str:
    return "encoder" if str(value).lower() == "encoder" else "llm"


def _normalize_regime(
    *,
    model_type: str,
    raw_regime: str | None = None,
    mode: str | None = None,
    experiment_name: str | None = None,
) -> str:
    if model_type == "encoder":
        return "encoder"

    regime = str(raw_regime or "").lower()
    if regime in {"llm_zeroshot", "zeroshot"}:
        return "zeroshot"
    if regime in {"llm_lora", "lora"}:
        return "lora"

    mode_value = str(mode or "").lower()
    exp_value = str(experiment_name or "").lower()
    if mode_value == "zeroshot" or "zeroshot" in exp_value:
        return "zeroshot"
    return "lora"


def _derive_status(
    *,
    exists: bool,
    regime: str,
    has_results: bool,
    has_inference_metrics: bool,
    has_predictions: bool,
    has_best_model: bool,
    has_best_lora_adapter: bool,
) -> str:
    if not exists:
        return "not_started"

    training_ready = has_best_model if regime == "encoder" else has_best_lora_adapter if regime == "lora" else True
    inference_ready = has_inference_metrics and has_predictions

    if training_ready and inference_ready:
        return "complete"
    if (has_inference_metrics and not has_predictions) or (has_predictions and not has_inference_metrics):
        return "failed"
    if has_inference_metrics or has_predictions:
        return "inferred"
    if training_ready or has_results:
        return "trained"
    return "failed"


def _artifact_flags(exp_dir: Path) -> dict[str, bool]:
    return {
        "has_results": (exp_dir / "results.yaml").exists(),
        "has_inference_metrics": (exp_dir / "inference_metrics.yaml").exists(),
        "has_predictions": (exp_dir / "test_predictions.json").exists(),
        "has_best_model": (exp_dir / "best_model").exists(),
        "has_best_lora_adapter": (exp_dir / "best_lora_adapter").exists(),
    }


def _build_summary_from_sources(
    *,
    experiment_name: str,
    dataset: str,
    exp_dir: Path,
    config_data: dict[str, Any] | None = None,
    train_data: dict[str, Any] | None = None,
    inference_data: dict[str, Any] | None = None,
) -> ExperimentSummary:
    config_data = config_data or {}
    train_data = train_data or {}
    inference_data = inference_data or {}
    merged = {**config_data, **train_data, **inference_data}

    model_type = _normalize_model_type(str(merged.get("model_type")))
    regime = _normalize_regime(
        model_type=model_type,
        raw_regime=merged.get("regime"),
        mode=merged.get("mode"),
        experiment_name=experiment_name,
    )
    flags = _artifact_flags(exp_dir)
    status = _derive_status(exists=exp_dir.exists(), regime=regime, **flags)

    output_dir = str(merged.get("output_dir", as_display_path(exp_dir)))
    model_name = str(merged.get("model_name", ""))

    return ExperimentSummary(
        id=experiment_name,
        experiment_name=experiment_name,
        model_name=model_name,
        model_type=model_type,
        regime=regime,
        dataset=dataset,
        status=status,
        output_dir=output_dir,
        **flags,
    )


def list_experiments(dataset: str = DEFAULT_DATASET) -> list[ExperimentSummary]:
    """List experiments by combining config discovery with result directories."""
    items: dict[str, ExperimentSummary] = {}

    configs = [cfg for cfg in list_configs() if cfg.dataset == dataset]
    for cfg in configs:
        exp_dir = experiment_dir(cfg.experiment_name, dataset)
        train_data = load_yaml_file(exp_dir / "results.yaml")
        inference_data = load_yaml_file(exp_dir / "inference_metrics.yaml")
        items[cfg.experiment_name] = _build_summary_from_sources(
            experiment_name=cfg.experiment_name,
            dataset=dataset,
            exp_dir=exp_dir,
            config_data=cfg.model_dump(),
            train_data=train_data,
            inference_data=inference_data,
        )

    dataset_dir = RESULTS_DIR / dataset
    if dataset_dir.exists():
        for exp_dir in sorted(dataset_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            experiment_name = exp_dir.name
            if experiment_name in items:
                continue
            cfg = get_config_by_experiment_name(experiment_name)
            train_data = load_yaml_file(exp_dir / "results.yaml")
            inference_data = load_yaml_file(exp_dir / "inference_metrics.yaml")
            items[experiment_name] = _build_summary_from_sources(
                experiment_name=experiment_name,
                dataset=dataset,
                exp_dir=exp_dir,
                config_data=cfg.model_dump() if cfg else {},
                train_data=train_data,
                inference_data=inference_data,
            )

    return sorted(items.values(), key=lambda item: item.experiment_name)


def get_experiment(experiment_id: str, dataset: str = DEFAULT_DATASET) -> ExperimentDetail | None:
    """Return one experiment including config info and artifact list."""
    summary = next((item for item in list_experiments(dataset) if item.id == experiment_id), None)
    if summary is None:
        return None

    config = get_config_by_experiment_name(summary.experiment_name)
    artifacts = list_artifacts(experiment_id=experiment_id, dataset=dataset)
    return ExperimentDetail(summary=summary, config=config, artifacts=artifacts)


def list_artifacts(experiment_id: str, dataset: str = DEFAULT_DATASET) -> list[ArtifactSummary]:
    """List top-level artifacts in an experiment directory."""
    exp_dir = experiment_dir(experiment_id, dataset)
    if not exp_dir.exists():
        return []

    artifacts: list[ArtifactSummary] = []
    for item in sorted(exp_dir.iterdir()):
        mime_type = None
        if item.is_file():
            suffix = item.suffix.lower()
            if suffix == ".json":
                mime_type = "application/json"
            elif suffix in {".yaml", ".yml"}:
                mime_type = "application/x-yaml"
            elif suffix in {".log", ".txt", ".tex"}:
                mime_type = "text/plain"
        artifacts.append(
            ArtifactSummary(
                name=item.name,
                path=as_display_path(item),
                kind="directory" if item.is_dir() else "file",
                mime_type=mime_type,
                size_bytes=item.stat().st_size if item.is_file() else None,
                exists=True,
            )
        )
    return artifacts
