"""Generate a five-minute Markdown dashboard for the active training runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from src.config import FINAL_EXPERIMENTS
from src.data.crossner_loader import (
    DOMAINS as CROSSNER_DOMAINS,
    TEST_SIZES as CROSSNER_TEST_SIZES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training_monitor.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "training_monitor.md"
DEFAULT_BROWSER_REFRESH_SECONDS = 15
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(
    r"(?P<percent>\d+)%\|[^\r\n]*?\|\s*"
    r"(?P<step>\d+)/(?P<total>\d+)\s*\[[^\]]*?"
    r"(?P<rate>\d+(?:\.\d+)?)(?P<unit>s/it|it/s)\]"
)
INFERENCE_PROGRESS_RE = re.compile(
    r"INFERENCE_PROGRESS\s+"
    r"(?P<completed>\d+)/(?P<total>\d+)\s+"
    r"elapsed=(?P<elapsed>\d+(?:\.\d+)?)s"
)
ERROR_RE = re.compile(
    r"Traceback|OutOfMemoryError|out of memory|CUDA error|RuntimeError|Exception",
    re.IGNORECASE,
)
ACTIVE_STATES = {"RUNNING", "COMPLETING", "PENDING", "CONFIGURING"}
CROSSNER_SEEDS = (42, 123, 456)
CROSSNER_QWEN = "qwen35-27b-qlora-3ep"
_CROSSNER_PREDICTION_HASHES: dict[Path, tuple[int, int, int, str]] = {}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    job_ids: tuple[int, ...]
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    eval_seconds_low: float = 0.0
    eval_seconds_high: float = 0.0
    restart_buffer_seconds: float = 0.0
    variant: str = "default"
    seed: int = 42
    canonical: bool = True
    historical: bool = False
    max_epochs: Optional[int] = None


@dataclass(frozen=True)
class CrossnerOutputLimitConfig:
    output_root: Path
    baseline_root: Path
    max_new_tokens: int


@dataclass(frozen=True)
class MonitorConfig:
    specs: tuple[ModelSpec, ...]
    refresh_seconds: int
    scheduler_refresh_seconds: int
    browser_refresh_seconds: int
    summary_job_id: Optional[int]
    result_job_ids: dict[str, tuple[int, ...]]
    result_time_limits_seconds: dict[str, float]
    crossner_output_limit: Optional[CrossnerOutputLimitConfig] = None


@dataclass(frozen=True)
class JobState:
    job_id: int
    state: str
    elapsed: str = "-"
    location: str = "-"
    exit_code: str = "-"


@dataclass(frozen=True)
class Progress:
    step: int
    total: int
    seconds_per_step: float

    @property
    def percent(self) -> float:
        return 100.0 * self.step / self.total if self.total else 0.0


@dataclass
class ModelSnapshot:
    spec: ModelSpec
    job: JobState
    progress: Optional[Progress]
    train_metrics: dict[str, Any]
    dev_metrics: dict[str, Any]
    checkpoint_step: Optional[int]
    checkpoint_time: Optional[datetime]
    results: dict[str, Any]
    alert: Optional[str]
    eta_low_seconds: Optional[float] = None
    eta_high_seconds: Optional[float] = None
    status_data: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalResultSnapshot:
    experiment_name: str
    regime: str
    status: str
    metrics: dict[str, Any]
    job: Optional[JobState] = None
    inference_progress: Optional[InferenceProgress] = None
    time_limit_seconds: Optional[float] = None


@dataclass(frozen=True)
class CrossnerRunSnapshot:
    seed: int
    experiment_name: str
    status: str
    job: Optional[JobState]
    status_data: dict[str, Any]
    completed_domains: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class CrossnerDashboard:
    config: CrossnerOutputLimitConfig
    output_root: Path
    baseline_root: Path
    runs: tuple[CrossnerRunSnapshot, ...]
    summary: tuple[dict[str, Any], ...]
    comparison_complete: bool
    aggregation_jobs: tuple[JobState, ...]
    warning: str = ""


@dataclass(frozen=True)
class InferenceProgress:
    completed: int
    total: int
    elapsed_seconds: float

    @property
    def percent(self) -> float:
        return 100.0 * self.completed / self.total if self.total else 0.0

    @property
    def remaining_seconds(self) -> Optional[float]:
        if self.completed <= 0:
            return None
        seconds_per_sample = self.elapsed_seconds / self.completed
        return max(self.total - self.completed, 0) * seconds_per_sample


def _read_tail(path: Path, max_bytes: int = 4_000_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def parse_progress(text: str) -> Optional[Progress]:
    clean = ANSI_RE.sub("", text)
    matches = list(PROGRESS_RE.finditer(clean))
    if not matches:
        return None
    match = matches[-1]
    recent_rates = []
    for sample in matches[-50:]:
        rate = float(sample.group("rate"))
        recent_rates.append(rate if sample.group("unit") == "s/it" else 1.0 / rate)
    return Progress(
        step=int(match.group("step")),
        total=int(match.group("total")),
        seconds_per_step=statistics.median(recent_rates),
    )


def parse_inference_progress(text: str) -> Optional[InferenceProgress]:
    matches = list(INFERENCE_PROGRESS_RE.finditer(ANSI_RE.sub("", text)))
    if not matches:
        return None
    match = matches[-1]
    completed = int(match.group("completed"))
    total = int(match.group("total"))
    elapsed_seconds = float(match.group("elapsed"))
    if completed < 0 or total <= 0 or completed > total or elapsed_seconds < 0:
        return None
    return InferenceProgress(completed, total, elapsed_seconds)


def parse_train_metrics(text: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for line in ANSI_RE.sub("", text).splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "loss" in value:
            latest = value
    return latest


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout


def query_jobs(job_ids: Iterable[int]) -> dict[int, JobState]:
    ids = tuple(dict.fromkeys(job_ids))
    if not ids:
        return {}
    joined = ",".join(str(job_id) for job_id in ids)
    jobs: dict[int, JobState] = {}

    try:
        output = _run(["squeue", "-h", "-j", joined, "-o", "%i|%T|%M|%R"])
        for line in output.splitlines():
            fields = line.strip().split("|", 3)
            if len(fields) == 4 and fields[0].isdigit():
                job_id = int(fields[0])
                jobs[job_id] = JobState(job_id, fields[1], fields[2], fields[3])
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass

    try:
        output = _run([
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            joined,
            "--format=JobIDRaw,State,Elapsed,ExitCode",
        ])
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 4 or not fields[0].isdigit():
                continue
            job_id = int(fields[0])
            if job_id in jobs:
                continue
            jobs[job_id] = JobState(
                job_id,
                fields[1].split()[0].rstrip("+"),
                fields[2],
                "-",
                fields[3],
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass
    return jobs


def choose_job(spec: ModelSpec, jobs: dict[int, JobState]) -> JobState:
    candidates = [jobs[job_id] for job_id in spec.job_ids if job_id in jobs]
    if not candidates:
        return JobState(spec.job_ids[-1], "UNKNOWN") if spec.job_ids else JobState(0, "PLANNED")
    priority = {"RUNNING": 0, "COMPLETING": 1, "CONFIGURING": 2, "PENDING": 3}
    return min(
        candidates,
        key=lambda job: (
            priority.get(job.state, 10),
            -spec.job_ids.index(job.job_id),
        ),
    )


def _find_log(job_id: int, suffix: str) -> Optional[Path]:
    if job_id <= 0:
        return None
    matches = sorted((PROJECT_ROOT / "logs").rglob(f"*{job_id}.{suffix}"))
    return matches[-1] if matches else None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _latest_checkpoint(result_dir: Path) -> tuple[Optional[int], Optional[datetime]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in result_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.removeprefix("checkpoint-")), path))
        except ValueError:
            continue
    if not checkpoints:
        return None, None
    step, path = max(checkpoints)
    state_file = path / "trainer_state.json"
    timestamp = datetime.fromtimestamp(
        state_file.stat().st_mtime if state_file.is_file() else path.stat().st_mtime
    ).astimezone()
    return step, timestamp


def _checkpoint_metrics(result_dir: Path, step: Optional[int]) -> dict[str, Any]:
    if step is None:
        return {}
    path = result_dir / f"checkpoint-{step}" / "trainer_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    for row in reversed(state.get("log_history", [])):
        if isinstance(row, dict) and "loss" in row:
            return row
    return {}


def _encoder_validation_metrics(
    result_dir: Path,
    best_validation_f1: Any,
) -> dict[str, Any]:
    target_f1 = _finite_number(best_validation_f1)
    checkpoints: list[tuple[int, Path]] = []
    for path in result_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(path.parent.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))

    rows: list[dict[str, Any]] = []
    for _, path in sorted(checkpoints, reverse=True):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        rows = [
            row
            for row in state.get("log_history", [])
            if isinstance(row, dict) and _finite_number(row.get("eval_f1")) is not None
        ]
        if rows:
            break
    if not rows:
        return {}

    if target_f1 is None:
        best_row = max(rows, key=lambda row: _finite_number(row["eval_f1"]) or 0.0)
    else:
        best_row = min(
            rows,
            key=lambda row: abs((_finite_number(row["eval_f1"]) or 0.0) - target_f1),
        )
    return {
        "best_f1": best_row.get("eval_f1"),
        "best_precision": best_row.get("eval_precision"),
        "best_recall": best_row.get("eval_recall"),
        "best_epoch": best_row.get("epoch"),
    }


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def estimate_remaining(snapshot: ModelSnapshot) -> tuple[Optional[float], Optional[float]]:
    if snapshot.results:
        return 0.0, 0.0
    progress = snapshot.progress
    if progress is None or progress.seconds_per_step <= 0:
        return None, None
    remaining_steps = max(progress.total - progress.step, 0)
    training_seconds = remaining_steps * progress.seconds_per_step
    completed_epochs = len(snapshot.dev_metrics.get("epoch_results", []))
    remaining_evals = max((snapshot.spec.epochs or 0) - completed_epochs, 0)
    low = training_seconds + remaining_evals * snapshot.spec.eval_seconds_low
    high = (
        training_seconds
        + remaining_evals * snapshot.spec.eval_seconds_high
        + snapshot.spec.restart_buffer_seconds
    )
    return low, max(low, high)


def collect_snapshot(
    spec: ModelSpec,
    jobs: dict[int, JobState],
    results_root: Path,
) -> ModelSnapshot:
    job = choose_job(spec, jobs)
    result_dir = results_root / spec.key
    status_data = _load_json(result_dir / "status.json")
    run_metadata = _load_yaml(result_dir / "run_metadata.yaml")
    if status_data and job.job_id == 0:
        raw_job_id = status_data.get("slurm_job_id")
        job_id = int(raw_job_id) if str(raw_job_id or "").isdigit() else 0
        job = JobState(
            job_id,
            str(status_data.get("status", "UNKNOWN")),
            "-",
            str(status_data.get("node") or "-"),
        )
    error_log = _find_log(job.job_id, "err")
    stdout_log = _find_log(job.job_id, "out")
    error_text = _read_tail(error_log) if error_log else ""
    stdout_text = _read_tail(stdout_log) if stdout_log else ""
    progress = parse_progress(error_text)
    checkpoint_step, checkpoint_time = _latest_checkpoint(result_dir)
    metrics = parse_train_metrics(stdout_text)
    if not metrics:
        metrics = _checkpoint_metrics(result_dir, checkpoint_step)
    results = _load_yaml(result_dir / "results.yaml")
    if spec.kind == "encoder":
        dev_metrics = _encoder_validation_metrics(
            result_dir,
            results.get("best_validation_f1"),
        )
    else:
        dev_metrics = _load_yaml(result_dir / "generative_eval_state.yaml")

    alert = None
    if job.state in {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED"}:
        alert = f"Slurm state {job.state}, exit {job.exit_code}"
    elif ERROR_RE.search(error_text):
        alert = "Error signature detected in the active job log"
    elif job.state == "UNKNOWN":
        alert = "Scheduler state unavailable"

    snapshot = ModelSnapshot(
        spec=spec,
        job=job,
        progress=progress,
        train_metrics=metrics,
        dev_metrics=dev_metrics,
        checkpoint_step=checkpoint_step,
        checkpoint_time=checkpoint_time,
        results=results,
        alert=alert,
        status_data=status_data,
        run_metadata=run_metadata,
    )
    snapshot.eta_low_seconds, snapshot.eta_high_seconds = estimate_remaining(snapshot)
    return snapshot


def _choose_result_job(
    job_ids: tuple[int, ...],
    jobs: dict[int, JobState],
) -> Optional[JobState]:
    candidates = [jobs[job_id] for job_id in job_ids if job_id in jobs]
    if not candidates:
        return JobState(job_ids[-1], "UNKNOWN") if job_ids else None
    priority = {
        "RUNNING": 0,
        "COMPLETING": 1,
        "CONFIGURING": 2,
        "PENDING": 3,
    }
    return min(
        candidates,
        key=lambda job: (
            priority.get(job.state, 10),
            -job_ids.index(job.job_id),
        ),
    )


def _result_status(
    *,
    inference_complete: bool,
    job: Optional[JobState],
    has_training_metrics: bool,
    has_artifacts: bool,
) -> str:
    if inference_complete:
        return "Complete"
    if job is not None:
        return {
            "RUNNING": "Inference running",
            "COMPLETING": "Inference completing",
            "CONFIGURING": "Inference starting",
            "PENDING": "Inference queued",
            "COMPLETED": "Result pending",
        }.get(job.state, job.state)
    if has_training_metrics:
        return "Test pending"
    return "Training" if has_artifacts else "Waiting"


def collect_final_results(
    results_root: Path,
    result_job_ids: Optional[dict[str, tuple[int, ...]]] = None,
    result_time_limits_seconds: Optional[dict[str, float]] = None,
    jobs: Optional[dict[int, JobState]] = None,
) -> list[FinalResultSnapshot]:
    """Collect every canonical experiment, including runs without metrics yet."""
    result_job_ids = result_job_ids or {}
    result_time_limits_seconds = result_time_limits_seconds or {}
    jobs = jobs or {}
    snapshots = []
    for experiment in FINAL_EXPERIMENTS.values():
        result_dir = results_root / experiment.experiment_name
        training_metrics = _load_yaml(result_dir / "results.yaml")
        inference_metrics = _load_yaml(result_dir / "inference_metrics.yaml")
        metrics = {**training_metrics, **inference_metrics}
        inference_complete = all(
            _finite_number(inference_metrics.get(name)) is not None
            for name in ("test_f1", "test_precision", "test_recall")
        )
        job = _choose_result_job(
            result_job_ids.get(experiment.experiment_name, ()),
            jobs,
        )
        stdout_log = _find_log(job.job_id, "out") if job else None
        inference_progress = parse_inference_progress(
            _read_tail(stdout_log) if stdout_log else ""
        )
        has_artifacts = result_dir.is_dir() and any(result_dir.iterdir())

        snapshots.append(FinalResultSnapshot(
            experiment_name=experiment.experiment_name,
            regime=experiment.regime,
            status=_result_status(
                inference_complete=inference_complete,
                job=job,
                has_training_metrics=bool(training_metrics),
                has_artifacts=has_artifacts,
            ),
            metrics=metrics,
            job=job,
            inference_progress=inference_progress,
            time_limit_seconds=result_time_limits_seconds.get(
                experiment.experiment_name
            ),
        ))
    return snapshots


def _crossner_roots(
    config: CrossnerOutputLimitConfig, results_root: Path,
) -> tuple[Path, Path]:
    root = results_root.parent
    return tuple(
        path if path.is_absolute() else root / path
        for path in (config.output_root, config.baseline_root)
    )


def _crossner_ledger(config: CrossnerOutputLimitConfig, results_root: Path) -> dict[str, Any]:
    output_root, _ = _crossner_roots(config, results_root)
    ledger = _load_json(output_root / "submissions.json")
    identity = ledger.get("identity")
    if identity is not None and (not isinstance(identity, dict)
                                 or identity.get("max_new_tokens") != config.max_new_tokens
                                 or identity.get("smoke_limit") is not None):
        return {}
    return ledger


def _crossner_submission_rows(ledger: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = ledger.get(key, [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _crossner_job_ids(rows: Iterable[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(
        int(row["job_id"]) for row in rows
        if str(row.get("job_id") or "").isdecimal()
        and row.get("smoke_limit") is None
    ))


def monitor_job_ids(settings: MonitorConfig, results_root: Optional[Path] = None) -> list[int]:
    """Discover newly submitted CrossNER and resumed jobs on each scheduler refresh."""
    job_ids = [job_id for spec in settings.specs for job_id in spec.job_ids]
    if settings.summary_job_id:
        job_ids.append(settings.summary_job_id)
    for result_jobs in settings.result_job_ids.values():
        job_ids.extend(result_jobs)
    config = settings.crossner_output_limit
    if config is not None:
        ledger = _crossner_ledger(config, results_root or resolve_results_root())
        for key in ("jobs", "aggregations"):
            job_ids.extend(_crossner_job_ids(_crossner_submission_rows(ledger, key)))
    return list(dict.fromkeys(job_ids))


def _crossner_summary_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in ("model", "evaluation", "population", "domain"))
        grouped.setdefault(key, []).append(row)
    result = []
    for (model, evaluation, population, domain), records in grouped.items():
        summary: dict[str, Any] = {
            "model": model, "evaluation": evaluation, "population": population,
            "domain": domain, "successful_seeds": len(records), "expected_seeds": 3,
        }
        for name in ("precision", "recall", "f1", "parse_failure_rate", "output_limit_rate",
                     "latency_ms", "generated_tokens", "delta_f1_vs_256", "delta_recall_vs_256"):
            values = [_finite_number(row.get(name)) for row in records]
            usable = [value for value in values if value is not None]
            summary[name + "_mean"] = statistics.fmean(usable) if len(usable) == len(records) else None
            summary[name + "_std"] = statistics.stdev(usable) if len(usable) == len(records) and len(usable) > 1 else None
        result.append(summary)
    return tuple(result)


def _crossner_predictions_match(path: Path, expected_hash: Any) -> bool:
    """Cache hashes of completed artifacts so unchanged files are cheap to monitor."""
    if not isinstance(expected_hash, str):
        return False
    try:
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = _CROSSNER_PREDICTION_HASHES.get(path)
        if cached is None or cached[:3] != identity:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            cached = (*identity, digest.hexdigest())
            _CROSSNER_PREDICTION_HASHES[path] = cached
        return cached[3] == expected_hash
    except (OSError, ValueError):
        return False


def _crossner_references_match(metrics: dict[str, Any], baseline_root: Path, seed: int, domain: str) -> bool:
    artifacts = metrics.get("reference_artifacts")
    if not isinstance(artifacts, dict):
        return False
    for evaluation, base in (("encoder_baseline", "deberta-v3-large-canonical"), ("baseline_256", CROSSNER_QWEN)):
        experiment = base + (f"-seed{seed}" if seed != 42 else "")
        expected_path = baseline_root / experiment / domain / "test_predictions.json"
        artifact = artifacts.get(evaluation)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            return False
        try:
            if Path(artifact["path"]).resolve() != expected_path.resolve():
                return False
        except (OSError, ValueError):
            return False
        baseline = _load_yaml(expected_path.parent / "inference_metrics.yaml")
        if (baseline.get("complete") is not True
                or baseline.get("sample_count") != CROSSNER_TEST_SIZES[domain]
                or not isinstance(artifact.get("contract_hash"), str)
                or baseline.get("contract_hash") != artifact["contract_hash"]
                or baseline.get("predictions_sha256") != artifact.get("predictions_sha256")
                or not _crossner_predictions_match(expected_path, artifact.get("predictions_sha256"))):
            return False
    return True


def collect_crossner_dashboard(
    config: CrossnerOutputLimitConfig, results_root: Path, jobs: dict[int, JobState],
) -> CrossnerDashboard:
    """Read only completed domain metrics; partial files never interrupt the training monitor."""
    output_root, baseline_root = _crossner_roots(config, results_root)
    ledger = _crossner_ledger(config, results_root)
    submissions = _crossner_submission_rows(ledger, "jobs")
    reference_rows: list[dict[str, Any]] = []
    runs = []
    evaluations = {"encoder_baseline", "baseline_256", f"output_limit_{config.max_new_tokens}"}
    for seed in CROSSNER_SEEDS:
        experiment = CROSSNER_QWEN + (f"-seed{seed}" if seed != 42 else "")
        directory = output_root / experiment
        status_data = _load_json(directory / "status.json")
        error = str(status_data.get("error") or "")
        if (directory / "status.json").exists() and not status_data:
            error = "Unreadable status file"
        if status_data and (status_data.get("smoke_limit") is not None
                            or status_data.get("max_new_tokens") != config.max_new_tokens
                            or status_data.get("seed") != seed):
            status_data = {}
            error = "Status identity differs; excluded from results"
        job_ids = _crossner_job_ids(
            row for row in submissions if row.get("experiment") == experiment
            and row.get("max_new_tokens", config.max_new_tokens) == config.max_new_tokens
        )
        job = _choose_result_job(job_ids, jobs)
        if job is None and str(status_data.get("job_id") or "").isdecimal():
            job_id = int(status_data["job_id"])
            job = jobs.get(job_id, JobState(job_id, str(status_data.get("status", "UNKNOWN"))))
        if (job is not None and job.state in ACTIVE_STATES
                and str(status_data.get("job_id") or "").isdecimal()
                and int(status_data["job_id"]) != job.job_id):
            # A resumed job has its own clock; the previous attempt's ETA is stale.
            status_data = {}
            error = ""
        completed = []
        for domain in CROSSNER_DOMAINS:
            metrics_path = directory / domain / "inference_metrics.yaml"
            metrics = _load_yaml(metrics_path)
            if not metrics:
                if metrics_path.exists():
                    error = "Unreadable domain metrics; excluded from results"
                continue
            contract = metrics.get("contract")
            if (not isinstance(contract, dict) or metrics.get("complete") is not True
                    or contract.get("smoke_limit") is not None):
                continue
            decoder = contract.get("decoder", {})
            source = contract.get("source", {})
            if (not isinstance(decoder, dict) or decoder.get("max_new_tokens") != config.max_new_tokens
                    or not isinstance(source, dict) or source.get("seed") != seed
                    or source.get("experiment_name") != experiment
                    or contract.get("domain") != domain
                    or metrics.get("sample_count") != CROSSNER_TEST_SIZES[domain]):
                error = "Domain contract differs; excluded from results"
                continue
            if not _crossner_predictions_match(metrics_path.parent / "test_predictions.json", metrics.get("predictions_sha256")):
                error = "Predictions missing or altered; excluded from results"
                continue
            if not _crossner_references_match(metrics, baseline_root, seed, domain):
                error = "Baseline artifacts missing or altered; excluded from comparison"
                continue
            completed.append(domain)
            references = metrics.get("reference_metrics", {})
            if not isinstance(references, dict):
                continue
            for population in ("full", "deduplicated"):
                population_metrics = references.get(population, {})
                if not isinstance(population_metrics, dict):
                    continue
                for evaluation, row in population_metrics.items():
                    if (evaluation not in evaluations or not isinstance(row, dict)
                            or row.get("seed") != seed
                            or any(_finite_number(row.get(name)) is None for name in ("precision", "recall", "f1"))):
                        continue
                    expected_model = "deberta-v3-large" if evaluation == "encoder_baseline" else CROSSNER_QWEN
                    expected_limit = (None if evaluation == "encoder_baseline" else
                                      config.max_new_tokens if evaluation.startswith("output_limit_") else 256)
                    if row.get("model") != expected_model or row.get("max_new_tokens") != expected_limit:
                        continue
                    reference_rows.append({**row, "evaluation": evaluation, "population": population, "domain": domain})
        if len(completed) == len(CROSSNER_DOMAINS):
            state = "COMPLETED"
        elif job is not None and job.state != "UNKNOWN":
            state = "Result pending" if job.state == "COMPLETED" else job.state
        else:
            state = str(status_data.get("status") or ("UNKNOWN" if job else "PLANNED"))
            if state == "COMPLETED":
                state = "Result pending"
        if state == "Result pending" and not error:
            error = "Completion artifacts missing or invalid"
        runs.append(CrossnerRunSnapshot(seed, experiment, state, job, status_data, tuple(completed), error))

    comparison = _load_yaml(output_root / "aggregate" / "comparison.yaml")
    summary = _crossner_summary_rows(reference_rows)
    comparison_complete = False
    report_rows = comparison.get("summary")
    warning = ""
    report_baseline = comparison.get("baseline_output_root")
    try:
        baseline_matches = isinstance(report_baseline, str) and Path(report_baseline).resolve() == baseline_root.resolve()
    except (OSError, ValueError):
        baseline_matches = False
    if (output_root / "aggregate" / "comparison.yaml").exists() and (
        comparison.get("max_new_tokens") != config.max_new_tokens or not baseline_matches or not isinstance(report_rows, list)
    ):
        warning = "Comparison report unreadable or metadata differs; excluded"
    if comparison.get("max_new_tokens") == config.max_new_tokens and baseline_matches and isinstance(report_rows, list):
        valid_rows = []
        for row in report_rows:
            if (not isinstance(row, dict)
                    or any(not isinstance(row.get(name), str) for name in ("model", "evaluation", "population", "domain"))
                    or type(row.get("successful_seeds")) is not int or type(row.get("expected_seeds")) is not int
                    or not 0 <= row["successful_seeds"] <= row["expected_seeds"] <= 3
                    or row.get("evaluation") not in evaluations
                    or row.get("population") not in {"full", "deduplicated"}
                    or row.get("domain") not in {*CROSSNER_DOMAINS, "pooled_micro", "domain_mean"}
                    or row.get("model") not in {"deberta-v3-large", CROSSNER_QWEN}):
                continue
            expected_model = "deberta-v3-large" if row["evaluation"] == "encoder_baseline" else CROSSNER_QWEN
            if row["model"] != expected_model:
                continue
            verified_count = sum(
                row["domain"] in run.completed_domains
                if row["domain"] in CROSSNER_DOMAINS else len(run.completed_domains) == len(CROSSNER_DOMAINS)
                for run in runs
            )
            if row.get("successful_seeds") != verified_count:
                continue
            valid_rows.append(row)
        if valid_rows:
            # Freshly completed domains can appear before the next CPU aggregation.
            merged = {(row["evaluation"], row["population"], row["domain"]): row for row in summary}
            for row in valid_rows:
                key = (row["evaluation"], row["population"], row["domain"])
                if row.get("successful_seeds", 0) >= merged.get(key, {}).get("successful_seeds", 0):
                    merged[key] = row
            summary = tuple(merged.values())
            expected_rows = len(evaluations) * 2 * (len(CROSSNER_DOMAINS) + 2)
            selection = comparison.get("selection", {})
            full_selection = isinstance(selection, dict) and all(
                isinstance(selection.get(name), list)
                and len(selection[name]) == len(expected)
                and all(value in expected for value in selection[name])
                and all(value in selection[name] for value in expected)
                for name, expected in (("models", [CROSSNER_QWEN]), ("seeds", CROSSNER_SEEDS), ("domains", CROSSNER_DOMAINS))
            )
            unique_rows = {(row["evaluation"], row["population"], row["domain"]) for row in valid_rows}
            comparison_complete = (
                comparison.get("complete") is True and comparison.get("selection_complete") is True
                and comparison.get("expected_evaluations") == 15 and comparison.get("successful_evaluations") == 15
                and comparison.get("missing_or_failed") == [] and full_selection
                and len(valid_rows) == len(unique_rows) == expected_rows
                and all(len(run.completed_domains) == len(CROSSNER_DOMAINS) for run in runs)
                and all(
                    row.get("successful_seeds") == 3 and row.get("expected_seeds") == 3
                    and all(_finite_number(row.get(name + "_mean")) is not None for name in ("precision", "recall", "f1"))
                    for row in valid_rows
                )
            )
        if len(valid_rows) != len(report_rows) or (comparison.get("complete") is True and not comparison_complete):
            warning = "Comparison report incomplete or contains invalid rows"
    aggregation_ids = _crossner_job_ids(_crossner_submission_rows(ledger, "aggregations"))
    aggregations = tuple(jobs.get(job_id, JobState(job_id, "UNKNOWN")) for job_id in aggregation_ids)
    return CrossnerDashboard(config, output_root, baseline_root, tuple(runs), summary, comparison_complete, aggregations, warning)


def _format_percent(value: Any) -> str:
    number = _finite_number(value)
    return "-" if number is None else f"{100.0 * number:.2f}%"


def _format_metric(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _format_interval(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} seconds"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} seconds"


def _format_epoch(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds <= 0:
        return "complete"
    hours = seconds / 3600.0
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


def _format_compact_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _format_clock_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


def _slurm_elapsed_seconds(value: str) -> Optional[float]:
    if not value or value == "-":
        return None
    try:
        days = 0
        clock = value
        if "-" in value:
            day_text, clock = value.split("-", 1)
            days = int(day_text)
        parts = [int(part) for part in clock.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = minutes = 0
            seconds = parts[0]
        else:
            return None
    except ValueError:
        return None
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _inference_time_cells(
    result: FinalResultSnapshot,
    now: datetime,
) -> tuple[str, str, str, str, str]:
    job = result.job
    elapsed_seconds = _slurm_elapsed_seconds(job.elapsed) if job else None
    elapsed_cell = _format_compact_duration(elapsed_seconds)
    if result.status == "Complete":
        return elapsed_cell, "100%", elapsed_cell, "complete", "complete"

    progress = result.inference_progress
    if progress is not None:
        remaining = progress.remaining_seconds
        total = (
            elapsed_seconds + remaining
            if elapsed_seconds is not None and remaining is not None
            else progress.elapsed_seconds + (remaining or 0.0)
        )
        finish = (
            (now + timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M %Z")
            if remaining is not None
            else "-"
        )
        return (
            elapsed_cell,
            f"{progress.percent:.1f}% ({progress.completed:,}/{progress.total:,})",
            f"~{_format_compact_duration(total)}",
            f"~{_format_compact_duration(remaining)}",
            finish,
        )

    limit = result.time_limit_seconds
    if job is not None and job.state in ACTIVE_STATES and limit is not None:
        if job.state == "RUNNING":
            remaining_budget = max(limit - (elapsed_seconds or 0.0), 0.0)
            finish = (
                now + timedelta(seconds=remaining_budget)
            ).strftime("by %Y-%m-%d %H:%M %Z")
            return (
                elapsed_cell,
                "measuring",
                f"measuring (limit: {_format_compact_duration(limit)})",
                f"measuring (budget: {_format_compact_duration(remaining_budget)})",
                f"{finish} if the full limit is used",
            )
        return (
            elapsed_cell,
            "not started",
            f"unknown (limit: {_format_compact_duration(limit)})",
            "unknown until start",
            "waiting for scheduler",
        )

    return elapsed_cell, "-", "-", "-", "-"


def _format_eta_range(low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None:
        return "-"
    if high <= 0:
        return "complete"
    return f"{_format_duration(low)} - {_format_duration(high)}"


def _format_finish(now: datetime, low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None:
        return "-"
    if high <= 0:
        return "complete"
    start = now + timedelta(seconds=low)
    end = now + timedelta(seconds=high)
    zone = now.tzname() or ""
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}-{end:%H:%M} {zone}".strip()
    return f"{start:%Y-%m-%d %H:%M} - {end:%Y-%m-%d %H:%M} {zone}".strip()


def _overall_live_estimate(
    snapshots: list[ModelSnapshot],
    now: datetime,
) -> tuple[str, str, str]:
    """Summarize canonical work without treating unscheduled runs as complete."""
    canonical = [
        snapshot
        for snapshot in snapshots
        if snapshot.spec.canonical and not snapshot.spec.historical
    ]
    tracked = canonical or snapshots
    if not tracked:
        return "NO RUNS", "-", "-"

    states = {snapshot.job.state.upper() for snapshot in tracked}
    failed_states = {
        "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED",
    }
    if states & failed_states:
        return "ATTENTION", "check failed jobs", "-"

    planned_count = sum(snapshot.job.state.upper() == "PLANNED" for snapshot in tracked)
    if planned_count:
        state = "PLANNED" if planned_count == len(tracked) else "PARTIALLY SUBMITTED"
        return state, "waiting for submission", "-"

    running = [
        snapshot
        for snapshot in tracked
        if snapshot.job.state.upper() in {"RUNNING", "COMPLETING"}
    ]
    queued = states & {"PENDING", "CONFIGURING"}
    if running:
        lows = [snapshot.eta_low_seconds for snapshot in running]
        highs = [snapshot.eta_high_seconds for snapshot in running]
        if any(value is None for value in [*lows, *highs]):
            return "Parallel", "measuring", "-"
        overall_low = max(float(value) for value in lows if value is not None)
        overall_high = max(float(value) for value in highs if value is not None)
        state = "RUNNING / QUEUED" if queued else "Parallel"
        return (
            state,
            _format_eta_range(overall_low, overall_high),
            _format_finish(now, overall_low, overall_high),
        )
    if queued:
        return "QUEUED", "waiting for scheduler", "-"

    if all(snapshot.job.state.upper() == "COMPLETED" or snapshot.results for snapshot in tracked):
        return "COMPLETED", "complete", "complete"
    return "UNKNOWN", "-", "-"


def _validation_f1(result: FinalResultSnapshot) -> Any:
    return (
        result.metrics.get("best_validation_f1")
        or result.metrics.get("best_dev_f1")
        or result.metrics.get("best_f1")
    )


def _regime_label(regime: str) -> str:
    return {
        "encoder": "Encoder",
        "llm_lora": "LLM LoRA/QLoRA",
        "llm_zeroshot": "LLM zero-shot",
    }.get(regime, regime)


def _progress_cell(snapshot: ModelSnapshot) -> str:
    if snapshot.progress is not None:
        return (
            f"{snapshot.progress.percent:.1f}% "
            f"({snapshot.progress.step:,}/{snapshot.progress.total:,})"
        )
    if snapshot.results:
        if snapshot.spec.kind == "decoder" and snapshot.spec.total_steps:
            total = snapshot.spec.total_steps
            return f"100.0% ({total:,}/{total:,})"
        return "100% (complete)"
    return "-"


def _epoch_cell(snapshot: ModelSnapshot) -> str:
    if snapshot.progress is not None and snapshot.spec.epochs:
        epoch = min(
            snapshot.spec.epochs,
            snapshot.progress.step
            / (snapshot.progress.total / snapshot.spec.epochs),
        )
        return f"{epoch:.2f}/{snapshot.spec.epochs}"
    if snapshot.results:
        epochs = (
            snapshot.results.get("num_train_epochs")
            if snapshot.spec.kind == "encoder"
            else snapshot.spec.epochs
        )
        if not epochs:
            return "-"
        return (
            f"{float(epochs):.2f}/{epochs}"
            if snapshot.spec.kind == "decoder"
            else f"{epochs}/{epochs}"
        )
    return "-"


def _best_validation_row(snapshot: ModelSnapshot) -> dict[str, Any]:
    if snapshot.spec.kind == "encoder":
        return {
            "dev_precision": snapshot.dev_metrics.get("best_precision"),
            "dev_recall": snapshot.dev_metrics.get("best_recall"),
            "dev_f1": snapshot.dev_metrics.get("best_f1")
            or snapshot.results.get("best_validation_f1"),
            "epoch": snapshot.dev_metrics.get("best_epoch"),
        }
    rows = snapshot.dev_metrics.get("epoch_results", [])
    best_epoch = snapshot.dev_metrics.get("best_epoch")
    return next(
        (
            row for row in rows
            if isinstance(row, dict) and row.get("epoch") == best_epoch
        ),
        {},
    )


def _seed_snapshot_status(snapshot: ModelSnapshot, now: datetime) -> str:
    status = str(snapshot.status_data.get("status") or snapshot.job.state)
    raw_update = snapshot.status_data.get("last_update")
    if status in {"RUNNING", "VALIDATING", "SAVING", "INFERENCE_RUNNING", "EVALUATION_RUNNING"} and raw_update:
        try:
            updated = datetime.fromisoformat(str(raw_update))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=now.tzinfo)
            if (now - updated.astimezone(now.tzinfo)).total_seconds() > 600:
                return f"STALE ({status})"
        except (TypeError, ValueError):
            return f"STALE ({status})"
    return status


def _seed_gpu(snapshot: ModelSnapshot) -> str:
    runtime = snapshot.run_metadata.get("runtime", {})
    cuda = runtime.get("cuda", {}) if isinstance(runtime, dict) else {}
    devices = cuda.get("devices", []) if isinstance(cuda, dict) else []
    if isinstance(devices, list) and devices:
        rendered = []
        for device in devices:
            if isinstance(device, dict):
                rendered.append(str(device.get("name") or device.get("uuid") or "unknown"))
            else:
                rendered.append(str(device))
        return ", ".join(rendered)
    return "-"


def _seed_update_cell(status_data: dict[str, Any], now: datetime) -> str:
    raw = status_data.get("last_update")
    if not raw:
        return "-"
    try:
        updated = datetime.fromisoformat(str(raw))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=now.tzinfo)
        age = max(0.0, (now - updated.astimezone(now.tzinfo)).total_seconds())
        return f"{raw} ({_format_clock_duration(age)} old)"
    except (TypeError, ValueError):
        return str(raw)


def _seed_peak_vram(status_data: dict[str, Any], results: dict[str, Any]) -> str:
    megabytes = _finite_number(status_data.get("peak_vram_mb"))
    if megabytes is None:
        raw_bytes = _finite_number(status_data.get("peak_vram_bytes"))
        if raw_bytes is not None:
            megabytes = raw_bytes / (1024.0 * 1024.0)
    if megabytes is None:
        megabytes = _finite_number(results.get("vram_peak_mb"))
    return "-" if megabytes is None else f"{megabytes:.1f} MiB"


def _crossner_cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ").replace("\r", " ")


def _crossner_summary_metric(row: dict[str, Any], name: str, *, unit: str = "%") -> str:
    mean = _finite_number(row.get(name + "_mean"))
    if mean is None:
        return "-"
    std = _finite_number(row.get(name + "_std"))
    scale = 100 if unit in {"%", "pp"} else 1
    cell = f"{mean * scale:.2f}"
    if std is not None:
        cell += f" ± {std * scale:.2f}"
    return cell + ("%" if unit == "%" else " " + unit)


def render_crossner_markdown(dashboard: CrossnerDashboard, now: datetime) -> list[str]:
    limit = dashboard.config.max_new_tokens
    lines = [
        "", "## CrossNER Output Limit Evaluation", "",
        f"Qwen 27B QLoRA: **256 → {limit} max_new_tokens**; frozen checkpoints, PER / ORG / LOC.",
        f"Results: `{_crossner_cell(dashboard.output_root)}`; baseline: `{_crossner_cell(dashboard.baseline_root)}`.",
        "", "| Seed | Limit | Status | Job | Domain | Domains complete | Progress | Elapsed | Remaining | Parse failures | Output limit | Updated / age | Error |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    expected_total = sum(CROSSNER_TEST_SIZES.values())
    for run in dashboard.runs:
        data = run.status_data
        completed = _finite_number(data.get("completed_samples"))
        verified = sum(CROSSNER_TEST_SIZES[domain] for domain in run.completed_domains)
        completed = max(completed or 0, verified)
        total = _finite_number(data.get("total_samples")) or expected_total
        if completed > total or total <= 0:
            completed, total = verified, expected_total
        progress = f"{100 * completed / total:.1f}% ({int(completed):,}/{int(total):,})" if total > 0 else "-"
        elapsed = _finite_number(data.get("elapsed_seconds"))
        eta = _finite_number(data.get("eta_seconds"))
        remaining = "complete" if run.status == "COMPLETED" else (
            _format_compact_duration(eta) if eta is not None and eta >= 0 else "unknown"
        )
        rates = []
        for key in ("parse_failure_count", "output_limit_count"):
            count = _finite_number(data.get(key))
            rates.append(_format_percent(count / completed) if count is not None and 0 <= count <= completed and completed > 0 else "-")
        updated = _seed_update_cell({"last_update": data.get("updated_at")}, now)
        error = run.error
        if run.job and run.job.state in {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED"} and not error:
            error = f"Slurm {run.job.state}, exit {run.job.exit_code}"
        lines.append(
            f"| {run.seed} | {limit} | {_crossner_cell(run.status)} | {run.job.job_id if run.job else '-'} | "
            f"{_crossner_cell(data.get('current_domain') or '-')} | {len(run.completed_domains)}/5 | {progress} | "
            f"{_format_compact_duration(elapsed) if elapsed is not None else run.job.elapsed if run.job else '-'} | "
            f"{remaining} | {rates[0]} | {rates[1]} | {updated} | {_crossner_cell(error or '-')} |"
        )
    aggregation_status = ", ".join(f"`{job.job_id}` ({job.state})" for job in dashboard.aggregation_jobs) or "not submitted"
    lines.extend(["", f"Comparison aggregation: {aggregation_status}."])
    complete = dashboard.comparison_complete and all(run.status == "COMPLETED" for run in dashboard.runs)
    lines.append(f"Comparison: **{'COMPLETE' if complete else 'INCOMPLETE'}**; means ± sample SD across completed seeds.")
    if dashboard.warning:
        lines.append(f"Warning: {_crossner_cell(dashboard.warning)}.")
    for population, label in (("full", "Full test set"), ("deduplicated", "Without source-training overlaps")):
        lines.extend([
            "", f"## CrossNER Comparison — {label}", "",
            "| Domain | Model / limit | Seeds | Status | Precision | Recall | F1 | ΔF1 vs 256 | ΔRecall vs 256 | Parse failures | Output limit | Mean latency |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for domain in (*CROSSNER_DOMAINS, "pooled_micro", "domain_mean"):
            for evaluation in ("encoder_baseline", "baseline_256", f"output_limit_{limit}"):
                row = next((row for row in dashboard.summary
                            if row.get("population") == population and row.get("domain") == domain
                            and row.get("evaluation") == evaluation), {})
                count = row.get("successful_seeds", 0)
                expected = row.get("expected_seeds", 3)
                state = "Complete" if count == 3 and expected == 3 else "Incomplete"
                model = "DeBERTa large" if evaluation == "encoder_baseline" else f"Qwen {256 if evaluation == 'baseline_256' else limit}"
                metrics = [_crossner_summary_metric(row, name) for name in ("precision", "recall", "f1")]
                deltas = [_crossner_summary_metric(row, name, unit="pp") for name in ("delta_f1_vs_256", "delta_recall_vs_256")]
                diagnostics = [_crossner_summary_metric(row, name) for name in ("parse_failure_rate", "output_limit_rate")]
                latency = _crossner_summary_metric(row, "latency_ms", unit="ms")
                lines.append(f"| {domain} | {model} | {_crossner_cell(count)}/{_crossner_cell(expected)} | {state} | "
                             + " | ".join([*metrics, *deltas, *diagnostics, latency]) + " |")
    lines.extend(["", "Δ values compare CrossNER with the 256-token CrossNER baseline. "
                  "Pooled micro scores appear after aggregation; missing historical output-limit telemetry is shown as '-'."])
    return lines


def render_markdown(
    snapshots: list[ModelSnapshot],
    now: datetime,
    refresh_seconds: int,
    summary_job: Optional[JobState],
    browser_refresh_seconds: int = DEFAULT_BROWSER_REFRESH_SECONDS,
    final_results: Optional[list[FinalResultSnapshot]] = None,
    crossner_dashboard: Optional[CrossnerDashboard] = None,
) -> str:
    decoder_snapshots = [s for s in snapshots if s.spec.kind == "decoder"]
    lines = [
        "# Training Live Monitor",
        "",
        f"**Updated:** {now:%Y-%m-%d %H:%M:%S %Z}  ",
        f"**Data refresh:** every {_format_interval(refresh_seconds)}  ",
        f"**Page refresh:** every {_format_interval(browser_refresh_seconds)}  ",
        f"**Next update:** {(now + timedelta(seconds=refresh_seconds)):%H:%M:%S %Z}",
        "",
        "## Live Estimate",
        "",
        "| Model | Status | Progress | Epoch | Realistic remaining | Expected finish |",
        "|---|---|---:|---:|---:|---|",
    ]
    for snapshot in snapshots:
        lines.append(
            "| {label} | {state} | {progress} | {epoch} | {remaining} | {finish} |".format(
                label=snapshot.spec.label,
                state=snapshot.job.state,
                progress=_progress_cell(snapshot),
                epoch=_epoch_cell(snapshot),
                remaining=_format_eta_range(
                    snapshot.eta_low_seconds,
                    snapshot.eta_high_seconds,
                ),
                finish=_format_finish(
                    now,
                    snapshot.eta_low_seconds,
                    snapshot.eta_high_seconds,
                ),
            )
        )

    overall_state, overall_remaining, overall_finish = _overall_live_estimate(
        snapshots,
        now,
    )
    lines.append(
        f"| **All models** | {overall_state} | - | - | "
        f"**{overall_remaining}** | **{overall_finish}** |"
    )
    lines.extend([
        "",
        f"**Estimated time until all models finish:** "
        f"{overall_remaining}",
        "",
        "## Live Training Metrics",
        "",
        "| Model | Job | Node | Runtime | Step time | Loss | Token accuracy | Learning rate | Grad norm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for snapshot in decoder_snapshots:
        recorded_runtime = _finite_number(
            snapshot.results.get("train_runtime_seconds")
        )
        if snapshot.progress is not None:
            seconds_per_step = snapshot.progress.seconds_per_step
        elif recorded_runtime is not None and snapshot.spec.total_steps:
            seconds_per_step = recorded_runtime / snapshot.spec.total_steps
        else:
            seconds_per_step = None
        rate = (
            f"{seconds_per_step:.2f} s"
            if seconds_per_step is not None
            else "-"
        )
        runtime = (
            _format_clock_duration(recorded_runtime)
            if recorded_runtime is not None
            else snapshot.job.elapsed
        )
        lines.append(
            "| {label} | {job} | {node} | {runtime} | {rate} | {loss} | "
            "{accuracy} | {lr} | {grad} |".format(
                label=snapshot.spec.label,
                job=snapshot.job.job_id,
                node=snapshot.job.location,
                runtime=runtime,
                rate=rate,
                loss=_format_metric(snapshot.train_metrics.get("loss")),
                accuracy=_format_percent(
                    snapshot.train_metrics.get("mean_token_accuracy")
                ),
                lr=_format_metric(snapshot.train_metrics.get("learning_rate"), 7),
                grad=_format_metric(snapshot.train_metrics.get("grad_norm")),
            )
        )

    lines.extend([
        "",
        "## Validation Results",
        "",
        "| Model | Best validation F1 | Precision | Recall | Best epoch | Parse failures |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for snapshot in snapshots:
        if snapshot.spec.kind == "encoder":
            lines.append(
                f"| {snapshot.spec.label} | "
                f"{_format_percent(snapshot.results.get('best_validation_f1') or snapshot.dev_metrics.get('best_f1'))} | "
                f"{_format_percent(snapshot.dev_metrics.get('best_precision'))} | "
                f"{_format_percent(snapshot.dev_metrics.get('best_recall'))} | "
                f"{_format_epoch(snapshot.dev_metrics.get('best_epoch'))} | "
                "N/A |"
            )
            continue
        epoch_results = snapshot.dev_metrics.get("epoch_results", [])
        best_epoch = snapshot.dev_metrics.get("best_epoch")
        best_row = next(
            (
                row
                for row in epoch_results
                if isinstance(row, dict) and row.get("epoch") == best_epoch
            ),
            {},
        )
        lines.append(
            f"| {snapshot.spec.label} | "
            f"{_format_percent(snapshot.dev_metrics.get('best_f1'))} | "
            f"{_format_percent(best_row.get('dev_precision'))} | "
            f"{_format_percent(best_row.get('dev_recall'))} | "
            f"{_format_epoch(best_epoch)} | "
            f"{_format_percent(best_row.get('parse_failure_rate'))} |"
        )

    lines.extend([
        "",
        "## Canonical Seed Study Runs",
        "",
        "| Model | Family | Regime | Variant | Max epochs | Seed | Role | Phase | Status | Job | Array | Node | GPU | Started | Elapsed | Epoch | Step | Progress | Train loss | Val loss | Val P | Val R | Val F1 | Best F1 | Best step | Best epoch | LR | Peak VRAM | ETA | Output | Updated / age | Error |",
        "|---|---|---|---|---:|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for snapshot in snapshots:
        status_data = snapshot.status_data
        best_row = _best_validation_row(snapshot)
        current_f1 = status_data.get("eval_f1")
        best_f1 = (
            status_data.get("best_eval_f1")
            or best_row.get("dev_f1")
            or snapshot.dev_metrics.get("best_f1")
            or snapshot.results.get("best_validation_f1")
        )
        epoch = status_data.get("epoch")
        if epoch is None and snapshot.progress is not None and snapshot.spec.epochs:
            epoch = snapshot.progress.step / (snapshot.progress.total / snapshot.spec.epochs)
        step = status_data.get("step")
        if step is None and snapshot.progress is not None:
            step = snapshot.progress.step
        phase = str(status_data.get("phase") or (
            "complete" if snapshot.results else "planned" if snapshot.job.state == "PLANNED" else "training"
        ))
        role = "historical" if snapshot.spec.historical else "canonical"
        run_name = f"{snapshot.spec.label}"
        max_steps = status_data.get("max_steps") or (
            snapshot.progress.total if snapshot.progress is not None else snapshot.spec.total_steps
        )
        step_cell = f"{step if step is not None else '-'}/{max_steps or '-'}"
        progress = status_data.get("progress_percent")
        if progress is None and snapshot.progress is not None:
            progress = snapshot.progress.percent
        progress_cell = "-" if _finite_number(progress) is None else f"{float(progress):.1f}%"
        elapsed = _finite_number(status_data.get("elapsed_seconds"))
        eta = _finite_number(status_data.get("eta_seconds"))
        best_step = status_data.get("best_step", snapshot.run_metadata.get("best_step"))
        best_epoch = status_data.get("best_epoch", snapshot.run_metadata.get("best_epoch"))
        lines.append(
            f"| {run_name} | {snapshot.spec.kind} | "
            f"{'full fine-tune' if snapshot.spec.kind == 'encoder' else 'QLoRA'} | "
            f"{snapshot.spec.variant} | {snapshot.spec.max_epochs or snapshot.spec.epochs or '-'} | "
            f"{snapshot.spec.seed} | {role} | {phase} | {_seed_snapshot_status(snapshot, now)} | "
            f"{snapshot.job.job_id or '-'} | {status_data.get('slurm_array_id') or '-'} | "
            f"{status_data.get('node') or snapshot.job.location} | {_seed_gpu(snapshot)} | "
            f"{status_data.get('start_time') or snapshot.run_metadata.get('start_time') or '-'} | "
            f"{_format_clock_duration(elapsed) if elapsed is not None else snapshot.job.elapsed} | "
            f"{_format_metric(epoch, 2)} | {step_cell} | {progress_cell} | "
            f"{_format_metric(status_data.get('train_loss', snapshot.train_metrics.get('loss')))} | "
            f"{_format_metric(status_data.get('eval_loss'))} | "
            f"{_format_percent(status_data.get('eval_precision', best_row.get('dev_precision')))} | "
            f"{_format_percent(status_data.get('eval_recall', best_row.get('dev_recall')))} | "
            f"{_format_percent(current_f1)} | {_format_percent(best_f1)} | "
            f"{best_step if best_step is not None else '-'} | {_format_epoch(best_epoch)} | "
            f"{_format_metric(status_data.get('learning_rate', snapshot.train_metrics.get('learning_rate')), 7)} | "
            f"{_seed_peak_vram(status_data, snapshot.results)} | "
            f"{_format_clock_duration(eta) if eta is not None else '-'} | "
            f"`results/multinerd/{snapshot.spec.key}` | {_seed_update_cell(status_data, now)} | "
            f"{status_data.get('failure_reason') or snapshot.alert or '-'} |"
        )

    if final_results is not None:
        lines.extend([
            "",
            "## All Final Experiments",
            "",
            "| Experiment | Regime | Status | Job | Validation F1 | Test F1 | Test precision | Test recall | Mean latency | Result folder |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for result in final_results:
            latency = _finite_number(result.metrics.get("latency_ms_mean"))
            latency_cell = "-" if latency is None else f"{latency:.1f} ms"
            job_cell = str(result.job.job_id) if result.job else "-"
            validation_cell = (
                "N/A"
                if result.regime == "llm_zeroshot"
                else _format_percent(_validation_f1(result))
            )
            lines.append(
                f"| {result.experiment_name} | "
                f"{_regime_label(result.regime)} | "
                f"{result.status} | "
                f"{job_cell} | "
                f"{validation_cell} | "
                f"{_format_percent(result.metrics.get('test_f1'))} | "
                f"{_format_percent(result.metrics.get('test_precision'))} | "
                f"{_format_percent(result.metrics.get('test_recall'))} | "
                f"{latency_cell} | "
                f"`results/multinerd/{result.experiment_name}` |"
            )

        lines.extend([
            "",
            "## Inference Time Estimates",
            "",
            "| Experiment | Job | Status | Elapsed | Progress | Estimated inference time | Estimated remaining | Expected finish |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ])
        for result in final_results:
            elapsed, progress, total, remaining, finish = _inference_time_cells(
                result,
                now,
            )
            job_cell = str(result.job.job_id) if result.job else "-"
            lines.append(
                f"| {result.experiment_name} | {job_cell} | {result.status} | "
                f"{elapsed} | {progress} | {total} | {remaining} | {finish} |"
            )

    if crossner_dashboard is not None:
        lines.extend(render_crossner_markdown(crossner_dashboard, now))

    lines.extend([
        "",
        "## Recovery Checkpoints",
        "",
        "| Model | Latest checkpoint | Saved at |",
        "|---|---:|---|",
    ])
    for snapshot in decoder_snapshots:
        saved = (
            snapshot.checkpoint_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            if snapshot.checkpoint_time
            else "-"
        )
        checkpoint = (
            f"checkpoint-{snapshot.checkpoint_step}"
            if snapshot.checkpoint_step is not None
            else "-"
        )
        lines.append(f"| {snapshot.spec.label} | {checkpoint} | {saved} |")

    alerts = [f"**{s.spec.label}:** {s.alert}" for s in snapshots if s.alert]
    lines.extend([
        "",
        "## Health",
        "",
        *(alerts or ["All selected jobs have no active error signature."]),
        "",
        f"Final comparison job: `{summary_job.job_id if summary_job else '-'}` "
        f"({summary_job.state if summary_job else 'UNKNOWN'})",
        "",
        "_ETAs combine the measured step rate with remaining epoch validation and "
        "configured restart buffers. Scheduler queue delays can move the finish time._",
        "",
    ])
    return "\n".join(lines)


def _inline_markdown(value: str) -> str:
    rendered = html.escape(value.rstrip())
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    return rendered


def _markdown_body(markdown: str) -> str:
    """Render the small, controlled Markdown subset used by this dashboard."""
    lines = markdown.splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("# "):
            rendered.append(f"<h1>{_inline_markdown(line[2:])}</h1>")
            index += 1
            continue
        if line.startswith("## "):
            rendered.append(f"<h2>{_inline_markdown(line[3:])}</h2>")
            index += 1
            continue
        if (
            line.startswith("|")
            and index + 1 < len(lines)
            and re.fullmatch(r"\|[\s:|-]+\|", lines[index + 1])
        ):
            headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([
                    cell.strip()
                    for cell in lines[index].strip().strip("|").split("|")
                ])
                index += 1
            table = ["<div class=\"table-wrap\"><table><thead><tr>"]
            table.extend(f"<th>{_inline_markdown(cell)}</th>" for cell in headers)
            table.append("</tr></thead><tbody>")
            for row in rows:
                table.append("<tr>")
                table.extend(f"<td>{_inline_markdown(cell)}</td>" for cell in row)
                table.append("</tr>")
            table.append("</tbody></table></div>")
            rendered.append("".join(table))
            continue
        rendered.append(f"<p>{_inline_markdown(line)}</p>")
        index += 1
    return "\n".join(rendered)


def render_html(
    markdown: str,
    browser_refresh_seconds: int = DEFAULT_BROWSER_REFRESH_SECONDS,
) -> str:
    if browser_refresh_seconds <= 0:
        raise ValueError("browser_refresh_seconds must be positive")
    body = _markdown_body(markdown)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{browser_refresh_seconds}">
  <title>Training Live Monitor</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --border: #d8dee9;
      --accent: #2563eb;
      --stripe: #f8fafc;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f172a;
        --panel: #111827;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --border: #374151;
        --accent: #60a5fa;
        --stripe: #172033;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1440px, calc(100% - 32px));
      margin: 24px auto;
      padding: 28px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 12px 32px rgb(15 23 42 / 8%);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(1.65rem, 4vw, 2.25rem); }}
    h2 {{
      margin: 32px 0 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
      font-size: 1.2rem;
    }}
    p {{ margin: 5px 0; }}
    code {{
      padding: 2px 5px;
      border: 1px solid var(--border);
      border-radius: 5px;
      background: var(--stripe);
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{
      padding: 9px 12px;
      border: 1px solid var(--border);
      text-align: left;
    }}
    th {{ background: color-mix(in srgb, var(--accent) 12%, var(--panel)); }}
    tbody tr:nth-child(even) {{ background: var(--stripe); }}
    .reload {{
      position: fixed;
      right: 16px;
      bottom: 16px;
      padding: 7px 11px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      box-shadow: 0 4px 14px rgb(15 23 42 / 12%);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>{body}</main>
  <div class="reload" aria-live="polite">
    Page reload in <span id="reload-countdown">{browser_refresh_seconds}</span>s
  </div>
  <script>
    const refreshSeconds = {browser_refresh_seconds};
    const loadedAt = Date.now();
    const countdown = document.getElementById("reload-countdown");
    window.setInterval(() => {{
      const elapsed = Math.floor((Date.now() - loadedAt) / 1000);
      countdown.textContent = String(Math.max(refreshSeconds - elapsed, 0));
    }}, 1000);
  </script>
</body>
</html>
"""


def load_config(path: Path) -> MonitorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    refresh_seconds = int(raw.get("refresh_seconds", 300))
    if refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be positive")
    scheduler_refresh_seconds = int(raw.get("scheduler_refresh_seconds", 60))
    if scheduler_refresh_seconds <= 0:
        raise ValueError("scheduler_refresh_seconds must be positive")
    browser_refresh_seconds = int(
        raw.get("browser_refresh_seconds", DEFAULT_BROWSER_REFRESH_SECONDS)
    )
    if browser_refresh_seconds <= 0:
        raise ValueError("browser_refresh_seconds must be positive")
    specs = []
    for item in raw["models"]:
        specs.append(ModelSpec(
            key=item["key"],
            label=item["label"],
            kind=item["kind"],
            job_ids=tuple(int(value) for value in item["job_ids"]),
            total_steps=item.get("total_steps"),
            epochs=item.get("epochs"),
            eval_seconds_low=float(item.get("eval_seconds_low", 0)),
            eval_seconds_high=float(item.get("eval_seconds_high", 0)),
            restart_buffer_seconds=float(item.get("restart_buffer_seconds", 0)),
            variant=str(item.get("variant", "default")),
            seed=int(item.get("seed", 42)),
            canonical=bool(item.get("canonical", True)),
            historical=bool(item.get("historical", False)),
            max_epochs=int(item["max_epochs"]) if item.get("max_epochs") is not None else None,
        ))
    summary_job_id = raw.get("summary_job_id")
    raw_result_jobs = raw.get("result_jobs", {})
    result_job_ids = {
        str(experiment_name): tuple(int(job_id) for job_id in job_ids)
        for experiment_name, job_ids in raw_result_jobs.items()
    }
    raw_time_limits = raw.get("result_time_limits_hours", {})
    result_time_limits_seconds = {
        str(experiment_name): float(hours) * 3600.0
        for experiment_name, hours in raw_time_limits.items()
    }
    if any(seconds <= 0 for seconds in result_time_limits_seconds.values()):
        raise ValueError("result_time_limits_hours values must be positive")
    crossner_output_limit = None
    raw_crossner = raw.get("crossner_output_limit")
    if raw_crossner is not None:
        if not isinstance(raw_crossner, dict):
            raise ValueError("crossner_output_limit must be a mapping")
        limit = raw_crossner.get("max_new_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 256:
            raise ValueError("crossner_output_limit.max_new_tokens must be an integer greater than 256")
        for name in ("output_root", "baseline_root"):
            if not isinstance(raw_crossner.get(name), str) or not raw_crossner[name].strip():
                raise ValueError(f"crossner_output_limit.{name} must be a nonempty path")
        crossner_output_limit = CrossnerOutputLimitConfig(
            output_root=Path(raw_crossner["output_root"]),
            baseline_root=Path(raw_crossner["baseline_root"]),
            max_new_tokens=limit,
        )
        if crossner_output_limit.output_root == crossner_output_limit.baseline_root:
            raise ValueError("CrossNER output and baseline roots must differ")
    return MonitorConfig(
        specs=tuple(specs),
        refresh_seconds=refresh_seconds,
        scheduler_refresh_seconds=scheduler_refresh_seconds,
        browser_refresh_seconds=browser_refresh_seconds,
        summary_job_id=int(summary_job_id) if summary_job_id else None,
        result_job_ids=result_job_ids,
        result_time_limits_seconds=result_time_limits_seconds,
        crossner_output_limit=crossner_output_limit,
    )


def filter_specs(
    specs: Iterable[ModelSpec],
    *,
    model: Optional[str] = None,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
    canonical: bool = False,
    historical: bool = False,
    phase: Optional[str] = None,
    completed: bool = False,
    failed: bool = False,
    results_root: Optional[Path] = None,
) -> tuple[ModelSpec, ...]:
    """Apply non-mutating CLI filters to monitor entries."""
    selected = list(specs)
    if model:
        needle = model.lower().replace("_", "-")
        selected = [
            spec for spec in selected
            if needle in spec.key.lower().replace("_", "-")
            or needle in spec.label.lower().replace("_", "-")
        ]
    if variant:
        selected = [spec for spec in selected if spec.variant == variant]
    if seed is not None:
        selected = [spec for spec in selected if spec.seed == seed]
    if canonical:
        selected = [spec for spec in selected if spec.canonical and not spec.historical]
    if historical:
        selected = [spec for spec in selected if spec.historical]
    if phase or completed or failed:
        root = results_root or resolve_results_root()
        filtered = []
        for spec in selected:
            status = _load_json(root / spec.key / "status.json")
            status_name = str(status.get("status", "")).upper()
            status_phase = str(status.get("phase", "planned")).lower()
            has_results = (root / spec.key / "results.yaml").is_file()
            has_inference = (root / spec.key / "inference_metrics.yaml").is_file()
            if phase and status_phase != phase.lower():
                continue
            if completed and not (status_name == "COMPLETED" or (has_results and has_inference)):
                continue
            if failed and status_name != "FAILED":
                continue
            filtered.append(spec)
        selected = filtered
    return tuple(selected)


def merge_submission_registry(
    settings: MonitorConfig,
    results_root: Optional[Path] = None,
) -> MonitorConfig:
    """Attach real submitted job IDs without modifying the tracked YAML."""
    root = results_root or resolve_results_root()
    registry_path = root.parent / "seed_studies" / "multinerd" / "submission_registry.yaml"
    registry = _load_yaml(registry_path)
    jobs = registry.get("jobs", [])
    if not isinstance(jobs, list):
        return settings
    merged_specs = []
    for spec in settings.specs:
        job_ids = list(spec.job_ids)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            output_name = Path(str(job.get("output_dir", ""))).name
            job_id = job.get("job_id")
            if (
                output_name == spec.key
                and job.get("phase") == "training"
                and str(job_id or "").isdigit()
            ):
                numeric = int(job_id)
                if numeric not in job_ids:
                    job_ids.append(numeric)
        merged_specs.append(replace(spec, job_ids=tuple(job_ids)))
    return replace(settings, specs=tuple(merged_specs))


def resolve_results_root() -> Path:
    configured = os.environ.get("BA_NER_RESULTS_ROOT")
    if configured:
        return Path(configured) / "multinerd"
    scratch = Path(f"/netscratch/{os.environ.get('USER', 'losman')}/ba-ner/results/multinerd")
    return scratch if scratch.is_dir() else PROJECT_ROOT / "results" / "multinerd"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(0o644)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def update_dashboard(
    config_path: Path,
    output_path: Path,
    html_output_path: Optional[Path] = None,
    *,
    settings: Optional[MonitorConfig] = None,
    jobs: Optional[dict[int, JobState]] = None,
) -> str:
    settings = settings or load_config(config_path)
    if jobs is None:
        jobs = query_jobs(monitor_job_ids(settings))
    results_root = resolve_results_root()
    snapshots = [
        collect_snapshot(spec, jobs, results_root)
        for spec in settings.specs
    ]
    final_results = collect_final_results(
        results_root,
        settings.result_job_ids,
        settings.result_time_limits_seconds,
        jobs,
    )
    crossner_dashboard = None
    if settings.crossner_output_limit is not None:
        config = settings.crossner_output_limit
        try:
            crossner_dashboard = collect_crossner_dashboard(config, results_root, jobs)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            output_root, baseline_root = _crossner_roots(config, results_root)
            crossner_dashboard = CrossnerDashboard(
                config, output_root, baseline_root, (), (), False, (),
                f"CrossNER artifacts could not be read: {exc}",
            )
    now = datetime.now().astimezone()
    summary_job = (
        jobs.get(settings.summary_job_id)
        if settings.summary_job_id
        else None
    )
    rendered = render_markdown(
        snapshots,
        now,
        settings.refresh_seconds,
        summary_job,
        settings.browser_refresh_seconds,
        final_results,
        crossner_dashboard,
    )
    write_atomic(output_path, rendered)
    if html_output_path is not None:
        write_atomic(
            html_output_path,
            render_html(rendered, settings.browser_refresh_seconds),
        )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Auto-refreshing HTML output (defaults next to --output)",
    )
    parser.add_argument("--interval", "--refresh", dest="interval", type=int, help="Override refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    parser.add_argument("--stdout", action="store_true", help="Print each rendered snapshot")
    parser.add_argument("--model", help="Filter by model key or label")
    parser.add_argument("--variant", help="Filter by setup variant, e.g. 3ep")
    parser.add_argument("--seed", type=int, choices=(42, 123, 456))
    role = parser.add_mutually_exclusive_group()
    role.add_argument("--canonical", action="store_true")
    role.add_argument("--historical", action="store_true")
    parser.add_argument("--phase")
    outcome = parser.add_mutually_exclusive_group()
    outcome.add_argument("--completed", action="store_true")
    outcome.add_argument("--failed", action="store_true")
    parser.add_argument("--all", action="store_true", help="Show all configured runs (default)")
    args = parser.parse_args()

    settings = load_config(args.config)
    interval = args.interval or settings.refresh_seconds
    settings = replace(
        settings,
        refresh_seconds=interval,
        specs=filter_specs(
            settings.specs,
            model=args.model,
            variant=args.variant,
            seed=args.seed,
            canonical=args.canonical,
            historical=args.historical,
            phase=args.phase,
            completed=args.completed,
            failed=args.failed,
        ),
    )
    settings = merge_submission_registry(settings)
    html_output = args.html_output or args.output.with_suffix(".html")
    cached_jobs: Optional[dict[int, JobState]] = None
    last_scheduler_refresh: Optional[float] = None
    while True:
        try:
            monotonic_now = time.monotonic()
            if (
                cached_jobs is None
                or last_scheduler_refresh is None
                or monotonic_now - last_scheduler_refresh
                >= settings.scheduler_refresh_seconds
            ):
                cached_jobs = query_jobs(monitor_job_ids(settings))
                last_scheduler_refresh = monotonic_now
            rendered = update_dashboard(
                args.config,
                args.output,
                html_output,
                settings=settings,
                jobs=cached_jobs,
            )
            if args.stdout:
                print(rendered, flush=True)
            else:
                print(
                    f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}: "
                    f"updated {args.output} and {html_output}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}: "
                f"monitor update failed: {exc}",
                flush=True,
            )
        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
