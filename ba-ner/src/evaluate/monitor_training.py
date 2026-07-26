"""Generate a five-minute Markdown dashboard for the active training runs."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from src.config import FINAL_EXPERIMENTS


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


@dataclass(frozen=True)
class MonitorConfig:
    specs: tuple[ModelSpec, ...]
    refresh_seconds: int
    scheduler_refresh_seconds: int
    browser_refresh_seconds: int
    summary_job_id: Optional[int]
    result_job_ids: dict[str, tuple[int, ...]]
    result_time_limits_seconds: dict[str, float]


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
        return JobState(spec.job_ids[-1], "UNKNOWN")
    priority = {"RUNNING": 0, "COMPLETING": 1, "CONFIGURING": 2, "PENDING": 3}
    return min(
        candidates,
        key=lambda job: (
            priority.get(job.state, 10),
            -spec.job_ids.index(job.job_id),
        ),
    )


def _find_log(job_id: int, suffix: str) -> Optional[Path]:
    matches = sorted((PROJECT_ROOT / "logs").glob(f"*_{job_id}.{suffix}"))
    return matches[-1] if matches else None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
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
    if snapshot.results and snapshot.spec.kind == "encoder":
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
                f"≤ {_format_compact_duration(limit)} (limit)",
                f"≤ {_format_compact_duration(remaining_budget)} (limit)",
                finish,
            )
        return (
            elapsed_cell,
            "not started",
            f"≤ {_format_compact_duration(limit)} (limit)",
            f"≤ {_format_compact_duration(limit)} after start",
            "after scheduling",
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
    if snapshot.results and snapshot.spec.kind == "encoder":
        return "100% (complete)"
    if snapshot.progress is None:
        return "-"
    return (
        f"{snapshot.progress.percent:.1f}% "
        f"({snapshot.progress.step:,}/{snapshot.progress.total:,})"
    )


def _epoch_cell(snapshot: ModelSnapshot) -> str:
    if snapshot.spec.kind == "encoder":
        epochs = snapshot.results.get("num_train_epochs")
        return f"{epochs}/{epochs}" if epochs else "-"
    if snapshot.progress is None or not snapshot.spec.epochs:
        return "-"
    epoch = min(
        snapshot.spec.epochs,
        snapshot.progress.step / (snapshot.progress.total / snapshot.spec.epochs),
    )
    return f"{epoch:.2f}/{snapshot.spec.epochs}"


def render_markdown(
    snapshots: list[ModelSnapshot],
    now: datetime,
    refresh_seconds: int,
    summary_job: Optional[JobState],
    browser_refresh_seconds: int = DEFAULT_BROWSER_REFRESH_SECONDS,
    final_results: Optional[list[FinalResultSnapshot]] = None,
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

    active_lows = [
        s.eta_low_seconds
        for s in decoder_snapshots
        if s.eta_low_seconds is not None
    ]
    active_highs = [
        s.eta_high_seconds
        for s in decoder_snapshots
        if s.eta_high_seconds is not None
    ]
    overall_low = max(active_lows) if active_lows else None
    overall_high = max(active_highs) if active_highs else None
    lines.append(
        "| **All models** | Parallel | - | - | "
        f"**{_format_eta_range(overall_low, overall_high)}** | "
        f"**{_format_finish(now, overall_low, overall_high)}** |"
    )
    lines.extend([
        "",
        f"**Estimated time until all models finish:** "
        f"{_format_eta_range(overall_low, overall_high)}",
        "",
        "## Live Training Metrics",
        "",
        "| Model | Job | Node | Runtime | Step time | Loss | Token accuracy | Learning rate | Grad norm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for snapshot in decoder_snapshots:
        rate = (
            f"{snapshot.progress.seconds_per_step:.2f} s"
            if snapshot.progress is not None
            else "-"
        )
        lines.append(
            "| {label} | {job} | {node} | {runtime} | {rate} | {loss} | "
            "{accuracy} | {lr} | {grad} |".format(
                label=snapshot.spec.label,
                job=snapshot.job.job_id,
                node=snapshot.job.location,
                runtime=snapshot.job.elapsed,
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
    return MonitorConfig(
        specs=tuple(specs),
        refresh_seconds=refresh_seconds,
        scheduler_refresh_seconds=scheduler_refresh_seconds,
        browser_refresh_seconds=browser_refresh_seconds,
        summary_job_id=int(summary_job_id) if summary_job_id else None,
        result_job_ids=result_job_ids,
        result_time_limits_seconds=result_time_limits_seconds,
    )


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
        all_job_ids = [
            job_id
            for spec in settings.specs
            for job_id in spec.job_ids
        ]
        if settings.summary_job_id:
            all_job_ids.append(settings.summary_job_id)
        for result_jobs in settings.result_job_ids.values():
            all_job_ids.extend(result_jobs)
        jobs = query_jobs(all_job_ids)
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
    parser.add_argument("--interval", type=int, help="Override refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    parser.add_argument("--stdout", action="store_true", help="Print each rendered snapshot")
    args = parser.parse_args()

    settings = load_config(args.config)
    interval = args.interval or settings.refresh_seconds
    settings = replace(settings, refresh_seconds=interval)
    html_output = args.html_output or args.output.with_suffix(".html")
    cached_jobs: Optional[dict[int, JobState]] = None
    last_scheduler_refresh: Optional[float] = None
    all_job_ids = [
        job_id
        for spec in settings.specs
        for job_id in spec.job_ids
    ]
    if settings.summary_job_id:
        all_job_ids.append(settings.summary_job_id)
    for result_jobs in settings.result_job_ids.values():
        all_job_ids.extend(result_jobs)
    while True:
        try:
            monotonic_now = time.monotonic()
            if (
                cached_jobs is None
                or last_scheduler_refresh is None
                or monotonic_now - last_scheduler_refresh
                >= settings.scheduler_refresh_seconds
            ):
                cached_jobs = query_jobs(all_job_ids)
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
