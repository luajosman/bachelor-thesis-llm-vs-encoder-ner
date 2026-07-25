"""Generate a five-minute Markdown dashboard for the active training runs."""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training_monitor.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "training_monitor.md"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(
    r"(?P<percent>\d+)%\|[^\r\n]*?\|\s*"
    r"(?P<step>\d+)/(?P<total>\d+)\s*\[[^\]]*?"
    r"(?P<rate>\d+(?:\.\d+)?)(?P<unit>s/it|it/s)\]"
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
        state = __import__("json").loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    for row in reversed(state.get("log_history", [])):
        if isinstance(row, dict) and "loss" in row:
            return row
    return {}


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
    dev_metrics = _load_yaml(result_dir / "generative_eval_state.yaml")
    results = _load_yaml(result_dir / "results.yaml")

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


def _format_percent(value: Any) -> str:
    number = _finite_number(value)
    return "-" if number is None else f"{100.0 * number:.2f}%"


def _format_metric(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds <= 0:
        return "complete"
    hours = seconds / 3600.0
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


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
) -> str:
    decoder_snapshots = [s for s in snapshots if s.spec.kind == "decoder"]
    lines = [
        "# Training Live Monitor",
        "",
        f"**Updated:** {now:%Y-%m-%d %H:%M:%S %Z}  ",
        f"**Refresh:** every {refresh_seconds // 60} minutes  ",
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
                f"{_format_percent(snapshot.results.get('best_validation_f1'))} | "
                "- | - | - | - |"
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
            f"{best_epoch if best_epoch is not None else '-'} | "
            f"{_format_percent(best_row.get('parse_failure_rate'))} |"
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
        f"Final summary job: `{summary_job.job_id if summary_job else '-'}` "
        f"({summary_job.state if summary_job else 'UNKNOWN'})",
        "",
        "_ETAs combine the measured step rate with remaining epoch validation and "
        "configured restart buffers. Scheduler queue delays can move the finish time._",
        "",
    ])
    return "\n".join(lines)


def load_config(path: Path) -> tuple[list[ModelSpec], int, Optional[int]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    refresh_seconds = int(raw.get("refresh_seconds", 300))
    if refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be positive")
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
    return specs, refresh_seconds, int(summary_job_id) if summary_job_id else None


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
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def update_dashboard(config_path: Path, output_path: Path) -> str:
    specs, refresh_seconds, summary_job_id = load_config(config_path)
    all_job_ids = [job_id for spec in specs for job_id in spec.job_ids]
    if summary_job_id:
        all_job_ids.append(summary_job_id)
    jobs = query_jobs(all_job_ids)
    results_root = resolve_results_root()
    snapshots = [collect_snapshot(spec, jobs, results_root) for spec in specs]
    now = datetime.now().astimezone()
    summary_job = jobs.get(summary_job_id) if summary_job_id else None
    rendered = render_markdown(snapshots, now, refresh_seconds, summary_job)
    write_atomic(output_path, rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=int, help="Override refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    parser.add_argument("--stdout", action="store_true", help="Print each rendered snapshot")
    args = parser.parse_args()

    _, configured_interval, _ = load_config(args.config)
    interval = args.interval or configured_interval
    while True:
        try:
            rendered = update_dashboard(args.config, args.output)
            if args.stdout:
                print(rendered, flush=True)
            else:
                print(
                    f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}: "
                    f"updated {args.output}",
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
