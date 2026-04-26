"""Safe job execution for whitelisted ML pipeline commands."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket

from dashboard_backend.models.schemas import ActionName, Job, JobLogsResponse
from dashboard_backend.utils.paths import BA_NER_ROOT, LOGS_DIR, as_display_path, ensure_runtime_directories
from dashboard_backend.utils.yaml_json import read_text_file


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


ACTION_COMMANDS: dict[ActionName, list[str]] = {
    "run_all": [sys.executable, "scripts/run_all.py"],
    "encoder_only": [sys.executable, "scripts/run_all.py", "--encoder-only"],
    "zeroshot_only": [sys.executable, "scripts/run_all.py", "--zeroshot-only"],
    "lora_only": [sys.executable, "scripts/run_all.py", "--decoder-only", "--finetuned-only"],
    "deberta_base": [sys.executable, "scripts/run_all.py", "--model", "deberta_base"],
    "deberta_large": [sys.executable, "scripts/run_all.py", "--model", "deberta_large"],
    "qwen35_08b_zs": [sys.executable, "scripts/run_all.py", "--model", "qwen35_08b_zs"],
    "qwen35_08b_lora": [sys.executable, "scripts/run_all.py", "--model", "qwen35_08b"],
    "qwen35_4b_zs": [sys.executable, "scripts/run_all.py", "--model", "qwen35_4b_zs"],
    "qwen35_4b_lora": [sys.executable, "scripts/run_all.py", "--model", "qwen35_4b"],
    "qwen35_27b_zs": [sys.executable, "scripts/run_all.py", "--model", "qwen35_27b_zs"],
    "qwen35_27b_lora": [sys.executable, "scripts/run_all.py", "--model", "qwen35_27b"],
    "compare_only": [sys.executable, "scripts/run_all.py", "--eval-only"],
}


@dataclass
class _RuntimeState:
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None


class JobService:
    """In-memory job registry with async subprocess execution."""

    def __init__(self) -> None:
        ensure_runtime_directories()
        self._jobs: dict[str, Job] = {}
        self._runtime: dict[str, _RuntimeState] = {}

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.started_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def create_job(self, action: ActionName) -> Job:
        if action not in ACTION_COMMANDS:
            raise ValueError(f"Unsupported action: {action}")

        job_id = uuid.uuid4().hex[:12]
        log_path = LOGS_DIR / f"{job_id}_{action}.log"
        command = ACTION_COMMANDS[action]
        job = Job(
            job_id=job_id,
            action=action,
            status="queued",
            command_label=" ".join(command[1:]),
            log_path=as_display_path(log_path),
        )
        self._jobs[job_id] = job
        runtime = _RuntimeState()
        runtime.task = asyncio.create_task(self._run_job(job_id, command, log_path))
        self._runtime[job_id] = runtime
        return job

    async def _run_job(self, job_id: str, command: list[str], log_path: Path) -> None:
        job = self._jobs[job_id]
        job.status = "running"
        job.started_at = _utcnow()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(command)}\n")
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(BA_NER_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._runtime[job_id].process = process
            assert process.stdout is not None
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                log_file.write(line.decode("utf-8", errors="replace"))
                log_file.flush()
            exit_code = await process.wait()
            job.exit_code = exit_code
            if job.status != "cancelled":
                job.status = "completed" if exit_code == 0 else "failed"

        job.finished_at = _utcnow()
        if job.started_at is not None:
            job.duration = (job.finished_at - job.started_at).total_seconds()

    async def cancel_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        runtime = self._runtime.get(job_id)
        if job is None or runtime is None or runtime.process is None:
            return job

        if job.status not in {"queued", "running"}:
            return job

        job.status = "cancelled"
        runtime.process.terminate()
        with contextlib.suppress(ProcessLookupError):
            await runtime.process.wait()
        job.finished_at = _utcnow()
        if job.started_at is not None:
            job.duration = (job.finished_at - job.started_at).total_seconds()
        return job

    def get_logs(self, job_id: str) -> JobLogsResponse | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        log_path = BA_NER_ROOT / job.log_path
        return JobLogsResponse(job=job, log=read_text_file(log_path))

    async def stream_logs(self, websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        last_payload = ""
        try:
            while True:
                payload = self.get_logs(job_id)
                if payload is None:
                    await websocket.send_json({"job_id": job_id, "status": "missing", "log": ""})
                    break
                message = payload.model_dump(mode="json")
                serialized = str(message)
                if serialized != last_payload:
                    await websocket.send_json(message)
                    last_payload = serialized
                if payload.job.status in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(1.0)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()


job_service = JobService()
