#!/usr/bin/env python3
"""Plan, validate, submit, and inspect the fifteen-run canonical seed matrix."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config  # noqa: E402
from src.evaluate.aggregate_seeds import aggregate_seed_study, resolve_results_dir  # noqa: E402
from src.seed_study import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    EXPECTED_NEW_TRAINING_RUNS,
    ConfigDifference,
    RunDescriptor,
    SeedStudyError,
    atomic_write_yaml,
    full_run_config_hash,
    inspect_output_path,
    iter_run_descriptors,
    load_manifest,
    recursive_config_diff,
    resolve_run_output,
    scientific_config_hash,
    validate_manifest,
    validate_seed_equivalence,
)
from src.seed_provenance import (  # noqa: E402
    reference_provenance_issues,
    scientific_contract,
    verify_cached_revisions,
)


MODEL_ALIASES = {
    "deberta_base": "deberta-v3-base",
    "deberta_large": "deberta-v3-large",
    "qwen35_08b": "qwen35-08b-qlora",
    "qwen35_4b": "qwen35-4b-qlora",
    "qwen35_27b": "qwen35-27b-qlora-3ep",
    "qwen35_27b_3ep": "qwen35-27b-qlora-3ep",
}


def select_new_runs(
    *,
    models: Sequence[str] = (),
    seeds: Sequence[int] = (),
    config_paths: Sequence[str] = (),
    encoder_only: bool = False,
    decoder_only: bool = False,
    deberta_only: bool = False,
    qwen_only: bool = False,
    variant: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> list[RunDescriptor]:
    descriptors = [run for run in iter_run_descriptors(manifest) if run.is_new and run.canonical]
    normalized_models = {
        MODEL_ALIASES.get(model, model.replace("_", "-"))
        for model in models
    }
    if normalized_models:
        descriptors = [run for run in descriptors if run.group_key in normalized_models]
    if config_paths:
        normalized_configs = {
            Path(value).as_posix().removeprefix("./")
            for value in config_paths
        }
        descriptors = [
            run for run in descriptors
            if Path(run.config_path).as_posix() in normalized_configs
        ]
    if seeds:
        descriptors = [run for run in descriptors if run.seed in set(seeds)]
    if encoder_only or deberta_only:
        descriptors = [run for run in descriptors if run.model_family == "encoder"]
    if decoder_only or qwen_only:
        descriptors = [run for run in descriptors if run.model_family == "decoder"]
    if variant:
        descriptors = [run for run in descriptors if run.variant == variant]
    return descriptors


def build_plan(runs: Iterable[RunDescriptor]) -> list[Dict[str, Any]]:
    manifest = load_manifest()
    groups = {str(group["key"]): group for group in manifest["groups"]}
    plan: list[Dict[str, Any]] = []
    for run in runs:
        group = groups[run.group_key]
        cfg = load_experiment_config(PROJECT_ROOT / run.config_path)
        if group.get("derivation_config") and run.seed == 42:
            reference_path = str(group["derivation_config"])
            allowed = {"num_train_epochs"} if run.group_key == "qwen35-27b-qlora-3ep" else set()
        elif group.get("derivation_config"):
            reference_path = str(group["reference_config"])
            allowed = set()
        else:
            reference_path = str(group["reference_config"])
            allowed = set()
        reference = load_experiment_config(PROJECT_ROOT / reference_path)
        differences = validate_seed_equivalence(
            reference,
            cfg,
            additionally_allowed_fields=allowed,
        )
        resolved_output = resolve_run_output(cfg)
        contract = scientific_contract(cfg)
        phase_commands = {
            phase: command_preview(run, phase)
            for phase in ("training", "inference", "evaluation")
        }
        plan.append({
            "group": run.group_key,
            "model": run.label,
            "model_name": run.model_name,
            "model_family": run.model_family,
            "regime": run.regime,
            "variant": run.variant,
            "seed": run.seed,
            "max_epochs": run.max_epochs,
            "canonical": run.canonical,
            "reference_config": reference_path,
            "reference_output_dir": _reference_output(group, run.seed),
            "reference_status": _reference_status(group, run.seed),
            "planned_config": run.config_path,
            "planned_output_dir": run.output_dir,
            "resolved_output_dir": str(resolved_output),
            "planned_log_dir": str(_log_dir(run)),
            "scientific_config_hash": scientific_config_hash(cfg),
            "full_run_config_hash": full_run_config_hash(cfg),
            "scientific_code_hash": contract["scientific_code_hash"],
            "prompt_hash": contract["prompt"].get("prompt_sha256"),
            "parser_hash": contract["prompt"].get("parser_sha256"),
            "model_revision": contract["model_revision"],
            "dataset_revision": contract["dataset_revision"],
            "reference_provenance": group.get("reference_provenance", {}),
            "config_diff": [asdict(difference) for difference in differences],
            "validation": "PASS",
            "output_status": inspect_output_path(resolved_output),
            "resources": run.resources,
            "commands": phase_commands,
            "dependencies": "training -> afterok -> inference -> afterok -> evaluation -> afterany -> partial-aware aggregation",
            "planned_status": (
                "PLANNED_CANONICAL" if run.status == "planned_canonical" else "PLANNED"
            ),
        })
    return plan


def preflight(
    runs: Sequence[RunDescriptor],
    *,
    phases: Sequence[str] = ("training", "inference", "evaluation"),
    run_tests: bool = True,
    allow_resume_failed: bool = False,
) -> Dict[str, Any]:
    manifest = load_manifest()
    manifest_report = validate_manifest()
    plan = build_plan(runs)
    errors: list[str] = []
    warnings: list[str] = []

    selected_groups = {run.group_key for run in runs}
    selected_manifest = {
        "groups": [
            group for group in manifest["groups"]
            if str(group["key"]) in selected_groups
        ]
    }
    errors.extend(
        "Unverified historical reference provenance: " + issue
        for issue in reference_provenance_issues(selected_manifest)
    )

    required_packages = (
        "torch", "transformers", "datasets", "accelerate", "peft", "trl",
        "bitsandbytes", "seqeval", "yaml", "rich", "numpy",
    )
    missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
    if missing_packages:
        errors.append("Missing packages: " + ", ".join(missing_packages))
    required_files = (
        "src/encoder/train.py",
        "src/encoder/inference.py",
        "src/decoder/train.py",
        "src/decoder/inference.py",
        "src/evaluate/validate_seed_run.py",
        "src/evaluate/aggregate_seeds.py",
        "scripts/cluster/job_encoder_train.sh",
        "scripts/cluster/job_encoder_infer.sh",
        "scripts/cluster/job_decoder_lora_train.sh",
        "scripts/cluster/job_decoder_lora_infer.sh",
        "scripts/cluster/job_seed_evaluate.sh",
        "scripts/cluster/job_seed_aggregate.sh",
    )
    for filename in required_files:
        if not (PROJECT_ROOT / filename).is_file():
            errors.append(f"Missing required file: {filename}")
    errors.extend(_training_contract_issues(runs))

    planned_outputs = [item["resolved_output_dir"] for item in plan]
    historical_outputs = {
        resolve_run_output(load_experiment_config(PROJECT_ROOT / run.config_path)).resolve()
        for run in iter_run_descriptors()
        if run.historical
    }
    if len(planned_outputs) != len(set(planned_outputs)):
        errors.append("Planned output paths collide")
    planned_logs = [item["planned_log_dir"] for item in plan]
    if len(planned_logs) != len(set(planned_logs)):
        errors.append("Planned log paths collide")

    for item in plan:
        output_status = item["output_status"]
        if "training" in phases and output_status != "FREE":
            if output_status in {"COMPLETED", "RUNNING"}:
                warnings.append(
                    f"{item['group']} seed {item['seed']}: {output_status}; the run will be skipped"
                )
            elif allow_resume_failed and output_status in {"FAILED", "INCOMPLETE_WITH_CHECKPOINTS"}:
                warnings.append(
                    f"{item['group']} seed {item['seed']}: explicit same-run resume/restart requested"
                )
            else:
                errors.append(
                    f"{item['group']} seed {item['seed']}: training output is not free ({output_status})"
                )
        if "training" not in phases and output_status == "FREE":
            errors.append(
                f"{item['group']} seed {item['seed']}: output is missing for phases {list(phases)}"
            )
        if Path(item["resolved_output_dir"]).resolve() in historical_outputs:
            errors.append(
                f"{item['group']} seed {item['seed']}: a new run resolves to a historical path"
            )

    if len([run for run in iter_run_descriptors() if run.is_new]) != EXPECTED_NEW_TRAINING_RUNS:
        errors.append(
            f"The complete manifest does not contain exactly "
            f"{EXPECTED_NEW_TRAINING_RUNS} new runs"
        )
    if manifest.get("zero_shot", {}).get("repeated_for_seeds") is not False:
        errors.append("Zero-shot runs are incorrectly included in the seed matrix")

    hf_home = Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None
    if hf_home is None:
        scratch = _scratch_root()
        hf_home = scratch / "hf-cache" if scratch is not None else PROJECT_ROOT / ".hf_cache"
    cache_errors, cache_warnings = verify_cached_revisions(
        hf_home=hf_home,
        model_names={run.model_name for run in runs},
    )
    errors.extend(cache_errors)
    warnings.extend(cache_warnings)

    scheduler_conflicts, scheduler_warnings = _scheduler_conflicts(runs, phases)
    errors.extend(scheduler_conflicts)
    warnings.extend(scheduler_warnings)
    resource_errors, resource_warnings = _scheduler_resource_issues(runs, phases)
    errors.extend(resource_errors)
    warnings.extend(resource_warnings)

    cuda_visible = False
    try:
        import torch
        cuda_visible = bool(torch.cuda.is_available())
    except Exception:
        pass
    if not cuda_visible:
        warnings.append("CUDA is not visible on the login node; compute-node preflight is required")

    disk_root = resolve_results_dir()
    disk_probe = disk_root if disk_root.exists() else disk_root.parent
    try:
        disk = shutil.disk_usage(disk_probe)
        disk_free_bytes = disk.free
        minimum_free_bytes = int(manifest.get("minimum_free_bytes", 0))
        if disk.free < minimum_free_bytes:
            errors.append(
                f"Insufficient free space at {disk_probe}: {disk.free} bytes available, "
                f"{minimum_free_bytes} required"
            )
    except OSError:
        disk_free_bytes = None
        warnings.append(f"Could not inspect disk space at {disk_probe}")

    write_probe_root = disk_root if disk_root.exists() else disk_root.parent
    try:
        with tempfile.NamedTemporaryFile(
            dir=write_probe_root,
            prefix=".seed-study-write-probe-",
            delete=True,
        ) as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        storage_writable = True
    except OSError as exc:
        storage_writable = False
        errors.append(f"Results storage is not writable at {write_probe_root}: {exc}")

    tests = {"ran": False, "returncode": None, "output": None}
    if run_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        tests = {
            "ran": True,
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip(),
        }
        if result.returncode != 0:
            errors.append("Unit tests failed")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest": {
            "valid": manifest_report["valid"],
            "new_run_count": manifest_report["new_run_count"],
        },
        "selected_run_count": len(runs),
        "planned_training_run_count": len(runs) if "training" in phases else 0,
        "planned_inference_run_count": len(runs) if "inference" in phases else 0,
        "planned_evaluation_run_count": len(runs) if "evaluation" in phases else 0,
        "zero_shot_duplicates": 0,
        "cuda_visible_on_current_host": cuda_visible,
        "compute_node_cuda_preflight_required": True,
        "disk_free_bytes": disk_free_bytes,
        "minimum_free_bytes": int(manifest.get("minimum_free_bytes", 0)),
        "results_storage_writable": storage_writable,
        "hub_cache_root": str(hf_home),
        "tests": tests,
        "plan": plan,
    }


def _scheduler_conflicts(
    runs: Sequence[RunDescriptor],
    phases: Sequence[str],
) -> tuple[list[str], list[str]]:
    if shutil.which("squeue") is None:
        return [], ["squeue is unavailable; active-job collision checks require the cluster login node"]
    expected_names = {
        _job_name(run, phase)
        for run in runs
        for phase in phases
        if phase in {"training", "inference", "evaluation"}
    }
    try:
        command = ["squeue", "-h", "-o", "%i|%j|%T"]
        user = os.environ.get("USER")
        if user:
            command[1:1] = ["-u", user]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [f"Could not inspect active SLURM jobs: {exc}"]
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return [], [f"Could not inspect active SLURM jobs: {detail}"]
    conflicts: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 2)
        if len(fields) == 3 and fields[1] in expected_names:
            conflicts.append(
                f"Active SLURM job conflicts with planned phase: "
                f"job_id={fields[0]}, job_name={fields[1]}, state={fields[2]}"
            )
    return conflicts, []


def _training_contract_issues(runs: Sequence[RunDescriptor]) -> list[str]:
    issues: list[str] = []
    families = {run.model_family for run in runs}
    requirements = {
        "encoder": {
            "src/encoder/train.py": (
                "set_seed(seed)",
                "data_seed=seed",
                "metric_for_best_model=cfg.get(\"metric_for_best_model\", \"f1\")",
                "load_best_model_at_end=cfg.get(\"load_best_model_at_end\", True)",
            ),
            "src/encoder/inference.py": ("best validation checkpoint",),
        },
        "decoder": {
            "src/decoder/train.py": (
                "set_seed(seed)",
                "data_seed=seed",
                "GenerativeDevEvalCallback",
                "best_lora_adapter",
            ),
            "src/decoder/inference.py": (
                "highest_generative_validation_f1",
                "do_sample=False",
            ),
        },
    }
    for family in sorted(families):
        for filename, fragments in requirements[family].items():
            try:
                source = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
            except OSError as exc:
                issues.append(f"Could not verify training contract in {filename}: {exc}")
                continue
            missing = [fragment for fragment in fragments if fragment not in source]
            if missing:
                issues.append(
                    f"{family} training/checkpoint contract is not verifiable in {filename}; "
                    f"missing markers: {missing}"
                )
    return issues


def _scheduler_resource_issues(
    runs: Sequence[RunDescriptor],
    phases: Sequence[str],
) -> tuple[list[str], list[str]]:
    if not any(phase in {"training", "inference"} for phase in phases):
        return [], []
    if shutil.which("sinfo") is None:
        return [], ["sinfo is unavailable; SLURM partition/GRES checks require the cluster login node"]
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P|%G|%m"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [f"Could not inspect SLURM resources: {exc}"]
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return [], [f"Could not inspect SLURM resources: {detail}"]
    available = {
        fields[0].rstrip("*"): fields[1:]
        for line in result.stdout.splitlines()
        if len(fields := line.split("|", 2)) == 3
    }
    errors: list[str] = []
    for run in runs:
        for resource_key in ("train", "inference"):
            if ("training" if resource_key == "train" else "inference") not in phases:
                continue
            resources = run.resources.get(resource_key, {})
            partition = str(resources.get("partition", ""))
            if partition not in available:
                errors.append(
                    f"{run.group_key} {resource_key}: SLURM partition {partition!r} is unavailable"
                )
    return errors, []


def command_preview(
    run: RunDescriptor,
    phase: str,
    dependency: Optional[str] = None,
    *,
    allow_resume_failed: bool = False,
) -> list[str]:
    resources = dict(run.resources.get("train" if phase == "training" else "inference", {}))
    job_name = _job_name(run, phase)
    command = ["sbatch", "--parsable", f"--job-name={job_name}"]
    if phase in {"training", "inference"}:
        command.extend([
            f"--partition={resources['partition']}",
            f"--gres={resources['gres']}",
            f"--mem={resources['memory']}",
            f"--time={resources['time']}",
        ])
    else:
        command.extend(["--partition=batch", "--mem=8G", "--time=00:30:00"])
    command.extend([
        f"--output={_log_dir(run)}/{phase}-%j.out",
        f"--error={_log_dir(run)}/{phase}-%j.err",
    ])
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    export_values = _base_exports()
    if phase == "training" and resources.get("requeue"):
        command.extend(["--requeue", "--signal=B:USR1@300"])
        export_values.extend(["BA_NER_REQUEUE_ON_SIGNAL=1", "BA_NER_ALLOW_RESUME=1"])
    if phase == "training" and allow_resume_failed:
        export_values.extend(["BA_NER_ALLOW_RESUME=1", "BA_NER_ALLOW_RESTART=1"])
    command.append("--export=" + ",".join(export_values))

    cfg = run.config_path
    output = resolve_run_output(load_experiment_config(PROJECT_ROOT / cfg))
    if phase == "training":
        script = (
            "scripts/cluster/job_encoder_train.sh"
            if run.model_family == "encoder"
            else "scripts/cluster/job_decoder_lora_train.sh"
        )
        command.extend([script, cfg])
    elif phase == "inference":
        if run.model_family == "encoder":
            command.extend([
                "scripts/cluster/job_encoder_infer.sh",
                cfg,
                str(output / "best_model"),
            ])
        else:
            command.extend([
                "scripts/cluster/job_decoder_lora_infer.sh",
                cfg,
                str(output / "best_lora_adapter"),
            ])
    elif phase == "evaluation":
        command.extend(["scripts/cluster/job_seed_evaluate.sh", cfg])
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return command


def submit_pipelines(
    runs: Sequence[RunDescriptor],
    *,
    phases: Sequence[str],
    skip_cluster_preflight: bool = False,
    allow_resume_failed: bool = False,
) -> Dict[str, Any]:
    registry_path = resolve_results_dir() / "seed_studies" / "multinerd" / "submission_registry.yaml"
    submitted: list[Dict[str, Any]] = []
    registry: Dict[str, Any] = {
        "submission_started_at": datetime.now(timezone.utc).isoformat(),
        "submission_updated_at": datetime.now(timezone.utc).isoformat(),
        "submission_status": "INITIALIZING",
        "preflight_job_id": None,
        "aggregate_job_id": None,
        "jobs": submitted,
    }

    def persist(**updates: Any) -> None:
        registry.update(updates)
        registry["submission_updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_yaml(registry_path, registry)

    persist()
    try:
        if not skip_cluster_preflight:
            preflight_job = submit_compute_preflight_and_wait(
                runs,
                on_submitted=lambda job_id: persist(
                    preflight_job_id=job_id,
                    submission_status="COMPUTE_PREFLIGHT_RUNNING",
                )
            )
            persist(
                preflight_job_id=preflight_job,
                submission_status="COMPUTE_PREFLIGHT_COMPLETED",
            )
        else:
            preflight_job = None
            persist(submission_status="SUBMITTING", compute_preflight_skipped=True)

        evaluation_jobs: list[str] = []
        for run in runs:
            current_output = resolve_run_output(load_experiment_config(PROJECT_ROOT / run.config_path))
            current_status = inspect_output_path(current_output)
            if "training" in phases and current_status in {"COMPLETED", "RUNNING"}:
                submitted.append({
                    "group": run.group_key,
                    "variant": run.variant,
                    "seed": run.seed,
                    "phase": "pipeline",
                    "job_id": None,
                    "job_name": None,
                    "dependency": None,
                    "output_dir": str(current_output),
                    "log_dir": str(_log_dir(run)),
                    "command": None,
                    "status": f"SKIPPED_{current_status}",
                })
                persist(submission_status="SUBMITTING")
                continue
            _log_dir(run).mkdir(parents=True, exist_ok=True)
            dependency: Optional[str] = None
            jobs: Dict[str, str] = {}
            for phase in ("training", "inference", "evaluation"):
                if phase not in phases:
                    continue
                command = command_preview(
                    run,
                    phase,
                    dependency,
                    allow_resume_failed=allow_resume_failed,
                )
                job_id = _submit(command)
                jobs[phase] = job_id
                submitted.append({
                    "group": run.group_key,
                    "variant": run.variant,
                    "seed": run.seed,
                    "phase": phase,
                    "job_id": job_id,
                    "job_name": _job_name(run, phase),
                    "dependency": dependency,
                    "output_dir": str(resolve_run_output(load_experiment_config(PROJECT_ROOT / run.config_path))),
                    "log_dir": str(_log_dir(run)),
                    "command": command,
                    "status": "SUBMITTED",
                })
                dependency = job_id
                persist(submission_status="SUBMITTING")
            if "evaluation" in jobs:
                evaluation_jobs.append(jobs["evaluation"])

        aggregate_job: Optional[str] = None
        if "evaluation" in phases and evaluation_jobs:
            aggregate_log = PROJECT_ROOT / "logs" / "seed-studies" / "aggregate"
            aggregate_log.mkdir(parents=True, exist_ok=True)
            command = [
                "sbatch", "--parsable", "--job-name=ner-seeds-aggregate",
                "--partition=batch", "--mem=8G", "--time=00:30:00",
                f"--output={aggregate_log}/aggregate-%j.out",
                f"--error={aggregate_log}/aggregate-%j.err",
                f"--dependency=afterany:{':'.join(evaluation_jobs)}",
                "--export=" + ",".join(_base_exports()),
                "scripts/cluster/job_seed_aggregate.sh",
            ]
            aggregate_job = _submit(command)
            submitted.append({
                "group": "all", "variant": "all", "seed": None,
                "phase": "aggregation", "job_id": aggregate_job,
                "job_name": "ner-seeds-aggregate",
                "dependency": f"afterany:{':'.join(evaluation_jobs)}",
                "output_dir": str(resolve_results_dir() / "seed_studies" / "multinerd"),
                "log_dir": str(aggregate_log), "command": command,
                "status": "SUBMITTED",
            })
            persist(aggregate_job_id=aggregate_job, submission_status="SUBMITTING")

        persist(
            submitted_at=datetime.now(timezone.utc).isoformat(),
            preflight_job_id=preflight_job,
            aggregate_job_id=aggregate_job,
            submission_status="SUBMITTED",
        )
    except BaseException as exc:
        persist(
            submission_status="PARTIAL_SUBMISSION_FAILED",
            submission_error=f"{type(exc).__name__}: {exc}",
        )
        raise

    registry["registry_path"] = str(registry_path)
    return registry


def submit_compute_preflight_and_wait(
    runs: Sequence[RunDescriptor],
    timeout_seconds: int = 1800,
    *,
    on_submitted: Optional[Callable[[str], None]] = None,
) -> str:
    log_dir = PROJECT_ROOT / "logs" / "seed-studies" / "preflight"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = _compute_preflight_command(runs, log_dir=log_dir)
    job_id = _submit(command)
    if on_submitted is not None:
        on_submitted(job_id)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        states = [line.split("|", 1)[0].split()[0].rstrip("+") for line in result.stdout.splitlines() if line.strip()]
        if any(state == "COMPLETED" for state in states):
            return job_id
        failed = next((state for state in states if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}), None)
        if failed:
            raise SeedStudyError(
                f"Compute-node preflight job {job_id} ended as {failed}; no training jobs were submitted"
            )
        time.sleep(10)
    raise SeedStudyError(
        f"Compute-node preflight job {job_id} did not finish within {timeout_seconds}s; no training jobs were submitted"
    )


def _compute_preflight_command(
    runs: Sequence[RunDescriptor],
    *,
    log_dir: Optional[Path] = None,
) -> list[str]:
    if not runs:
        raise SeedStudyError("Compute-node preflight requires at least one selected run")
    log_dir = log_dir or PROJECT_ROOT / "logs" / "seed-studies" / "preflight"
    selector_args = [
        value
        for run in runs
        for value in ("--config", run.config_path)
    ]
    return [
        "sbatch", "--parsable", "--job-name=ner-seeds-preflight",
        "--partition=H100", "--gres=gpu:1", "--mem=16G", "--time=00:20:00",
        f"--output={log_dir}/preflight-%j.out",
        f"--error={log_dir}/preflight-%j.err",
        "--export=" + ",".join([*_base_exports(), "REQUIRE_CUDA=1", "BA_NER_SEED_PREFLIGHT=1"]),
        "scripts/cluster/preflight.sh",
        *selector_args,
    ]


def status_report() -> list[Dict[str, Any]]:
    registry_path = resolve_results_dir() / "seed_studies" / "multinerd" / "submission_registry.yaml"
    registry = {}
    if registry_path.is_file():
        try:
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            registry = {}
    submitted_by_output: Dict[str, list[Mapping[str, Any]]] = {}
    for job in registry.get("jobs", []) if isinstance(registry, dict) else []:
        if isinstance(job, dict):
            submitted_by_output.setdefault(str(job.get("output_dir")), []).append(job)
    rows: list[Dict[str, Any]] = []
    for run in iter_run_descriptors():
        cfg = load_experiment_config(PROJECT_ROOT / run.config_path)
        output = resolve_run_output(cfg)
        status_file = output / "status.json"
        status_data: Dict[str, Any] = {}
        if status_file.is_file():
            try:
                status_data = __import__("json").loads(status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status_data = {"status": "STALE_OR_CORRUPT"}
        submitted_jobs = submitted_by_output.get(str(output), [])
        latest_job = submitted_jobs[-1] if submitted_jobs else {}
        job_id = status_data.get("slurm_job_id") or latest_job.get("job_id") or run.inference_job_id or run.training_job_id
        scheduler_state = _scheduler_state(str(job_id)) if str(job_id or "").isdigit() else None
        rows.append({
            "model": run.label,
            "variant": run.variant,
            "seed": run.seed,
            "role": "historical" if run.historical else "canonical",
            "phase": status_data.get("phase", latest_job.get("phase", "unknown")),
            "status": status_data.get("status", scheduler_state or _existing_status(run, output)),
            "slurm_job_id": job_id,
            "output_dir": str(output),
            "last_update": status_data.get("last_update"),
        })
    return rows


def _scheduler_state(job_id: str) -> Optional[str]:
    try:
        active = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            cwd=PROJECT_ROOT, check=False, capture_output=True, text=True, timeout=15,
        )
        value = active.stdout.strip().splitlines()
        if value:
            return value[0].strip().upper()
        accounting = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
            cwd=PROJECT_ROOT, check=False, capture_output=True, text=True, timeout=15,
        )
        values = [line.split("|", 1)[0].split()[0].rstrip("+").upper() for line in accounting.stdout.splitlines() if line.strip()]
        return values[0] if values else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def render_dry_run(plan: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Seed study dry run",
        "=" * 100,
    ]
    for index, item in enumerate(plan, start=1):
        lines.extend([
            f"[{index:02d}] {item['model']} | {item['variant']} | seed {item['seed']} | {item['planned_status']}",
            f"  model_id: {item['model_name']}",
            f"  regime: {item['regime']} | max_epochs: {item['max_epochs']} | canonical: {item['canonical']}",
            f"  reference_config: {item['reference_config']}",
            f"  reference_output: {item['reference_output_dir']}",
            f"  reference_status: {item['reference_status']}",
            f"  planned_config: {item['planned_config']}",
            f"  planned_output: {item['planned_output_dir']}",
            f"  resolved_output: {item['resolved_output_dir']} ({item['output_status']})",
            f"  log_dir: {item['planned_log_dir']}",
            f"  scientific_config_hash: {item['scientific_config_hash']}",
            f"  full_run_config_hash: {item['full_run_config_hash']}",
            f"  scientific_code_hash: {item['scientific_code_hash']}",
            f"  prompt_hash: {item['prompt_hash']}",
            f"  parser_hash: {item['parser_hash']}",
            f"  model_revision: {item['model_revision']}",
            f"  dataset_revision: {item['dataset_revision']}",
            f"  reference_provenance: {item['reference_provenance'].get('status', 'missing')}",
            f"  resources: {dict(item['resources'])}",
            f"  dependencies: {item['dependencies']}",
            "  config_diff:",
        ])
        for difference in item["config_diff"]:
            lines.append(
                f"    - {difference['path']}: {difference['reference']!r} -> {difference['candidate']!r}"
            )
        lines.append("  planned_commands:")
        for phase, command in item["commands"].items():
            lines.append(f"    {phase}: {_shell_join(command)}")

    all_descriptors = list(iter_run_descriptors())
    historical = [run for run in all_descriptors if run.historical]
    historical_deberta = [run for run in historical if run.model_family == "encoder"]
    historical_qwen08 = [run for run in historical if run.group_key == "qwen35-08b-qlora"]
    historical_qwen4 = [run for run in historical if run.group_key == "qwen35-4b-qlora"]
    historical_qwen27 = [run for run in historical if run.group_key == "qwen35-27b-qlora-3ep"]
    existing = [run for run in all_descriptors if run.canonical and not run.is_new]
    conflicts = sum(item["output_status"] != "FREE" for item in plan)
    lines.extend(["", "Read-only existing references"])
    for run in [*existing, *historical]:
        output = resolve_run_output(load_experiment_config(PROJECT_ROOT / run.config_path))
        lines.append(
            f"  {run.label} | {run.variant} | seed {run.seed} | "
            f"{_existing_status(run, output)} | {output}"
        )
    lines.extend([
        "",
        "Dry-run summary",
        f"Planned new training runs: {len(plan)}",
        f"Planned new inference runs: {len(plan)}",
        f"Planned new evaluation runs: {len(plan)}",
        f"New canonical DeBERTa-v3-base runs: {sum(item['group'] == 'deberta-v3-base' for item in plan)}",
        f"New canonical DeBERTa-v3-large runs: {sum(item['group'] == 'deberta-v3-large' for item in plan)}",
        f"New canonical Qwen3.5-0.8B runs: {sum(item['group'] == 'qwen35-08b-qlora' for item in plan)}",
        f"New canonical Qwen3.5-4B runs: {sum(item['group'] == 'qwen35-4b-qlora' for item in plan)}",
        f"New canonical Qwen3.5-27B 3ep runs: {sum(item['group'] == 'qwen35-27b-qlora-3ep' for item in plan)}",
        f"Historical DeBERTa runs preserved: {len(historical_deberta)}",
        f"Historical Qwen3.5-0.8B runs preserved: {len(historical_qwen08)}",
        f"Historical Qwen3.5-4B runs preserved: {len(historical_qwen4)}",
        f"Historical Qwen3.5-27B 2ep runs preserved: {len(historical_qwen27)}",
        "Duplicated zero-shot runs: 0",
        "Existing result directories modified: 0",
        f"Output path conflicts: {conflicts}",
        "Scientific config mismatches within canonical seed groups: 0",
        "Cross-seed resumes: 0",
        "Cross-variant resumes: 0",
        f"Existing canonical Seed-42 runs referenced read-only: {len(existing)}",
    ])
    return "\n".join(lines)


def _phases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.train_only:
        return ("training",)
    if args.inference_only:
        return ("inference",)
    if args.evaluation_only or args.eval_only:
        return ("evaluation",)
    if args.aggregate_only:
        return ()
    return ("training", "inference", "evaluation")


def _reference_output(group: Mapping[str, Any], seed: int) -> str:
    is_fresh_group = bool(group.get("derivation_config") and group.get("historical_runs"))
    if is_fresh_group and seed == 42:
        historical = group.get("historical_runs", [{}])[0]
        return f"{historical.get('output_dir', 'unknown')} (historical, read-only)"
    if is_fresh_group:
        return f"{group['canonical_runs'][0]['output_dir']} (canonical seed 42 config)"
    return str(group["canonical_runs"][0]["output_dir"])


def _reference_status(group: Mapping[str, Any], seed: int) -> str:
    is_fresh_group = bool(group.get("derivation_config") and group.get("historical_runs"))
    if is_fresh_group and seed == 42:
        historical = group.get("historical_runs", [{}])[0]
        return str(historical.get("status", "unknown"))
    if is_fresh_group:
        return "new_canonical_seed42_config"
    return str(group["canonical_runs"][0].get("status", "unknown"))


def _job_name(run: RunDescriptor, phase: str) -> str:
    short = {
        "deberta-v3-base": "deb-base",
        "deberta-v3-large": "deb-large",
        "qwen35-08b-qlora": "qwen08b",
        "qwen35-4b-qlora": "qwen4b",
        "qwen35-27b-qlora-3ep": "qwen27b-3ep",
    }[run.group_key]
    phase_short = {"training": "train", "inference": "infer", "evaluation": "eval"}[phase]
    return f"ner-{short}-s{run.seed}-{phase_short}"


def _log_dir(run: RunDescriptor) -> Path:
    return PROJECT_ROOT / "logs" / "seed-studies" / run.group_key / f"seed-{run.seed}"


def _venv_path() -> Path:
    configured = os.environ.get("BA_NER_VENV")
    return Path(configured).resolve() if configured else (PROJECT_ROOT / ".venv").resolve()


def _scratch_root() -> Optional[Path]:
    configured = os.environ.get("BA_NER_SCRATCH")
    if configured:
        return Path(configured)
    candidate = Path(f"/netscratch/{os.environ.get('USER', 'losman')}/ba-ner")
    return candidate if candidate.is_dir() else None


def _base_exports() -> list[str]:
    values = ["ALL", f"BA_NER_VENV={_venv_path()}"]
    scratch = _scratch_root()
    if scratch is not None:
        values.append(f"BA_NER_SCRATCH={scratch}")
    expected_commit = os.environ.get("BA_NER_EXPECTED_GIT_COMMIT")
    if expected_commit:
        values.append(f"BA_NER_EXPECTED_GIT_COMMIT={expected_commit}")
    return values


def _git_repository_state() -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or status.returncode != 0:
        detail = commit.stderr.strip() or status.stderr.strip() or "unknown Git error"
        raise SeedStudyError(f"Could not verify repository freeze: {detail}")
    return commit.stdout.strip(), status.stdout.strip()


def require_submission_repository_freeze() -> str:
    commit, worktree_status = _git_repository_state()
    if not commit:
        raise SeedStudyError("Could not determine the commit for the frozen training snapshot")
    if worktree_status:
        preview = "\n".join(worktree_status.splitlines()[:20])
        raise SeedStudyError(
            "Training submission requires a clean committed worktree. "
            "Commit or remove these changes first:\n" + preview
        )
    os.environ["BA_NER_EXPECTED_GIT_COMMIT"] = commit
    return commit


def _submit(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SeedStudyError(
            f"Submission failed ({_shell_join(command)}): {result.stderr.strip() or result.stdout.strip()}"
        )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SeedStudyError(f"SLURM returned an invalid job ID: {result.stdout!r}")
    return job_id


def _existing_status(run: RunDescriptor, output: Path) -> str:
    state = inspect_output_path(output)
    if run.historical:
        return (
            "EXISTING_COMPLETED_HISTORICAL"
            if state == "COMPLETED"
            else "EXISTING_RUNNING_HISTORICAL"
            if state == "RUNNING"
            else f"EXISTING_{state}_HISTORICAL"
        )
    if run.source == "existing":
        return "EXISTING_COMPLETED" if state == "COMPLETED" else f"EXISTING_{state}"
    return "PLANNED_CANONICAL" if state == "FREE" else state


def _shell_join(command: Sequence[str]) -> str:
    import shlex
    return shlex.join(str(value) for value in command)


def _print_status(rows: Sequence[Mapping[str, Any]]) -> None:
    print("Model | Variant | Seed | Role | Phase | Status | Job | Output")
    print("-" * 140)
    for row in rows:
        print(
            f"{row['model']} | {row['variant']} | {row['seed']} | {row['role']} | "
            f"{row['phase']} | {row['status']} | {row['slurm_job_id'] or '-'} | {row['output_dir']}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 123, 456), default=[])
    family = parser.add_mutually_exclusive_group()
    family.add_argument("--encoder-only", action="store_true")
    family.add_argument("--decoder-only", action="store_true")
    family.add_argument("--deberta-only", action="store_true")
    family.add_argument("--qwen-only", action="store_true")
    parser.add_argument("--variant")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--train-only", action="store_true")
    phase.add_argument("--inference-only", action="store_true")
    phase.add_argument("--evaluation-only", action="store_true")
    phase.add_argument("--eval-only", action="store_true")
    phase.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--group", help="Aggregation group for --aggregate-only")
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Explicitly allow only a hash-validated same-run resume/restart",
    )
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-cluster-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.status:
        _print_status(status_report())
        return 0
    if args.aggregate_only:
        report = aggregate_seed_study(resolve_results_dir(), group_key=args.group)
        print(yaml.safe_dump(report, sort_keys=False) if args.json else "Aggregation completed")
        return 0

    runs = select_new_runs(
        models=args.model,
        seeds=args.seeds,
        config_paths=args.config,
        encoder_only=args.encoder_only,
        decoder_only=args.decoder_only,
        deberta_only=args.deberta_only,
        qwen_only=args.qwen_only,
        variant=args.variant,
    )
    if not runs:
        raise SeedStudyError("The selected filters contain no new training runs")
    if args.submit:
        require_submission_repository_freeze()
    phases = _phases(args)
    report = preflight(
        runs,
        phases=phases,
        run_tests=not args.skip_tests,
        allow_resume_failed=args.resume_failed,
    )
    if args.preflight:
        print(yaml.safe_dump(report, sort_keys=False) if args.json else _render_preflight(report))
        return 0 if report["valid"] else 1
    if args.dry_run or not args.submit:
        plan_output = (
            yaml.safe_dump(report, sort_keys=False)
            if args.json
            else render_dry_run(report["plan"])
        )
        print(plan_output)
        if not report["valid"] and not args.json:
            print("\nSubmission gates\n" + _render_preflight(report))
        return 0 if report["valid"] else 1
    if not report["valid"]:
        print(_render_preflight(report), file=sys.stderr)
        return 1

    plan = report["plan"]
    if args.submit:
        registry = submit_pipelines(
            runs,
            phases=phases,
            skip_cluster_preflight=args.skip_cluster_preflight,
            allow_resume_failed=args.resume_failed,
        )
        print(yaml.safe_dump(registry, sort_keys=False))
        return 0
    return 0


def _render_preflight(report: Mapping[str, Any]) -> str:
    lines = [
        f"Preflight: {'PASS' if report['valid'] else 'FAIL'}",
        f"Selected runs: {report['selected_run_count']}",
        f"Manifest new runs: {report['manifest']['new_run_count']}",
        f"CUDA visible here: {report['cuda_visible_on_current_host']}",
        f"Tests: {'PASS' if report['tests']['returncode'] == 0 else 'FAIL' if report['tests']['ran'] else 'SKIPPED'}",
    ]
    lines.extend(f"WARNING: {warning}" for warning in report["warnings"])
    lines.extend(f"ERROR: {error}" for error in report["errors"])
    if report["tests"]["output"]:
        lines.extend(["", report["tests"]["output"]])
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeedStudyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
