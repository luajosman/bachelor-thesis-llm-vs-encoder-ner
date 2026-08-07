"""Safety, provenance, and configuration utilities for the multi-seed study."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import yaml
from transformers import TrainerCallback

from src.config import SEED_STUDY_EXPERIMENTS, load_experiment_config, output_dir_from_config
from src.run_metadata import collect_run_metadata
from src.seed_provenance import (
    DATASET_REVISION,
    DATASET_SPLIT_SIZES,
    MODEL_REVISIONS,
    execution_snapshot,
    scientific_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "configs" / "seed_study_manifest.yaml"
STATUS_FILENAME = "status.json"
METADATA_FILENAME = "run_metadata.yaml"
LOCK_FILENAME = ".run.lock"
EXPECTED_NEW_TRAINING_RUNS = 15

# These fields identify a run operationally; removing them leaves the complete
# scientific/technical configuration used for group equivalence and hashing.
OPERATIONAL_FIELDS = frozenset({
    "seed",
    "data_seed",
    "experiment_name",
    "run_name",
    "output_dir",
    "logging_dir",
    "job_name",
    "job_id",
    "run_id",
    "process_id",
    "pid",
    "timestamp",
    "created_at",
    "start_time",
    "end_time",
    "last_update",
    "temporary_dir",
    "tmp_dir",
})


class SeedStudyError(RuntimeError):
    """Base class for safety failures which must block submission or a run."""


class ConfigMismatchError(SeedStudyError):
    """Raised when two supposedly equivalent configurations differ."""


class OutputCollisionError(SeedStudyError):
    """Raised when a run would write into an unsafe existing path."""


class ResumeValidationError(SeedStudyError):
    """Raised when a checkpoint does not belong to the exact same run."""


class RunLockedError(SeedStudyError):
    """Raised when another process already owns a run directory."""


@dataclass(frozen=True)
class ConfigDifference:
    path: str
    kind: str
    reference: Any
    candidate: Any

    def render(self) -> str:
        return (
            f"Path: {self.path}\n"
            f"Kind: {self.kind}\n"
            f"Reference: {self.reference!r}\n"
            f"Candidate: {self.candidate!r}"
        )


@dataclass(frozen=True)
class RunDescriptor:
    group_key: str
    label: str
    model_name: str
    model_family: str
    regime: str
    variant: str
    max_epochs: int
    seed: int
    config_path: str
    output_dir: str
    source: str
    status: str
    canonical: bool
    historical: bool
    included_in_primary_seed_aggregation: bool
    read_only: bool
    resources: Mapping[str, Any]
    identifier: Optional[str] = None
    training_job_id: Optional[int] = None
    inference_job_id: Optional[int] = None

    @property
    def is_new(self) -> bool:
        return self.source == "new"


@dataclass
class RunContext:
    descriptor: RunDescriptor
    config_path: Path
    config: Dict[str, Any]
    output_dir: Path
    scientific_config_hash: str
    full_run_config_hash: str
    phase: str
    lock_payload: Dict[str, Any]
    resume_checkpoint: Optional[str] = None
    resumed: bool = False
    started_monotonic: float = 0.0

    def update_status(self, **updates: Any) -> None:
        update_run_status(self.output_dir, updates)

    def update_metadata(self, **updates: Any) -> None:
        update_run_metadata(self.output_dir, updates)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("groups"), list):
        raise SeedStudyError(f"Invalid seed-study manifest: {manifest_path}")
    return raw


def iter_run_descriptors(
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    include_historical: bool = True,
) -> Iterator[RunDescriptor]:
    manifest = manifest or load_manifest()
    for group in manifest["groups"]:
        common = {
            "group_key": str(group["key"]),
            "label": str(group["label"]),
            "model_name": str(group["model_name"]),
            "model_family": str(group["model_family"]),
            "regime": str(group["regime"]),
            "variant": str(group.get("variant", "default")),
            "max_epochs": int(group["max_epochs"]),
            "resources": group.get("resources", {}),
        }
        for run in group.get("canonical_runs", []):
            yield RunDescriptor(
                **common,
                seed=int(run["seed"]),
                config_path=str(run["config"]),
                output_dir=str(run["output_dir"]),
                source=str(run["source"]),
                status=str(run.get("status", "unknown")),
                canonical=True,
                historical=False,
                included_in_primary_seed_aggregation=True,
                read_only=bool(run.get("read_only", False)),
            )
        if include_historical:
            for run in group.get("historical_runs", []):
                yield RunDescriptor(
                    **{**common, "max_epochs": int(run["max_epochs"])},
                    seed=int(run["seed"]),
                    config_path=str(run["config"]),
                    output_dir=str(run["output_dir"]),
                    source=str(run["source"]),
                    status=str(run.get("status", "unknown")),
                    canonical=False,
                    historical=True,
                    included_in_primary_seed_aggregation=bool(
                        run.get("included_in_primary_seed_aggregation", False)
                    ),
                    read_only=bool(run.get("read_only", True)),
                    identifier=str(run.get("id")) if run.get("id") else None,
                    training_job_id=_optional_int(run.get("training_job_id")),
                    inference_job_id=_optional_int(run.get("inference_job_id")),
                )


def descriptor_for_config(
    config_path: str | Path,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Optional[RunDescriptor]:
    normalized = _normalize_config_path(config_path)
    for descriptor in iter_run_descriptors(manifest):
        if _normalize_config_path(descriptor.config_path) == normalized:
            return descriptor
    return None


def recursive_config_diff(
    reference: Any,
    candidate: Any,
    *,
    path: str = "<root>",
) -> list[ConfigDifference]:
    """Compare nested mappings/lists, preserving type and structural changes."""
    if type(reference) is not type(candidate):
        return [ConfigDifference(path, "type_mismatch", reference, candidate)]

    if isinstance(reference, dict):
        differences: list[ConfigDifference] = []
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        for key in sorted(reference_keys - candidate_keys, key=str):
            differences.append(ConfigDifference(_join(path, key), "missing", reference[key], None))
        for key in sorted(candidate_keys - reference_keys, key=str):
            differences.append(ConfigDifference(_join(path, key), "additional", None, candidate[key]))
        for key in sorted(reference_keys & candidate_keys, key=str):
            differences.extend(
                recursive_config_diff(reference[key], candidate[key], path=_join(path, key))
            )
        return differences

    if isinstance(reference, list):
        differences = []
        if len(reference) != len(candidate):
            differences.append(ConfigDifference(path, "list_length", len(reference), len(candidate)))
        for index, (ref_value, candidate_value) in enumerate(zip(reference, candidate)):
            differences.extend(
                recursive_config_diff(
                    ref_value,
                    candidate_value,
                    path=f"{path}[{index}]",
                )
            )
        return differences

    if reference != candidate:
        return [ConfigDifference(path, "value", reference, candidate)]
    return []


def validate_seed_equivalence(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    additionally_allowed_fields: Iterable[str] = (),
) -> list[ConfigDifference]:
    allowed = OPERATIONAL_FIELDS | frozenset(additionally_allowed_fields)
    differences = recursive_config_diff(reference, candidate)
    forbidden = [difference for difference in differences if _leaf_name(difference.path) not in allowed]
    if forbidden:
        rendered = "\n\n".join(difference.render() for difference in forbidden)
        raise ConfigMismatchError(
            "Scientific configuration mismatch. Only approved seed-specific fields may differ.\n\n"
            + rendered
        )
    return differences


def validate_seed_group(configs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    if set(configs) != {42, 123, 456}:
        raise ConfigMismatchError(f"Expected seeds 42, 123, 456; got {sorted(configs)}")
    reference = configs[42]
    diffs = {
        seed: validate_seed_equivalence(reference, config)
        for seed, config in sorted(configs.items())
        if seed != 42
    }
    hashes = {seed: scientific_config_hash(config) for seed, config in configs.items()}
    if len(set(hashes.values())) != 1:
        raise ConfigMismatchError(f"Scientific config hashes differ: {hashes}")
    full_hashes = {seed: full_run_config_hash(config) for seed, config in configs.items()}
    if len(set(full_hashes.values())) != len(full_hashes):
        raise ConfigMismatchError(f"Full run config hashes are not unique: {full_hashes}")
    return {"diffs": diffs, "scientific_hashes": hashes, "full_hashes": full_hashes}


def scientific_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "resolved_config": _without_operational_fields(config),
    }
    if {"model_name", "model_type"}.issubset(config):
        payload["scientific_contract"] = scientific_contract(config)
    return payload


def scientific_config_hash(config: Mapping[str, Any]) -> str:
    return _sha256(scientific_config(config))


def full_run_config_hash(config: Mapping[str, Any]) -> str:
    payload: Dict[str, Any] = {"resolved_config": dict(config)}
    if {"model_name", "model_type"}.issubset(config):
        payload["scientific_contract"] = scientific_contract(config)
    return _sha256(payload)


def validate_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    manifest = load_manifest(path)
    groups_report: Dict[str, Any] = {}
    all_outputs: Dict[str, str] = {}
    all_full_hashes: Dict[str, str] = {}
    historical_outputs: Dict[str, str] = {}
    new_run_count = 0

    for group in manifest["groups"]:
        for run in group.get("historical_runs", []):
            config_path = PROJECT_ROOT / str(run["config"])
            cfg = load_experiment_config(config_path)
            output = str(run["output_dir"])
            if str(cfg["output_dir"]) != output:
                raise SeedStudyError(f"{config_path}: historical output_dir does not match manifest")
            if output in historical_outputs:
                raise SeedStudyError(
                    f"Historical output collision: {output} is used by "
                    f"{historical_outputs[output]} and {group['key']}"
                )
            if run.get("canonical") is not False or run.get("read_only") is not True:
                raise SeedStudyError(f"{group['key']}: historical runs must be non-canonical and read-only")
            if run.get("included_in_primary_seed_aggregation") is not False:
                raise SeedStudyError(f"{group['key']}: historical runs cannot enter the primary aggregate")
            if run.get("scientific_config_hash") != scientific_config_hash(cfg):
                raise SeedStudyError(f"{config_path}: historical scientific_config_hash is stale")
            if run.get("full_run_config_hash") != full_run_config_hash(cfg):
                raise SeedStudyError(f"{config_path}: historical full_run_config_hash is stale")
            historical_outputs[output] = str(group["key"])

    for group in manifest["groups"]:
        runs = group.get("canonical_runs", [])
        seeds = {int(run["seed"]) for run in runs}
        if seeds != {42, 123, 456}:
            raise SeedStudyError(f"{group['key']}: canonical seeds must be 42, 123, 456")

        configs: Dict[int, Dict[str, Any]] = {}
        for run in runs:
            config_path = PROJECT_ROOT / str(run["config"])
            cfg = load_experiment_config(config_path)
            seed = int(run["seed"])
            if int(cfg.get("seed", -1)) != seed:
                raise SeedStudyError(f"{config_path}: seed does not match manifest seed {seed}")
            if str(cfg["output_dir"]) != str(run["output_dir"]):
                raise SeedStudyError(f"{config_path}: output_dir does not match manifest")
            if int(cfg.get("num_train_epochs", -1)) != int(group["max_epochs"]):
                raise SeedStudyError(f"{config_path}: num_train_epochs does not match group")
            configs[seed] = cfg

            output = str(run["output_dir"])
            if output in historical_outputs:
                raise SeedStudyError(
                    f"Canonical output {output} collides with historical group "
                    f"{historical_outputs[output]}"
                )
            if output in all_outputs:
                raise SeedStudyError(
                    f"Output collision: {output} is used by {all_outputs[output]} and {group['key']} seed {seed}"
                )
            all_outputs[output] = f"{group['key']} seed {seed}"
            full_hash = full_run_config_hash(cfg)
            stored_scientific_hash = run.get("scientific_config_hash")
            stored_full_hash = run.get("full_run_config_hash")
            if not isinstance(stored_scientific_hash, str) or len(stored_scientific_hash) != 64:
                raise SeedStudyError(f"{config_path}: manifest scientific_config_hash is missing")
            if not isinstance(stored_full_hash, str) or len(stored_full_hash) != 64:
                raise SeedStudyError(f"{config_path}: manifest full_run_config_hash is missing")
            if stored_scientific_hash != scientific_config_hash(cfg):
                raise SeedStudyError(
                    f"{config_path}: manifest scientific_config_hash is stale"
                )
            if stored_full_hash != full_hash:
                raise SeedStudyError(f"{config_path}: manifest full_run_config_hash is stale")
            if full_hash in all_full_hashes:
                raise SeedStudyError(
                    f"Duplicate full config hash for {all_full_hashes[full_hash]} and {group['key']} seed {seed}"
                )
            all_full_hashes[full_hash] = f"{group['key']} seed {seed}"
            new_run_count += int(run.get("source") == "new")

        group_report = validate_seed_group(configs)
        group_report["reference_provenance"] = group.get("reference_provenance", {})
        if group.get("derivation_config") and group["key"] != "qwen35-27b-qlora-3ep":
            original = load_experiment_config(PROJECT_ROOT / str(group["derivation_config"]))
            derivation_diffs = validate_seed_equivalence(original, configs[42])
            group_report["historical_to_canonical_seed42_diff"] = derivation_diffs
        elif group["key"] == "qwen35-27b-qlora-3ep":
            original = load_experiment_config(PROJECT_ROOT / str(group["derivation_config"]))
            derivation_diffs = validate_seed_equivalence(
                original,
                configs[42],
                additionally_allowed_fields={"num_train_epochs"},
            )
            epoch_diffs = [d for d in derivation_diffs if _leaf_name(d.path) == "num_train_epochs"]
            if len(epoch_diffs) != 1 or epoch_diffs[0].reference != 2 or epoch_diffs[0].candidate != 3:
                raise ConfigMismatchError(
                    "Qwen3.5-27B canonical seed 42 must change num_train_epochs exactly from 2 to 3"
                )
            group_report["historical_to_canonical_seed42_diff"] = derivation_diffs

        groups_report[str(group["key"])] = group_report

    expected = int(manifest.get("planned_new_training_runs", -1))
    if new_run_count != EXPECTED_NEW_TRAINING_RUNS or expected != EXPECTED_NEW_TRAINING_RUNS:
        raise SeedStudyError(
            f"Expected exactly {EXPECTED_NEW_TRAINING_RUNS} new training runs, "
            f"got {new_run_count} (manifest declares {expected})"
        )
    if manifest.get("zero_shot", {}).get("repeated_for_seeds") is not False:
        raise SeedStudyError("Zero-shot experiments must not be repeated for training seeds")

    return {
        "valid": True,
        "new_run_count": new_run_count,
        "groups": groups_report,
        "unique_output_count": len(all_outputs),
        "historical_output_count": len(historical_outputs),
        "unique_full_hash_count": len(all_full_hashes),
    }


def resolve_run_output(config: Mapping[str, Any]) -> Path:
    """Resolve the configured path, using the shared scratch root when active."""
    if os.environ.get("BA_NER_RESULTS_ROOT"):
        return output_dir_from_config(dict(config))
    scratch = Path(f"/netscratch/{os.environ.get('USER', 'losman')}/ba-ner/results")
    configured = Path(str(config["output_dir"]))
    if scratch.is_dir() and configured.parts and configured.parts[0] == "results":
        return scratch.joinpath(*configured.parts[1:])
    return output_dir_from_config(dict(config))


def inspect_output_path(path: Path) -> str:
    if not path.exists():
        return "FREE"
    if not path.is_dir():
        return "COLLISION_NON_DIRECTORY"
    status = _load_json(path / STATUS_FILENAME)
    status_value = str(status.get("status", "")).upper()
    if (path / LOCK_FILENAME).exists() or status_value in {
        "RUNNING", "VALIDATING", "SAVING", "INFERENCE_RUNNING", "EVALUATION_RUNNING"
    }:
        return "RUNNING"
    if status_value == "COMPLETED" or (
        (path / "results.yaml").is_file() and (path / "inference_metrics.yaml").is_file()
    ):
        return "COMPLETED"
    if status_value == "FAILED":
        return "FAILED"
    if any(path.glob("checkpoint-*")):
        return "INCOMPLETE_WITH_CHECKPOINTS"
    if (path / METADATA_FILENAME).is_file():
        return status_value or "INITIALIZED"
    return "UNKNOWN_EXISTING"


def validate_resume(
    *,
    expected_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
    checkpoint: Path,
) -> None:
    expected_scientific = scientific_config_hash(expected_config)
    expected_full = full_run_config_hash(expected_config)
    checks = {
        "model_name": (metadata.get("model_name"), expected_config.get("model_name")),
        "seed": (metadata.get("seed"), expected_config.get("seed")),
        "num_train_epochs": (metadata.get("num_train_epochs"), expected_config.get("num_train_epochs")),
        "scientific_config_hash": (metadata.get("scientific_config_hash"), expected_scientific),
        "full_run_config_hash": (metadata.get("full_run_config_hash"), expected_full),
        "output_dir": (str(metadata.get("resolved_output_dir")), str(output_dir)),
    }
    mismatches = [f"{key}: {actual!r} != {expected!r}" for key, (actual, expected) in checks.items() if actual != expected]
    try:
        checkpoint.relative_to(output_dir)
    except ValueError:
        mismatches.append(f"checkpoint {checkpoint} is outside {output_dir}")
    if checkpoint.parent != output_dir or not checkpoint.name.startswith("checkpoint-"):
        mismatches.append(f"checkpoint {checkpoint} is not a direct checkpoint of this run")
    if mismatches:
        raise ResumeValidationError("Unsafe resume blocked:\n- " + "\n- ".join(mismatches))


@contextmanager
def managed_run_phase(config_path: str | Path, phase: str) -> Iterator[Optional[RunContext]]:
    """Guard a new run phase with immutable metadata, hashes, and a lock."""
    config_path = Path(config_path)
    descriptor = descriptor_for_config(config_path)
    if descriptor is None:
        yield None
        return
    if descriptor.read_only:
        raise OutputCollisionError(
            f"Reference/historical run is read-only and cannot enter phase {phase}: {descriptor.output_dir}"
        )
    if descriptor.config_path not in {spec.config_path for spec in SEED_STUDY_EXPERIMENTS.values()}:
        raise SeedStudyError(f"Managed run is not registered: {descriptor.config_path}")

    cfg = load_experiment_config(config_path)
    output_dir = output_dir_from_config(cfg)
    scientific_hash = scientific_config_hash(cfg)
    full_hash = full_run_config_hash(cfg)
    existed = output_dir.exists()
    if not existed:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise OutputCollisionError(f"Output path is not a directory: {output_dir}")

    metadata_path = output_dir / METADATA_FILENAME
    metadata = _load_yaml(metadata_path)
    resume_checkpoint: Optional[str] = None
    resumed = False
    if existed:
        if not metadata:
            raise OutputCollisionError(
                f"Existing output has no managed provenance and will not be touched: {output_dir}"
            )
        _validate_existing_metadata(metadata, cfg, output_dir)
        if phase == "training":
            completed = (output_dir / "results.yaml").is_file()
            if completed:
                raise OutputCollisionError(f"Completed training run will not be overwritten: {output_dir}")
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            )
            allow_resume = os.environ.get("BA_NER_ALLOW_RESUME") == "1" or int(
                os.environ.get("SLURM_RESTART_COUNT", "0") or 0
            ) > 0
            allow_restart = os.environ.get("BA_NER_ALLOW_RESTART") == "1"
            if checkpoints:
                if not allow_resume:
                    raise ResumeValidationError(
                        f"Checkpoints exist but explicit same-run resume was not enabled: {output_dir}"
                    )
                checkpoint = checkpoints[-1]
                validate_resume(
                    expected_config=cfg,
                    metadata=metadata,
                    output_dir=output_dir,
                    checkpoint=checkpoint,
                )
                resume_checkpoint = str(checkpoint)
                resumed = True
            elif str(_load_json(output_dir / STATUS_FILENAME).get("status", "")).upper() == "FAILED" and not allow_restart:
                raise ResumeValidationError(
                    f"Failed run requires explicit restart mode: {output_dir}"
                )
        elif phase in {"inference", "evaluation"} and not (output_dir / "results.yaml").is_file():
            raise OutputCollisionError(f"Training result is missing for {phase}: {output_dir}")

    phase_started_at = utc_now()
    phase_started_monotonic = time.monotonic()
    lock_payload = {
        "experiment_name": cfg["experiment_name"],
        "variant": descriptor.variant,
        "seed": int(cfg["seed"]),
        "phase": phase,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": phase_started_at,
        "full_run_config_hash": full_hash,
    }
    _acquire_lock(output_dir / LOCK_FILENAME, lock_payload)

    try:
        if not metadata:
            snapshot = execution_snapshot()
            metadata = _build_metadata(
                descriptor=descriptor,
                cfg=cfg,
                config_path=config_path,
                output_dir=output_dir,
                scientific_hash=scientific_hash,
                full_hash=full_hash,
            )
            atomic_write_yaml(metadata_path, metadata)
            atomic_write_yaml(output_dir / "config_resolved.yaml", cfg)
            atomic_write_bytes(output_dir / "config_source.yaml", config_path.read_bytes())
            atomic_write_yaml(output_dir / "scientific_contract.yaml", scientific_contract(cfg))
            environment_dir = output_dir / "environment"
            environment_dir.mkdir(exist_ok=True)
            _write_package_versions(environment_dir / "package_versions.txt", metadata)
            atomic_write_yaml(environment_dir / "code_snapshot.yaml", snapshot)
            update_run_metadata(output_dir, {
                "execution_code_hash": snapshot["sha256"],
                "execution_code_file_hashes": snapshot["files"],
                "git_status_snapshot": snapshot["git"],
                "environment_snapshot_path": str(environment_dir / "code_snapshot.yaml"),
            })

        update_run_metadata(output_dir, {
            "resumed": resumed,
            "resume_checkpoint": resume_checkpoint,
        })

        _record_phase_transition(
            output_dir,
            phase=phase,
            status="RUNNING",
            timestamp=phase_started_at,
            exit_code=None,
        )

        initial_status = {
            "experiment_name": cfg["experiment_name"],
            "model_name": cfg["model_name"],
            "variant": descriptor.variant,
            "canonical": descriptor.canonical,
            "historical": descriptor.historical,
            "seed": int(cfg["seed"]),
            "max_epochs": int(cfg["num_train_epochs"]),
            "phase": phase,
            "status": {
                "training": "RUNNING",
                "inference": "INFERENCE_RUNNING",
                "evaluation": "EVALUATION_RUNNING",
            }.get(phase, "RUNNING"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
            "resumed": resumed,
            "resume_checkpoint": resume_checkpoint,
            "scientific_config_hash": scientific_hash,
            "full_run_config_hash": full_hash,
            "start_time": phase_started_at,
            "elapsed_seconds": 0.0,
            "last_update": utc_now(),
        }
        atomic_write_json(output_dir / STATUS_FILENAME, initial_status)
        context = RunContext(
            descriptor=descriptor,
            config_path=config_path,
            config=cfg,
            output_dir=output_dir,
            scientific_config_hash=scientific_hash,
            full_run_config_hash=full_hash,
            phase=phase,
            lock_payload=lock_payload,
            resume_checkpoint=resume_checkpoint,
            resumed=resumed,
            started_monotonic=phase_started_monotonic,
        )
        yield context
    except BaseException as exc:
        finished_at = utc_now()
        elapsed = max(0.0, time.monotonic() - phase_started_monotonic)
        try:
            update_run_status(output_dir, {
                "status": "FAILED",
                "phase": phase,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "end_time": finished_at,
                "elapsed_seconds": elapsed,
                "exit_code": 1,
            })
            _record_phase_transition(
                output_dir,
                phase=phase,
                status="FAILED",
                timestamp=finished_at,
                exit_code=1,
                elapsed_seconds=elapsed,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise
    else:
        finished_at = utc_now()
        elapsed = max(0.0, time.monotonic() - phase_started_monotonic)
        final_status = {
            "training": "TRAINING_COMPLETED",
            "inference": "INFERENCE_COMPLETED",
            "evaluation": "COMPLETED",
        }.get(phase, "COMPLETED")
        update_run_status(output_dir, {
            "status": final_status,
            "phase": phase,
            "end_time": finished_at,
            "elapsed_seconds": elapsed,
            "exit_code": 0,
        })
        _record_phase_transition(
            output_dir,
            phase=phase,
            status=final_status,
            timestamp=finished_at,
            exit_code=0,
            elapsed_seconds=elapsed,
        )
    finally:
        _release_lock(output_dir / LOCK_FILENAME, lock_payload)


class RunStatusCallback(TrainerCallback):
    """Trainer callback that only writes status and never consumes RNG state."""

    def __init__(self, context: RunContext):
        self.context = context
        self.best_eval_f1: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_epoch: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(state, status="RUNNING")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        updates = {
            key: logs[key]
            for key in (
                "loss", "eval_loss", "eval_precision", "eval_recall", "eval_f1",
                "learning_rate", "grad_norm"
            )
            if key in logs
        }
        if "loss" in updates:
            updates["train_loss"] = updates.pop("loss")
        self._write(state, status="RUNNING", **updates)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        current = metrics.get("eval_f1")
        if isinstance(current, (int, float)):
            current_value = float(current)
            if self.best_eval_f1 is None or current_value > self.best_eval_f1:
                self.best_eval_f1 = current_value
                self.best_step = int(getattr(state, "global_step", 0) or 0)
                epoch = getattr(state, "epoch", None)
                self.best_epoch = float(epoch) if isinstance(epoch, (int, float)) else None
        self._write(
            state,
            status="VALIDATING",
            eval_loss=metrics.get("eval_loss"),
            eval_precision=metrics.get("eval_precision"),
            eval_recall=metrics.get("eval_recall"),
            eval_f1=current,
            best_eval_f1=self.best_eval_f1,
            best_step=self.best_step,
            best_epoch=self.best_epoch,
        )

    def on_save(self, args, state, control, **kwargs):
        self._write(state, status="SAVING")

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, status="TRAINING_COMPLETED")

    def _write(self, state: Any, *, status: str, **updates: Any) -> None:
        try:
            epoch = getattr(state, "epoch", None)
            step = int(getattr(state, "global_step", 0) or 0)
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            progress = (100.0 * step / max_steps) if max_steps else None
            clean_updates = {key: value for key, value in updates.items() if value is not None}
            elapsed = max(0.0, time.monotonic() - self.context.started_monotonic)
            eta = (
                elapsed * (max_steps - step) / step
                if max_steps and step > 0 and step <= max_steps
                else None
            )
            peak_vram_bytes = None
            try:
                import torch
                if torch.cuda.is_available():
                    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
            except Exception:
                peak_vram_bytes = None
            self.context.update_status(
                status=status,
                phase="training",
                epoch=epoch,
                step=step,
                max_steps=max_steps,
                progress_percent=progress,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                peak_vram_bytes=peak_vram_bytes,
                **clean_updates,
            )
        except Exception:
            # Monitoring is deliberately non-fatal and semantically inert.
            return


def update_run_status(output_dir: Path, updates: Mapping[str, Any]) -> None:
    status_path = output_dir / STATUS_FILENAME
    current = _load_json(status_path)
    current.update(dict(updates))
    current["last_update"] = utc_now()
    atomic_write_json(status_path, current)


def update_run_metadata(output_dir: Path, updates: Mapping[str, Any]) -> None:
    metadata_path = output_dir / METADATA_FILENAME
    current = _load_yaml(metadata_path)
    if not current:
        raise SeedStudyError(f"Run metadata is missing or invalid: {metadata_path}")
    current.update(dict(updates))
    current["last_update"] = utc_now()
    atomic_write_yaml(metadata_path, current)


def record_dataset_metadata(context: Optional[RunContext], dataset: Any) -> None:
    if context is None:
        return
    fingerprints: Dict[str, Any] = {}
    split_sizes: Dict[str, int] = {}
    for name in ("train", "validation", "test"):
        try:
            split = dataset[name]
        except (KeyError, TypeError):
            continue
        split_sizes[name] = len(split)
        fingerprints[name] = getattr(split, "_fingerprint", None)
    context.update_metadata(
        dataset_fingerprint=fingerprints or None,
        dataset_split_sizes=split_sizes or DATASET_SPLIT_SIZES,
        dataset_metadata_provenance="captured_at_runtime",
    )


def _record_phase_transition(
    output_dir: Path,
    *,
    phase: str,
    status: str,
    timestamp: str,
    exit_code: Optional[int],
    elapsed_seconds: Optional[float] = None,
    failure_reason: Optional[str] = None,
) -> None:
    metadata_path = output_dir / METADATA_FILENAME
    metadata = _load_yaml(metadata_path)
    if not metadata:
        return
    history = metadata.get("phase_history")
    if not isinstance(history, list):
        history = []
    event = {
        "phase": phase,
        "status": status,
        "timestamp": timestamp,
        "exit_code": exit_code,
    }
    if elapsed_seconds is not None:
        event["elapsed_seconds"] = elapsed_seconds
    if failure_reason is not None:
        event["failure_reason"] = failure_reason
    history.append(event)
    metadata.update({
        "phase": phase,
        "status": status,
        "last_update": timestamp,
        "exit_code": exit_code,
        "phase_history": history,
    })
    if status == "RUNNING":
        metadata["start_time"] = timestamp
    else:
        metadata["end_time"] = timestamp
        metadata["elapsed_time"] = elapsed_seconds
        metadata["failure_reason"] = failure_reason
    atomic_write_yaml(metadata_path, metadata)


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, lambda handle: json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False))


def atomic_write_yaml(path: Path, value: Any) -> None:
    _atomic_write(path, lambda handle: yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True))


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _atomic_write(path: Path, writer: Any) -> None:
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
            writer(handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _build_metadata(
    *,
    descriptor: RunDescriptor,
    cfg: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    scientific_hash: str,
    full_hash: str,
) -> Dict[str, Any]:
    runtime = collect_run_metadata(dict(cfg))
    contract = scientific_contract(cfg)
    model_revision = MODEL_REVISIONS.get(str(cfg["model_name"]))
    effective_batch_size = (
        int(cfg.get("per_device_train_batch_size", 1))
        * int(cfg.get("gradient_accumulation_steps", 1))
    )
    return {
        "experiment_name": cfg["experiment_name"],
        "experiment_base_name": descriptor.group_key,
        "run_name": f"{cfg['experiment_name']}_multinerd",
        "model_name": cfg["model_name"],
        "model_revision": model_revision,
        "model_revision_provenance": "historical_hub_cache_ref",
        "tokenizer_name": cfg["model_name"],
        "tokenizer_revision": model_revision,
        "tokenizer_revision_provenance": "historical_hub_cache_ref",
        "model_type": cfg["model_type"],
        "mode": cfg.get("mode"),
        "regime": descriptor.regime,
        "variant": descriptor.variant,
        "seed": int(cfg["seed"]),
        "data_seed": cfg.get("data_seed", cfg.get("seed")),
        "num_train_epochs": cfg["num_train_epochs"],
        "canonical": descriptor.canonical,
        "historical_or_exploratory": descriptor.historical,
        "included_in_primary_seed_aggregation": descriptor.included_in_primary_seed_aggregation,
        "scientific_config_hash": scientific_hash,
        "full_run_config_hash": full_hash,
        "scientific_contract_version": contract["version"],
        "scientific_code_hash": contract["scientific_code_hash"],
        "prompt_hash": contract["prompt"].get("prompt_sha256"),
        "parser_hash": contract["prompt"].get("parser_sha256"),
        "config_source": str(config_path),
        "configured_output_dir": cfg["output_dir"],
        "resolved_output_dir": str(output_dir),
        "dataset": cfg["dataset"],
        "dataset_config": None,
        "dataset_revision": DATASET_REVISION,
        "dataset_revision_provenance": "historical_hub_cache_ref",
        "dataset_fingerprint": None,
        "dataset_language": cfg["dataset_language"],
        "dataset_split_sizes": DATASET_SPLIT_SIZES,
        "label_count": 31,
        "entity_type_count": 15,
        "created_at": utc_now(),
        "start_time": None,
        "end_time": None,
        "elapsed_time": None,
        "exit_code": None,
        "failure_reason": None,
        "phase": "initialization",
        "phase_history": [],
        "status": "INITIALIZED",
        "resumed": False,
        "resume_checkpoint": None,
        "runtime": runtime,
        "training": {
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
            "per_device_eval_batch_size": cfg.get("per_device_eval_batch_size"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
            "effective_batch_size_per_process": effective_batch_size,
            "learning_rate": cfg.get("learning_rate"),
            "lr_scheduler_type": cfg.get("lr_scheduler_type"),
            "optimizer": cfg.get("optim", "trainer_default"),
            "warmup_ratio": cfg.get("warmup_ratio"),
            "weight_decay": cfg.get("weight_decay"),
            "max_seq_length": cfg.get("max_seq_length", cfg.get("max_length")),
            "quantization": "4bit_qlora" if cfg.get("use_qlora") else "none",
            "lora": {
                "r": cfg.get("lora_r"),
                "alpha": cfg.get("lora_alpha"),
                "dropout": cfg.get("lora_dropout"),
                "target_modules": cfg.get("target_modules"),
            } if cfg.get("model_type") == "decoder" else None,
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
    }


def _validate_existing_metadata(
    metadata: Mapping[str, Any],
    cfg: Mapping[str, Any],
    output_dir: Path,
) -> None:
    expected = {
        "experiment_name": cfg["experiment_name"],
        "model_name": cfg["model_name"],
        "seed": cfg["seed"],
        "num_train_epochs": cfg["num_train_epochs"],
        "scientific_config_hash": scientific_config_hash(cfg),
        "full_run_config_hash": full_run_config_hash(cfg),
        "resolved_output_dir": str(output_dir),
    }
    mismatches = [
        f"{key}: {metadata.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ResumeValidationError("Existing run metadata does not match:\n- " + "\n- ".join(mismatches))


def _acquire_lock(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        existing = _load_json(path)
        raise RunLockedError(f"Run is already locked: {path}; owner={existing}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _release_lock(path: Path, expected: Mapping[str, Any]) -> None:
    current = _load_json(path)
    if current == dict(expected):
        path.unlink(missing_ok=True)


def _write_package_versions(path: Path, metadata: Mapping[str, Any]) -> None:
    packages = metadata.get("runtime", {}).get("packages", {})
    lines = [
        f"{name}=={details.get('version') if details.get('installed') else 'unavailable'}"
        for name, details in sorted(packages.items())
    ]
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_operational_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_operational_fields(item)
            for key, item in value.items()
            if str(key) not in OPERATIONAL_FIELDS
        }
    if isinstance(value, list):
        return [_without_operational_fields(item) for item in value]
    return value


def _join(path: str, key: Any) -> str:
    return str(key) if path == "<root>" else f"{path}.{key}"


def _leaf_name(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return leaf.split("[", 1)[0]


def _normalize_config_path(path: str | Path) -> str:
    value = Path(path).as_posix()
    marker = "/configs/"
    if marker in value:
        return "configs/" + value.split(marker, 1)[1]
    return value.removeprefix("./")


def _optional_int(value: Any) -> Optional[int]:
    return int(value) if value is not None else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError, TypeError):
        return {}
