"""Scientific and execution provenance for the canonical multi-seed study.

The ordinary YAML configuration hash is not sufficient for this study: prompt,
parser, preprocessing, metric, model revision, and dataset revision changes can
alter a result without changing a YAML file.  This module builds a deterministic
contract from those inputs and also snapshots every repository file used to run
the managed pipelines, including files which have not yet been committed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_CONTRACT_VERSION = 2
DATASET_REVISION = "2814b78e7af4b5a1f1886fe7ad49632de4d9dd25"
DATASET_SPLIT_SIZES = {"train": 131_280, "validation": 16_410, "test": 16_454}
MODEL_REVISIONS = {
    "microsoft/deberta-v3-base": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
    "microsoft/deberta-v3-large": "64a8c8eab3e352a784c658aef62be1662607476f",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}

COMMON_SCIENTIFIC_FILES = (
    "src/config.py",
    "src/data/dataset_loader.py",
    "src/evaluate/metrics.py",
    "src/seed_provenance.py",
)
FAMILY_SCIENTIFIC_FILES = {
    "encoder": (
        "src/data/preprocess_encoder.py",
        "src/encoder/train.py",
        "src/encoder/inference.py",
        "src/evaluate/validate_seed_run.py",
    ),
    "decoder": (
        "src/data/preprocess_decoder.py",
        "src/decoder/generation.py",
        "src/decoder/parse_output.py",
        "src/decoder/train.py",
        "src/decoder/inference.py",
        "src/evaluate/validate_seed_run.py",
    ),
}
EXECUTION_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "scripts/cluster/*.sh",
    "configs/*.yaml",
    "requirements.txt",
    "setup.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def scientific_code_manifest(model_type: str) -> Dict[str, str]:
    try:
        family_files = FAMILY_SCIENTIFIC_FILES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model type for provenance: {model_type}") from exc
    paths = (*COMMON_SCIENTIFIC_FILES, *family_files)
    return {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in paths
    }


def prompt_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    if config.get("model_type") != "decoder":
        return {"applicable": False, "prompt_sha256": None, "thinking_enabled": None}

    # Imports are local to avoid adding decoder dependencies to encoder-only tools.
    from src.data.dataset_loader import MULTINERD_LABEL_LIST
    from src.data.preprocess_decoder import build_system_prompt

    entity_types = [
        label[2:]
        for label in MULTINERD_LABEL_LIST
        if label.startswith("B-")
    ]
    prompt = build_system_prompt(entity_types)
    return {
        "applicable": True,
        "profile": "task_v1",
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "thinking_enabled": False,
        "generation_prompt": True,
        "do_sample": False,
        "parser_sha256": file_sha256(PROJECT_ROOT / "src/decoder/parse_output.py"),
    }


def scientific_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    model_name = str(config["model_name"])
    revision = MODEL_REVISIONS.get(model_name)
    code_manifest = scientific_code_manifest(str(config["model_type"]))
    return {
        "version": SCIENTIFIC_CONTRACT_VERSION,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "dataset_revision": DATASET_REVISION,
        "dataset_language": config.get("dataset_language"),
        "dataset_split_sizes": DATASET_SPLIT_SIZES,
        "code_files": code_manifest,
        "scientific_code_hash": canonical_hash(code_manifest),
        "prompt": prompt_contract(config),
        "evaluation": {
            "metric": "strict_entity_level_iob2",
            "checkpoint_selection": (
                "highest_validation_f1"
                if config.get("model_type") == "encoder"
                else "highest_generative_validation_f1"
            ),
            "test_used_for_selection": False,
        },
    }


def execution_snapshot_manifest() -> Dict[str, str]:
    paths: set[Path] = set()
    for pattern in EXECUTION_PATTERNS:
        paths.update(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in sorted(paths)
    }


def execution_snapshot() -> Dict[str, Any]:
    files = execution_snapshot_manifest()
    return {
        "files": files,
        "sha256": canonical_hash(files),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "status": _git("status", "--porcelain=v1"),
        },
    }


def reference_provenance_issues(manifest: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    accepted = {"verified", "reconstructed_verified"}
    for group in manifest.get("groups", []):
        provenance = group.get("reference_provenance", {})
        status = str(provenance.get("status", "missing"))
        if status not in accepted:
            reason = provenance.get("blocker") or "reference provenance is not verified"
            issues.append(f"{group.get('key')}: {status}: {reason}")
    return issues


def verify_cached_revisions(
    *,
    hf_home: Path,
    model_names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Validate the immutable Hub revisions used by the historical runs.

    Missing cache entries are warnings because a compute node may populate them;
    a present but different ``refs/main`` is an error and must never be silently
    accepted for a canonical seed run.
    """
    errors: list[str] = []
    warnings: list[str] = []
    expected = {"Babelscape/multinerd": DATASET_REVISION}
    expected.update({name: MODEL_REVISIONS[name] for name in model_names})
    for identifier, revision in sorted(expected.items()):
        kind = "datasets" if identifier == "Babelscape/multinerd" else "models"
        owner, name = identifier.split("/", 1)
        ref = hf_home / "hub" / f"{kind}--{owner}--{name}" / "refs" / "main"
        if not ref.is_file():
            warnings.append(f"Hub revision cache is unavailable for {identifier}: {ref}")
            continue
        actual = ref.read_text(encoding="utf-8").strip()
        if actual != revision:
            errors.append(
                f"Hub revision mismatch for {identifier}: expected {revision}, got {actual}"
            )
    return errors, warnings


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
