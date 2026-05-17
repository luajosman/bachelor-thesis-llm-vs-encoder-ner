"""Runtime metadata helpers for reproducible experiment outputs."""

from __future__ import annotations

import importlib.metadata
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "seqeval",
    "numpy",
    "pyyaml",
    "rich",
    "matplotlib",
)


def collect_run_metadata(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Collect best-effort runtime metadata without requiring optional packages."""
    cfg = cfg or {}
    metadata: Dict[str, Any] = {
        "git": _git_metadata(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hostname": socket.gethostname(),
        "packages": _package_versions(TRACKED_PACKAGES),
        "cuda": _cuda_metadata(),
    }

    revisions = {
        key: cfg[key]
        for key in ("model_revision", "dataset_revision")
        if key in cfg
    }
    if revisions:
        metadata["revisions"] = revisions

    return metadata


def _git_metadata() -> Dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _git_dirty(),
    }


def _git_dirty() -> bool | None:
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def _run_git(args: list[str]) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_versions(packages: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    versions: Dict[str, Dict[str, Any]] = {}
    for package in packages:
        try:
            versions[package] = {
                "installed": True,
                "version": importlib.metadata.version(package),
            }
        except importlib.metadata.PackageNotFoundError:
            versions[package] = {
                "installed": False,
                "version": None,
            }
    return versions


def _cuda_metadata() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_importable": False,
            "error": str(exc),
            "available": False,
            "device_count": 0,
            "devices": [],
        }

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    devices = [
        torch.cuda.get_device_name(i)
        for i in range(device_count)
    ]
    return {
        "torch_importable": True,
        "available": available,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": device_count,
        "devices": devices,
    }
