"""Shared path helpers for the filesystem-backed dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BA_NER_ROOT = REPO_ROOT / "ba-ner"
CONFIGS_DIR = BA_NER_ROOT / "configs"
RESULTS_DIR = BA_NER_ROOT / "results"
DEFAULT_DATASET = "multinerd"
DEFAULT_RESULTS_DIR = RESULTS_DIR / DEFAULT_DATASET
LOGS_DIR = BA_NER_ROOT / "logs" / "dashboard"


def ensure_project_paths() -> None:
    """Expose the original ML project package path for imports like ``src.*``."""
    project_root = str(BA_NER_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def ensure_runtime_directories() -> None:
    """Create runtime directories used by the dashboard when needed."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def as_display_path(path: Path) -> str:
    """Return a repo-relative path when possible for cleaner API responses."""
    try:
        return str(path.relative_to(BA_NER_ROOT))
    except ValueError:
        return str(path)


def experiment_dir(experiment_id: str, dataset: str = DEFAULT_DATASET) -> Path:
    """Resolve an experiment directory under the standard results layout."""
    return RESULTS_DIR / dataset / experiment_id


def config_file_by_id(config_id: str) -> Path:
    """Resolve a config file by stem id."""
    return CONFIGS_DIR / f"{config_id}.yaml"


ensure_project_paths()
