"""Safe helpers for reading YAML and JSON from the local filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary, returning an empty dict on failure."""
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_json_file(path: Path, default: Any | None = None) -> Any:
    """Load a JSON file with a caller-provided default value."""
    if not path.exists() or not path.is_file():
        return [] if default is None else default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return [] if default is None else default


def read_text_file(path: Path) -> str:
    """Read a text file safely and return an empty string on failure."""
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
