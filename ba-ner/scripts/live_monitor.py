#!/usr/bin/env python3
"""CLI wrapper for the repository's existing manifest-aware live monitor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate.monitor_training import main


if __name__ == "__main__":
    main()
