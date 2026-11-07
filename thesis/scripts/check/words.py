#!/usr/bin/env python3
"""Fließtextwörter je Kapitel (ohne Tabellen, Abbildungen, Listings, Gleichungen).

Aufruf: python3 words.py chapters/*.tex
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texclean import load_tex, words  # noqa: E402


def count(path: str) -> int:
    return sum(len(words(ln.text)) for ln in load_tex(path))


def main() -> int:
    total = 0
    for path in sys.argv[1:]:
        n = count(path)
        total += n
        print(f"{n:7d}  {path}")
    if len(sys.argv) > 2:
        print(f"{total:7d}  gesamt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
