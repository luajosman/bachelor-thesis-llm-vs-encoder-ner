#!/usr/bin/env python3
"""Zahlen im (geänderten) Kapiteltext gegen summary.json und generated/*.tex prüfen.

Aufruf:
  python3 numbers.py --diff HASH chapters/05_….tex      # nur hinzugefügte Zeilen des Diffs
  python3 numbers.py chapters/05_….tex                  # ganze Datei
Optionen: --summary PFAD (Standard thesis/generated/summary.json), --generated GLOB

Jede Zahl wird gesucht als Wert, Wert*100 (F1-Brüche) und auf 0–3 Nachkommastellen
gerundet in allen numerischen Blättern von summary.json sowie als Literal in den
generierten Tabellen. Jahreszahlen und sehr kleine Ganzzahlen werden als SKIP markiert.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texclean import clean_line  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.abspath(os.path.join(HERE, "..", ".."))

NUM_RE = re.compile(r"(?<![A-Za-z\\{])(\d{1,3}(?:\\,\d{3})+(?:,\d+)?|\d+(?:,\d+)?)(?![\d}])")


def leaves(obj, out: list[float]):
    if isinstance(obj, dict):
        for v in obj.values():
            leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            leaves(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))


def to_float(token: str) -> float:
    return float(token.replace("\\,", "").replace(",", "."))


def candidates(v: float) -> set[float]:
    c = set()
    for base in (v, v * 100.0):
        for d in (0, 1, 2, 3):
            c.add(round(base, d))
    return c


def added_lines(hash_: str, path: str) -> list[tuple[int, str]]:
    repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=os.path.dirname(path) or ".").decode().strip()
    rel = os.path.relpath(os.path.abspath(path), repo)
    diff = subprocess.check_output(["git", "diff", "-U0", hash_, "--", rel], cwd=repo).decode("utf-8", "replace")
    out = []
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_line = int(m.group(1)) if m else 0
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            out.append((new_line, line[1:]))
            new_line += 1
        elif line.startswith("-"):
            continue
        else:
            new_line += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--diff", default=None, help="Commit-Hash; nur hinzugefügte Zeilen prüfen")
    ap.add_argument("--summary", default=os.path.join(THESIS, "generated", "summary.json"))
    ap.add_argument("--generated", default=os.path.join(THESIS, "generated", "*.tex"))
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as fh:
        summary = json.load(fh)
    vals: list[float] = []
    leaves(summary, vals)
    cand_index: set[float] = set()
    for v in vals:
        cand_index |= candidates(v)
    generated_text = ""
    for p in glob.glob(args.generated):
        with open(p, encoding="utf-8") as fh:
            generated_text += fh.read() + "\n"

    if args.diff:
        lines = added_lines(args.diff, args.tex)
    else:
        with open(args.tex, encoding="utf-8") as fh:
            lines = [(i + 1, l.rstrip("\n")) for i, l in enumerate(fh)]

    n_total = n_found = n_unbelegt = 0
    for no, raw in lines:
        text = clean_line(raw) if not raw.lstrip().startswith("%") else ""
        if not text:
            continue
        for m in NUM_RE.finditer(raw.replace("{,}", ",")):
            tok = m.group(1)
            try:
                x = to_float(tok)
            except ValueError:
                continue
            n_total += 1
            literal = tok in generated_text
            is_year = tok.isdigit() and 1900 <= int(tok) <= 2035
            small = tok.isdigit() and int(tok) <= 12
            if literal or x in cand_index:
                status = "ok"
                n_found += 1
            elif is_year or small:
                status = "SKIP"
            else:
                status = "UNBELEGT"
                n_unbelegt += 1
            if status != "ok":
                ctx = text[max(0, text.find(tok.replace("\\,", " ")) - 40):][:110]
                print(f"  Zeile {no}: {tok:>12}  {status:8}  …{ctx}…")
    print(f"== {args.tex}: {n_total} Zahlen, {n_found} belegt, {n_unbelegt} UNBELEGT")
    return 1 if n_unbelegt else 0


if __name__ == "__main__":
    sys.exit(main())
