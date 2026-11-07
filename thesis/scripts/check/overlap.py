#!/usr/bin/env python3
"""Wörtliche Überschneidungen (n-Wort-Shingles) zwischen Thesis-Text und Quelltexten.

Aufruf:
  python3 overlap.py KAPITEL.tex [weitere.tex] --against 'build/literature_audit_2026_11_06/*' 'review/*.md' --n 8 [--whitelist datei]

Meldet jede Folge von n Wörtern aus dem Fließtext (Captions/Tabellen ausgenommen),
die in einer der Vergleichsdateien vorkommt, mit Zeile und Quelle.
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texclean import load_tex, normalize_tokens, clean_line  # noqa: E402


def read_source(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if path.endswith((".html", ".htm")):
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
    if path.endswith(".tex"):
        text = "\n".join(clean_line(l) for l in text.splitlines())
    return text


def shingles(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield i, " ".join(tokens[i:i + n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="+")
    ap.add_argument("--against", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--whitelist", default=None, help="Textdatei; deren Shingles werden ignoriert")
    ap.add_argument("--keep-floats", action="store_true", help="Captions/Tabellen mitprüfen")
    args = ap.parse_args()

    index: dict[str, list[str]] = {}
    nsrc = 0
    for pattern in args.against:
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path):
                continue
            nsrc += 1
            toks = normalize_tokens(read_source(path))
            name = os.path.basename(path)
            for _, sh in shingles(toks, args.n):
                index.setdefault(sh, []).append(name)

    white: set[str] = set()
    if args.whitelist:
        toks = normalize_tokens(read_source(args.whitelist))
        white = {sh for _, sh in shingles(toks, args.n)}

    total_hits = 0
    for tex in args.tex:
        lines = load_tex(tex, drop_floats=not args.keep_floats)
        toks: list[str] = []
        line_of: list[int] = []
        for ln in lines:
            for t in normalize_tokens(ln.text):
                toks.append(t)
                line_of.append(ln.number)
        hits = []
        for i, sh in shingles(toks, args.n):
            if sh in white:
                continue
            srcs = index.get(sh)
            if srcs:
                hits.append((i, sh, sorted(set(srcs))))
        # zusammenhängende Treffer zu Läufen bündeln
        runs = []
        for i, sh, srcs in hits:
            if runs and i == runs[-1][1] + 1 and set(srcs) & set(runs[-1][3]):
                runs[-1][1] = i
                runs[-1][2] += " " + sh.split()[-1]
            else:
                runs.append([i, i, sh, srcs])
        print(f"== {tex}: {len(runs)} Treffer (n={args.n}, {nsrc} Vergleichsdateien)")
        for start, end, text, srcs in runs:
            print(f"  Zeile {line_of[start]}–{line_of[end + args.n - 1]}: \"{text}\"  <- {', '.join(srcs[:4])}")
        total_hits += len(runs)
    return 1 if total_hits else 0


if __name__ == "__main__":
    sys.exit(main())
