#!/usr/bin/env python3
"""Lange Sätze im Fließtext eines Kapitels (Captions/Tabellen ausgenommen).

Aufruf: python3 sentences.py KAPITEL.tex [--min 50] [--stats]
Ausgabe: Startzeile des Absatzes, Wortzahl, Satzanfang.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texclean import load_tex, paragraphs, split_sentences, words  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="+")
    ap.add_argument("--min", type=int, default=50, help="Schwelle in Wörtern (Standard 50)")
    ap.add_argument("--stats", action="store_true", help="nur Statistik ausgeben")
    args = ap.parse_args()
    for tex in args.tex:
        lines = load_tex(tex)
        n_sent, n_words, over40, over50, longest = 0, 0, 0, 0, []
        for start, text in paragraphs(lines):
            for s in split_sentences(text):
                w = len(words(s))
                if w == 0:
                    continue
                n_sent += 1
                n_words += w
                over40 += w > 40
                over50 += w > 50
                if w > args.min:
                    longest.append((start, w, s[:90]))
        avg = n_words / n_sent if n_sent else 0
        print(f"== {tex}: {n_sent} Sätze, Ø {avg:.1f} W/Satz, >40 W: {over40}, >50 W: {over50}")
        if not args.stats:
            for start, w, head in longest:
                print(f"  ab Zeile {start}: {w} W  \"{head}…\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
