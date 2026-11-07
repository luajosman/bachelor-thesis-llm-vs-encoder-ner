#!/usr/bin/env python3
"""Anker (Textanfänge) in einer LaTeX-Datei auf aktuelle Zeilennummern auflösen.

Aufruf:
  python3 resolve_anchors.py KAPITEL.tex "Anker eins" "Anker zwei" …
  python3 resolve_anchors.py KAPITEL.tex --from anker.txt      # eine Zeile je Anker
  python3 resolve_anchors.py KAPITEL.tex --from-md liste.md    # alle „…“-Zitate der Markdown-Datei

Vergleich nach Whitespace-Normalisierung; „…“ am Ende wird ignoriert. Ausgabe:
Zeile | „nicht gefunden“ | „mehrdeutig (Zeilen …)“.
"""
from __future__ import annotations

import argparse
import re
import sys


def norm(s: str) -> str:
    s = s.replace("…", "").replace("„", "").replace("“", "").replace("„", "").replace("“", "")
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("anchors", nargs="*")
    ap.add_argument("--from", dest="from_file")
    ap.add_argument("--from-md", dest="from_md")
    args = ap.parse_args()

    anchors = list(args.anchors)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as fh:
            anchors += [l.rstrip("\n") for l in fh if l.strip()]
    if args.from_md:
        with open(args.from_md, encoding="utf-8") as fh:
            md = fh.read()
        anchors += re.findall(r"„([^“]{12,})“", md)

    with open(args.tex, encoding="utf-8") as fh:
        lines = [norm(l) for l in fh]
    joined = [(i + 1, lines[i] + " " + (lines[i + 1] if i + 1 < len(lines) else "")) for i in range(len(lines))]

    for a in anchors:
        key = norm(a)
        # erstes Segment bis zu einem " / " (Mehrfachanker) nehmen
        key = key.split(" / ")[0].strip()
        if len(key) < 8:
            print(f"?? zu kurz: {a}")
            continue
        hits = [no for no, text in joined if key in text]
        hits = [h for i, h in enumerate(hits) if i == 0 or h != hits[i - 1] + 1]
        if not hits:
            # Rückfall: nur die ersten sechs Wörter
            short = " ".join(key.split()[:6])
            hits = [no for no, text in joined if short in text]
            hits = [h for i, h in enumerate(hits) if i == 0 or h != hits[i - 1] + 1]
            tag = " (Kurzanker)" if hits else ""
        else:
            tag = ""
        if not hits:
            print(f"-- nicht gefunden: {a[:70]}")
        elif len(hits) > 1:
            print(f"?? mehrdeutig {hits}: {a[:70]}")
        else:
            print(f"{hits[0]:5d}{tag}: {a[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
