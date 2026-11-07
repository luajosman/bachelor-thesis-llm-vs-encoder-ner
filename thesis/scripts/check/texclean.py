"""Gemeinsame Hilfsfunktionen: LaTeX-Quelltext in Fließtext mit Zeilenzuordnung umwandeln.

Verwendet von overlap.py, sentences.py, words.py und numbers.py.
Kein Anspruch auf vollständiges LaTeX-Parsing; es geht um Prüfheuristiken.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FLOAT_ENVS = (
    "table", "table*", "tabular", "tabularx", "longtable", "figure", "figure*",
    "tikzpicture", "axis", "lstlisting", "equation", "equation*", "align",
    "align*", "verbatim", "minipage",
)

# Abkürzungen, nach denen ein Punkt kein Satzende ist.
ABBREVIATIONS = {
    "z", "b", "u", "a", "bzw", "ca", "vgl", "abschn", "tab", "abb", "nr", "s",
    "d", "h", "etc", "evtl", "ggf", "inkl", "max", "min", "mio", "mrd", "sog",
    "usw", "al", "et", "vs", "fig", "eq", "gl", "kap", "o", "ä", "engl", "dt",
    "jh", "bd", "hrsg", "aufl", "ed", "eds", "pp", "p", "no", "vol", "std",
}


@dataclass
class Line:
    number: int
    text: str


def strip_comments(line: str) -> str:
    """Entfernt %-Kommentare, lässt \\% stehen."""
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if ch == "%":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def drop_environments(lines: list[Line], envs=FLOAT_ENVS) -> list[Line]:
    """Entfernt die genannten Umgebungen (inkl. Captions)."""
    out: list[Line] = []
    depth = 0
    begin_re = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in envs) + r")\}")
    end_re = re.compile(r"\\end\{(" + "|".join(re.escape(e) for e in envs) + r")\}")
    for ln in lines:
        opens = len(begin_re.findall(ln.text))
        closes = len(end_re.findall(ln.text))
        if depth == 0 and opens == 0:
            out.append(ln)
        depth += opens - closes
        if depth < 0:
            depth = 0
    return out


_CMD_DROP = re.compile(
    r"\\(cite|textcite|parencite|citep|citet|ref|eqref|label|autoref|cref|input|include|"
    r"includegraphics|url|href|vspace|hspace|setlength|footnotemark|footnotetext|"
    r"addtocounter|setcounter|pagebreak|newpage|clearpage|noindent|centering|"
    r"acro|acrodef)\*?(\[[^\]]*\])*\{[^{}]*\}"
)
_CMD_KEEP_ARG = re.compile(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])*\{")
_CMD_BARE = re.compile(r"\\[a-zA-Z@]+\*?")
_MATH_INLINE = re.compile(r"\$[^$]*\$")


def clean_line(text: str) -> str:
    """Macht aus einer LaTeX-Zeile Fließtext (Befehle entfernt, Argumente erhalten)."""
    t = strip_comments(text)
    t = t.replace("\\,", " ").replace("~", " ").replace("\\%", "%")
    t = t.replace("{,}", ",")
    t = _MATH_INLINE.sub(" ", t)
    for _ in range(3):
        t = _CMD_DROP.sub(" ", t)
    t = _CMD_KEEP_ARG.sub("", t)
    t = _CMD_BARE.sub(" ", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("--", "–")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def load_tex(path: str, drop_floats: bool = True) -> list[Line]:
    with open(path, encoding="utf-8") as fh:
        raw = [Line(i + 1, s.rstrip("\n")) for i, s in enumerate(fh)]
    if drop_floats:
        raw = drop_environments(raw)
    return [Line(ln.number, clean_line(ln.text)) for ln in raw]


def paragraphs(lines: list[Line]) -> list[tuple[int, str]]:
    """Fasst Zeilen zu Absätzen zusammen; liefert (Startzeile, Text)."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = None
    for ln in lines:
        if not ln.text:
            if buf:
                out.append((start, " ".join(buf)))
                buf, start = [], None
            continue
        if start is None:
            start = ln.number
        buf.append(ln.text)
    if buf:
        out.append((start, " ".join(buf)))
    return out


def split_sentences(text: str) -> list[str]:
    """Trennt Sätze heuristisch (Abkürzungen, Ordinalzahlen, Dezimalzahlen geschützt)."""
    pieces: list[str] = []
    buf = []
    tokens = text.split(" ")
    for i, tok in enumerate(tokens):
        buf.append(tok)
        if tok.endswith((".", "!", "?", ".)", ".“", ".\"")):
            core = tok.rstrip(".!?)“\"").lower()
            core = core.split("(")[-1]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if core in ABBREVIATIONS:
                continue
            if core.isdigit() and nxt and nxt[:1].islower():
                continue  # Ordinalzahl vor Kleinwort ("3. Epoche" selten, aber sicher)
            if nxt and nxt[:1].islower():
                continue
            pieces.append(" ".join(buf))
            buf = []
    if buf:
        pieces.append(" ".join(buf))
    return [p.strip() for p in pieces if p.strip()]


def words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text) if re.search(r"[A-Za-zÄÖÜäöüß0-9]", w)]


def normalize_tokens(text: str) -> list[str]:
    """Kleinbuchstaben, nur Wortzeichen; für Shingle-Vergleiche."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9äöüß]+", " ", text)
    return text.split()
