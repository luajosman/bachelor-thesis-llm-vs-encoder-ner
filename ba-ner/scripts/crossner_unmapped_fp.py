"""False positives on CrossNER entities of unmapped types (CPU only, thesis Section 5.7.4).

CrossNER is scored on PER, ORG and LOC; reference spans of all other native types (misc, event,
award, conference, ...) count as O, so a predicted PER/ORG/LOC span on such an entity is a false
positive.  For DeBERTa-v3-large (256-token CrossNER run) and Qwen3.5-27B QLoRA (1,024 tokens),
seeds 42/123/456, this script counts the pooled false positives that coincide with (exact) or
overlap (partial) an entity of an unmapped native type and recomputes pooled micro-F1 with those
false positives left out, as a sensitivity check of the type mapping.

    python scripts/crossner_unmapped_fp.py

Result on 8.11.2026 (means over seeds): recall 85.5 for both models, precision 88.8 (LLM) vs 84.5;
false positives 856 vs 1,243, of which on unmapped entities 259 exact + 194 partial vs 344 + 408;
gap LLM - encoder 2.13 points, without exact ones 1.73, without exact and partial ones 0.54.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAPPED = {"person", "politician", "scientist", "writer", "musicalartist", "researcher",
          "organisation", "politicalparty", "university", "band", "location", "country"}
DOMAINS = ("ai", "literature", "music", "politics", "science")
RUNS = {
    "Qwen3.5-27B (QLoRA, 1024)": [f"crossner_output_limit_1024/qwen35-27b-qlora-3ep{s}" for s in ("", "-seed123", "-seed456")],
    "DeBERTa-v3-large": [f"crossner/deberta-v3-large-canonical{s}" for s in ("", "-seed123", "-seed456")],
}


def spans(tags) -> set:
    """Strict IOB2 entity spans (start, end_exclusive, type)."""
    result, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, label = tag.partition("-")
        if start is not None and not (prefix == "I" and label == kind):
            result.add((start, i, kind))
            start, kind = None, None
        if prefix == "B":
            start, kind = i, label
    return result


def f1(tp: int, fp: int, fn: int) -> tuple:
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 100 * 2 * precision * recall / (precision + recall), 100 * precision, 100 * recall


def count(run_dir: str) -> dict:
    c = dict(tp=0, fp=0, fn=0, exact=0, partial=0)
    for domain in DOMAINS:
        for record in json.loads((RESULTS / run_dir / domain / "test_predictions.json").read_text(encoding="utf-8")):
            gold, pred = spans(record["gold_bio"]), spans(record["pred_bio"])
            unmapped = [s for s in spans(record["native_bio"]) if s[2] not in MAPPED]
            c["tp"] += len(pred & gold)
            c["fn"] += len(gold - pred)
            for s in pred - gold:
                c["fp"] += 1
                if any(u[0] == s[0] and u[1] == s[1] for u in unmapped):
                    c["exact"] += 1
                elif any(u[0] < s[1] and s[0] < u[1] for u in unmapped):
                    c["partial"] += 1
    return c


def main() -> int:
    scores = {}
    for name, dirs in RUNS.items():
        rows = [count(d) for d in dirs]
        scores[name] = [(f1(r["tp"], r["fp"], r["fn"])[0], f1(r["tp"], r["fp"] - r["exact"], r["fn"])[0],
                         f1(r["tp"], r["fp"] - r["exact"] - r["partial"], r["fn"])[0]) for r in rows]
        mean = lambda key: statistics.fmean(r[key] for r in rows)
        p, r_ = statistics.fmean(f1(r["tp"], r["fp"], r["fn"])[1] for r in rows), statistics.fmean(f1(r["tp"], r["fp"], r["fn"])[2] for r in rows)
        print(f"{name}: P {p:.1f} R {r_:.1f} | FP {mean('fp'):.0f}, exact {mean('exact'):.0f}, partial {mean('partial'):.0f}")
    llm, enc = scores.values()
    for i, label in enumerate(("as scored", "without exact", "without exact and partial")):
        print(f"gap {label}: {statistics.fmean(a[i] - b[i] for a, b in zip(llm, enc)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
