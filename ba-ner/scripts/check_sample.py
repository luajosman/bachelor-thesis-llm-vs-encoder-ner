"""Blind manual check of disagreements with the MultiNERD silver reference (CPU only).

Compares DeBERTa-v3-large (canonical, seed 42) with Qwen3.5-27B QLoRA (seed 42, merged 1,024-token
predictions) on the English test split and draws three strata of span-level cases:

  S1  span in reference and encoder, missing from the LLM (the encoder agrees with the reference)
  S2  span in reference and LLM, missing from the encoder (the LLM agrees with the reference)
  S3  span predicted identically by both models, missing from the reference

Each case shows the sentence and two variants of the conflict region (reference side vs. model side)
in random order as A and B, so the annotator does not know which variant is the reference.  Spans
follow strict IOB2 (an entity starts with B-; an orphan I- is no entity), as in the evaluation.

    python scripts/check_sample.py build      # writes cases.json, key.json, population.json
    python scripts/check_sample.py evaluate annotations.json   # writes summary.json

Outputs go to results/check_sample/ (git-ignored); the key is reproducible from the seed.  The blind
annotation page is scripts/check_sample_page.html with __CASES__ replaced by cases.json; it never
contains the key.  Judgments exported from the page map case id -> A | B | both_wrong | undecidable.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "results/multinerd/deberta-v3-large-canonical/test_predictions.json"
LLM = ROOT / "results/multinerd_output_limit_1024/qwen35-27b-qlora-3ep/test_predictions.json"
OUT = ROOT / "results/check_sample"
SEED = 20261109
PER_STRATUM = 60
REPEATS = 30
NOISY = {"ANIM", "BIO", "FOOD", "PLANT"}      # frequently confused per Tedeschi & Navigli (2022, Tab. 5)
FREQUENT = {"PER", "ORG", "LOC"}
JUDGMENTS = ("A", "B", "both_wrong", "undecidable")


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


def type_group(kind: str) -> str:
    return "noisy" if kind in NOISY else "frequent" if kind in FREQUENT else "other"


def load():
    encoder = json.loads(ENCODER.read_text(encoding="utf-8"))
    llm = json.loads(LLM.read_text(encoding="utf-8"))
    if len(encoder) != len(llm):
        raise ValueError("prediction files differ in length")
    rows = []
    for i, (e, l) in enumerate(zip(encoder, llm)):
        if e["tokens"] != l["tokens"] or e["gold"] != l["gold_bio"]:
            raise ValueError(f"sentence {i}: tokens or reference differ between the two files")
        rows.append({"tokens": e["tokens"], "gold": spans(e["gold"]), "enc": spans(e["pred"]),
                     "llm": spans(l["pred_bio"]), "regenerated": bool(l.get("regenerated"))})
    return rows


def overlapping(pool: set, start: int, end: int) -> set:
    return {s for s in pool if s[0] < end and start < s[1]}


def make_case(index: int, row: dict, stratum: str, span: tuple) -> dict:
    ref_side, model_side = {
        "S1": (row["gold"], row["llm"]),     # reference (= encoder) vs LLM
        "S2": (row["gold"], row["enc"]),     # reference (= LLM) vs encoder
        "S3": (row["gold"], row["llm"]),     # reference vs both models (LLM = encoder on this span)
    }[stratum]
    start, end = span[0], span[1]
    region = {span} | overlapping(ref_side | model_side, start, end)
    start, end = min(s[0] for s in region), max(s[1] for s in region)
    reference = sorted(overlapping(ref_side, start, end))
    model = sorted(overlapping(model_side, start, end))
    return {"sentence": index, "stratum": stratum, "type": span[2], "type_group": type_group(span[2]),
            "span": list(span), "region": [start, end], "reference": [list(s) for s in reference],
            "model": [list(s) for s in model], "regenerated": row["regenerated"]}


def population(rows) -> dict:
    strata = {"S1": [], "S2": [], "S3": []}
    for i, row in enumerate(rows):
        for span in sorted((row["gold"] & row["enc"]) - row["llm"]):
            strata["S1"].append(make_case(i, row, "S1", span))
        for span in sorted((row["gold"] & row["llm"]) - row["enc"]):
            strata["S2"].append(make_case(i, row, "S2", span))
        for span in sorted((row["enc"] & row["llm"]) - row["gold"]):
            strata["S3"].append(make_case(i, row, "S3", span))
    return strata


def proportional_sample(cases: list, n: int, rng: random.Random) -> list:
    """Self-weighting sample: allocation proportional to the type groups (largest remainder)."""
    groups = {}
    for case in cases:
        groups.setdefault(case["type_group"], []).append(case)
    quotas = {g: n * len(c) / len(cases) for g, c in groups.items()}
    alloc = {g: int(q) for g, q in quotas.items()}
    for g in sorted(quotas, key=lambda g: quotas[g] - alloc[g], reverse=True)[: n - sum(alloc.values())]:
        alloc[g] += 1
    return [case for g in sorted(groups) for case in rng.sample(groups[g], alloc[g])]


def build() -> int:
    rows = load()
    strata = population(rows)
    rng = random.Random(SEED)
    chosen = [case for name in ("S1", "S2", "S3") for case in proportional_sample(strata[name], PER_STRATUM, rng)]
    rng.shuffle(chosen)
    cases, key = [], {}
    for number, case in enumerate(chosen, 1):
        case_id = f"F{number:03d}"
        reference_first = rng.random() < 0.5
        tokens = rows[case["sentence"]]["tokens"]
        variants = [case["reference"], case["model"]] if reference_first else [case["model"], case["reference"]]
        cases.append({"id": case_id, "tokens": tokens, "region": case["region"],
                      "A": [{"start": s, "end": e, "type": t} for s, e, t in variants[0]],
                      "B": [{"start": s, "end": e, "type": t} for s, e, t in variants[1]]})
        key[case_id] = {**case, "reference_is": "A" if reference_first else "B", "repeat_of": None}
    for number, original in enumerate(rng.sample(cases, REPEATS), 1):
        case_id = f"W{number:02d}"
        swap = rng.random() < 0.5
        cases.append({**original, "id": case_id, "A": original["B"] if swap else original["A"],
                      "B": original["A"] if swap else original["B"], "round": 2})
        base = key[original["id"]]
        key[case_id] = {**base, "reference_is": ("B" if base["reference_is"] == "A" else "A") if swap else base["reference_is"],
                        "repeat_of": original["id"], "swapped": swap}
    repeats = cases[len(chosen):]
    rng.shuffle(repeats)
    cases = cases[: len(chosen)] + repeats
    for case in cases:
        case.setdefault("round", 1)
    summary = {name: {"cases": len(items), "by_type_group": dict(Counter(c["type_group"] for c in items)),
                      "by_type": dict(Counter(c["type"] for c in items).most_common()),
                      "in_regenerated_sentences": sum(c["regenerated"] for c in items)}
               for name, items in strata.items()}
    summary["sample"] = {name: dict(Counter(key[c["id"]]["type_group"] for c in cases
                                            if c["round"] == 1 and key[c["id"]]["stratum"] == name))
                         for name in ("S1", "S2", "S3")}
    summary["seed"], summary["per_stratum"], summary["repeats"] = SEED, PER_STRATUM, REPEATS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "population.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["cases"] if "cases" in v else v) for k, v in summary.items() if k in ("S1", "S2", "S3")}))
    print("sample by type group:", json.dumps(summary["sample"]))
    return 0


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)]


def verdict(judgment: str, reference_is: str) -> str:
    if judgment in ("both_wrong", "undecidable"):
        return judgment
    return "reference_correct" if judgment == reference_is else "model_correct"


def cohen_kappa(pairs) -> float | None:
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[x] * second[x] for x in labels) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def evaluate(path: str) -> int:
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    annotations = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(value not in JUDGMENTS for value in annotations.values()):
        raise ValueError(f"judgments must be one of {JUDGMENTS}")
    verdicts = {cid: verdict(annotations[cid], key[cid]["reference_is"]) for cid in key if cid in annotations}
    result = {"annotated": len(verdicts), "expected": len(key), "strata": {}}
    for name in ("S1", "S2", "S3"):
        ids = [cid for cid in key if key[cid]["stratum"] == name and key[cid]["repeat_of"] is None and cid in verdicts]
        counts = Counter(verdicts[cid] for cid in ids)
        result["strata"][name] = {
            "n": len(ids),
            **{label: {"count": counts.get(label, 0), "share": counts.get(label, 0) / len(ids) if ids else None,
                       "wilson95": wilson(counts.get(label, 0), len(ids))}
               for label in ("reference_correct", "model_correct", "both_wrong", "undecidable")},
            "by_type_group": {g: dict(Counter(verdicts[cid] for cid in ids if key[cid]["type_group"] == g))
                              for g in ("noisy", "frequent", "other")},
        }
    pairs = [(verdicts[key[cid]["repeat_of"]], verdicts[cid]) for cid in key
             if key[cid]["repeat_of"] is not None and cid in verdicts and key[cid]["repeat_of"] in verdicts]
    result["intra_rater"] = {"pairs": len(pairs), "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                             "cohen_kappa": cohen_kappa(pairs)}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        raise SystemExit(build())
    if len(sys.argv) == 3 and sys.argv[1] == "evaluate":
        raise SystemExit(evaluate(sys.argv[2]))
    print(__doc__)
    raise SystemExit(2)
