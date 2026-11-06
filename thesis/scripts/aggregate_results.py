#!/usr/bin/env python3
"""Aggregiert die Laufartefakte unter ba-ner/results/ zu den Tabellen,
Abbildungsdaten und Kennzahlen fuer Kapitel 5 der Bachelorarbeit.

Aufruf aus dem Repo-Stamm:
    python3 thesis/scripts/aggregate_results.py
Benoetigt PyYAML und numpy (pip3 install --user pyyaml numpy).

Eingaben:  ba-ner/results/{multinerd,crossner,crossner_output_limit_1024,multinerd_output_limit_1024}
           ba-ner/results/slurm_accounting_2026-08.csv  (sacct-Export; fehlt er, Hochrechnung)
           ba-ner/configs/crossner_overlap_exclusions.yaml
Ausgaben:  thesis/generated/summary.json  (alle Kennzahlen)
           thesis/generated/tab-*.tex     (tabular-Fragmente fuer \\input)
           thesis/generated/fig-*.dat     (Datentabellen fuer pgfplots)

Alle F1-, Precision- und Recall-Werte werden strikt auf Entitaetsebene nach
dem IOB2-Schema berechnet (wie seqeval im strikten Modus) und gegen die
gespeicherten Werte der Laeufe geprueft. Mittelwerte und
Stichprobenstandardabweichungen (ddof=1) ueber die Seeds 42, 123, 456.
Konfidenzintervalle: gepaartes Satz-Bootstrap (gleiche Stichprobe fuer alle
Laeufe, feste Zufallssaat, daher byteidentisch reproduzierbar).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "ba-ner" / "results"
CFG = ROOT / "ba-ner" / "configs"
OUT = ROOT / "thesis" / "generated"

TYPES = ["PER", "ORG", "LOC", "CEL", "DIS", "EVE", "MEDIA",
         "ANIM", "BIO", "FOOD", "PLANT", "INST", "MYTH", "TIME", "VEHI"]
TYPE_GROUPS = [
    ("Zuverl\\\"assig annotierte Typen", ["PER", "ORG", "LOC", "CEL", "DIS", "EVE", "MEDIA"]),
    ("H\\\"aufig verwechselte oder unvollst\\\"andig verlinkte Typen", ["ANIM", "BIO", "FOOD", "PLANT"]),
    ("\\\"Ubrige Typen", ["INST", "MYTH", "TIME", "VEHI"]),
]
SEEDS = [42, 123, 456]

TRAINED = {
    "deberta-base": dict(label="DeBERTa-v3-base", family="DeBERTa-v3", regime="Fine-Tuning", size="S",
                         params=184e6, gpu="H100",
                         runs=["deberta-v3-base-canonical", "deberta-v3-base-canonical-seed123",
                               "deberta-v3-base-canonical-seed456"]),
    "deberta-large": dict(label="DeBERTa-v3-large", family="DeBERTa-v3", regime="Fine-Tuning", size="M",
                          params=434e6, gpu="H100",
                          runs=["deberta-v3-large-canonical", "deberta-v3-large-canonical-seed123",
                                "deberta-v3-large-canonical-seed456"]),
    "qwen08-qlora": dict(label="Qwen3.5-0.8B", family="Qwen3.5", regime="QLoRA", size="S",
                         params=0.752e9, gpu="H100",
                         runs=["qwen35-08b-qlora-canonical", "qwen35-08b-qlora-canonical-seed123",
                               "qwen35-08b-qlora-canonical-seed456"]),
    "qwen4-qlora": dict(label="Qwen3.5-4B", family="Qwen3.5", regime="QLoRA", size="M",
                        params=4.206e9, gpu="A100",
                        runs=["qwen35-4b-qlora-canonical", "qwen35-4b-qlora-canonical-seed123",
                              "qwen35-4b-qlora-canonical-seed456"]),
    "qwen27-qlora": dict(label="Qwen3.5-27B", family="Qwen3.5", regime="QLoRA", size="L",
                         params=26.896e9, gpu="NVL",
                         runs=["qwen35-27b-qlora-3ep", "qwen35-27b-qlora-3ep-seed123",
                               "qwen35-27b-qlora-3ep-seed456"]),
}
ZEROSHOT = {
    "qwen08-zs": dict(label="Qwen3.5-0.8B", family="Qwen3.5", regime="Zero-Shot", size="S",
                      params=0.752e9, gpu="A100", run="qwen35-08b-zeroshot"),
    "qwen4-zs": dict(label="Qwen3.5-4B", family="Qwen3.5", regime="Zero-Shot", size="M",
                     params=4.206e9, gpu="A100", run="qwen35-4b-zeroshot"),
    "qwen27-zs": dict(label="Qwen3.5-27B", family="Qwen3.5", regime="Zero-Shot", size="L",
                      params=26.896e9, gpu="NVL", run="qwen35-27b-zeroshot"),
}
TABLE_ORDER = ["deberta-base", "deberta-large", "qwen08-zs", "qwen4-zs", "qwen27-zs",
               "qwen08-qlora", "qwen4-qlora", "qwen27-qlora"]
CROSS_RUNS = {
    "deberta-large": ["deberta-v3-large-canonical", "deberta-v3-large-canonical-seed123",
                      "deberta-v3-large-canonical-seed456"],
    "qwen27-qlora": ["qwen35-27b-qlora-3ep", "qwen35-27b-qlora-3ep-seed123", "qwen35-27b-qlora-3ep-seed456"],
}
CROSS_AGG_NAMES = {"deberta-large": "deberta-v3-large", "qwen27-qlora": "qwen35-27b-qlora-3ep"}
DOMAINS = [("ai", "K\\\"unstliche Intelligenz"), ("literature", "Literatur"), ("music", "Musik"),
           ("politics", "Politik"), ("science", "Naturwissenschaften")]
COARSE = ["PER", "ORG", "LOC"]
TOTAL_STEPS_QLORA = 24615                       # Optimierungsschritte je QLoRA-Lauf (3 Epochen)
BOOT_B, BOOT_SEED, BOOT_CHUNK = 2000, 42, 500   # Satz-Bootstrap: Replikate, Zufallssaat, Blockgroesse
LIMIT_ROOT = RES / "multinerd_output_limit_1024"
LIMIT_GROUPS = {"qwen08-qlora-1024": "qwen08-qlora", "qwen4-qlora-1024": "qwen4-qlora",
                "qwen27-qlora-1024": "qwen27-qlora"}    # Schluessel -> Basisgruppe (Laufnamen identisch)
SACCT = RES / "slurm_accounting_2026-08.csv"
SACCT_MODEL = {"deberta-base": "deb-base", "deberta-large": "deb-large", "qwen08-qlora": "qwen08b",
               "qwen4-qlora": "qwen4b", "qwen27-qlora": "qwen27b-3ep"}   # Gruppe -> Modell-Token im Slurm-Jobnamen
SACCT_JOB = re.compile(r"^ner-(?P<model>.+)-s(?P<seed>\d+)-(?P<phase>train|infer|eval)$")
MONTHS = ["Januar", "Februar", "M\\\"arz", "April", "Mai", "Juni", "Juli", "August", "September",
          "Oktober", "November", "Dezember"]
NUMBER_WORDS = {1: "Eine", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "F\\\"unf", 6: "Sechs"}


# --------------------------------------------------------------------------- Hilfsfunktionen
def load_yaml(path: Path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def entities(seq):
    """Strikte IOB2-Extraktion: Liste von (start, end_exklusiv, typ)."""
    ents = []
    cur = None
    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if cur:
                ents.append(tuple(cur))
            cur = [i, i + 1, tag[2:]]
        elif tag.startswith("I-") and cur and cur[2] == tag[2:] and cur[1] == i:
            cur[1] = i + 1
        else:
            if cur:
                ents.append(tuple(cur))
            cur = None
    if cur:
        ents.append(tuple(cur))
    return ents


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(records, gold_key, pred_key, types=None):
    """Mikro-Werte gesamt und je Typ (strikt)."""
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for rec in records:
        gold = set(entities(rec[gold_key]))
        pred = set(entities(rec[pred_key]))
        if types:
            gold = {e for e in gold if e[2] in types}
            pred = {e for e in pred if e[2] in types}
        for e in gold:
            support[e[2]] += 1
        for e in gold & pred:
            tp[e[2]] += 1
        for e in pred - gold:
            fp[e[2]] += 1
        for e in gold - pred:
            fn[e[2]] += 1
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_type = {t: dict(zip(("precision", "recall", "f1"), prf(tp[t], fp[t], fn[t])), support=support[t])
                for t in (types or TYPES)}
    return overall, per_type


def sentence_counts(records, gold_key, pred_key, allowed_types=None):
    """Je Satz (tp, fp, fn) gesamt und je Typ (strikt, wie score()) fuer das gepaarte Satz-Bootstrap.
    Rueckgabe: micro int32[n, 3], types int32[n, len(TYPES), 3]."""
    idx = {t: i for i, t in enumerate(TYPES)}
    micro = np.zeros((len(records), 3), dtype=np.int32)
    types = np.zeros((len(records), len(TYPES), 3), dtype=np.int32)
    for i, rec in enumerate(records):
        gold = set(entities(rec[gold_key]))
        pred = set(entities(rec[pred_key]))
        if allowed_types is not None:
            gold = {e for e in gold if e[2] in allowed_types}
            pred = {e for e in pred if e[2] in allowed_types}
        for j, group in enumerate((gold & pred, pred - gold, gold - pred)):
            micro[i, j] = len(group)
            for e in group:
                if e[2] in idx:
                    types[i, idx[e[2]], j] += 1
    return micro, types


def error_profile(records, gold_key, pred_key):
    """Ordnet jede Referenzentitaet genau einer Fehlerklasse zu."""
    counts = Counter()
    confusion = Counter()
    boundary_by_type = Counter()
    n_gold = 0
    additions = 0
    additions_by_type = Counter()
    for rec in records:
        gold = entities(rec[gold_key])
        pred = entities(rec[pred_key])
        pset = set(pred)
        matched = set()
        for ge in gold:
            n_gold += 1
            if ge in pset:
                counts["korrekt"] += 1
                matched.add(ge)
                continue
            overlapping = [pe for pe in pred if pe[0] < ge[1] and ge[0] < pe[1]]
            if not overlapping:
                counts["auslassung"] += 1
                continue
            same_span = [pe for pe in overlapping if pe[0] == ge[0] and pe[1] == ge[1]]
            if same_span:
                counts["typfehler"] += 1
                confusion[(ge[2], same_span[0][2])] += 1
                matched.update(same_span)
            else:
                same_type = [pe for pe in overlapping if pe[2] == ge[2]]
                if same_type:
                    counts["grenzfehler"] += 1
                    boundary_by_type[ge[2]] += 1
                    matched.update(same_type)
                else:
                    counts["grenz_typ"] += 1
                    confusion[(ge[2], overlapping[0][2])] += 1
                    matched.update(overlapping)
        for pe in pred:
            if pe in matched:
                continue
            if not any(pe[0] < ge[1] and ge[0] < pe[1] for ge in gold):
                additions += 1
                additions_by_type[pe[2]] += 1
    return dict(counts=dict(counts), n_gold=n_gold, additions=additions,
                confusion=confusion, boundary_by_type=boundary_by_type,
                additions_by_type=additions_by_type)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.fmean(values), statistics.stdev(values)


def de(x, nd=1):
    """Deutsche Dezimalschreibweise."""
    return f"{x:.{nd}f}".replace(".", ",")


def de_sep(x, nd=1):
    """Deutsche Dezimalschreibweise mit schmalem Tausendertrenner."""
    return f"{x:,.{nd}f}".replace(",", "\\,").replace(".", ",")


def pm(mean, std, nd=2):
    if std is None:
        return de(mean, nd)
    return f"{de(mean, nd)}\\,$\\pm$\\,{de(std, nd)}"


def signed(x, nd=2):
    s = de(abs(x), nd)
    return ("$+$" if x >= 0 else "$-$") + s


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"  geschrieben: {path.relative_to(ROOT)}")


def fmt_int(x):
    """Gerundete Ganzzahl mit \\, als Tausendertrenner."""
    return f"{round(x):,}".replace(",", "\\,")


def iso(ts):
    return datetime.fromisoformat(ts)


def parse_phase_history(meta):
    """Trainingsstarts, Trainingsabschluss und Inferenzphase aus run_metadata.yaml:phase_history.
    Jede Wiederaufnahme nach einem Slurm-Requeue schreibt einen weiteren training/RUNNING-Eintrag;
    elapsed_seconds des *_COMPLETED-Eintrags ist die Wall-Clock der letzten Zuteilung."""
    hist = (meta or {}).get("phase_history") or []
    starts = [iso(e["timestamp"]) for e in hist if e["phase"] == "training" and e["status"] == "RUNNING"]
    done = [e for e in hist if e["phase"] == "training" and e["status"] == "TRAINING_COMPLETED"]
    inf_start = [iso(e["timestamp"]) for e in hist if e["phase"] == "inference" and e["status"] == "RUNNING"]
    inf_done = [e for e in hist if e["phase"] == "inference" and e["status"] == "INFERENCE_COMPLETED"]
    return dict(
        training_starts=starts,
        training_completed=(iso(done[-1]["timestamp"]), done[-1]["elapsed_seconds"]) if done else None,
        inference=(inf_start[-1] if inf_start else None,
                   iso(inf_done[-1]["timestamp"]) if inf_done else None,
                   inf_done[-1]["elapsed_seconds"] if inf_done else None))


def load_sacct(path: Path = SACCT):
    """Slurm-Abrechnung (sacct -D -X -P, pipe-getrennt): Zuteilungen je (Modell-Token, Seed, Phase).
    Requeue-Zuteilungen desselben Jobs erscheinen als mehrere Zeilen; Zeilen unter 60 s sind die
    sofort gescheiterten automatischen Neustarts und werden ignoriert. None, wenn die Datei fehlt."""
    if not path.exists():
        return None
    rows = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            m = SACCT_JOB.match(row["JobName"])
            if not m:
                continue
            elapsed = int(row["ElapsedRaw"] or 0)
            if elapsed < 60:
                continue
            rows.setdefault((m["model"], int(m["seed"]), m["phase"]), []).append(dict(
                job_id=row["JobID"], state=row["State"], start=row["Start"], end=row["End"],
                elapsed_seconds=elapsed, partition=row["Partition"], node=row["NodeList"]))
    for lst in rows.values():
        lst.sort(key=lambda r: r["start"])
    return rows


def sacct_for(sacct, key, seed):
    """Trainings- und Inferenzzuteilungen eines Laufs (Gruppe, Seed) aus der Slurm-Abrechnung."""
    if sacct is None or key not in SACCT_MODEL:
        return None
    tok = SACCT_MODEL[key]
    return dict(train=sacct.get((tok, seed, "train"), []), infer=sacct.get((tok, seed, "infer"), []))


# --------------------------------------------------------------------------- MultiNERD-Laeufe
def analyse_multinerd_run(run_dir: Path, is_llm: bool, seed=None, sacct=None):
    preds = load_json(run_dir / "test_predictions.json")
    gold_key, pred_key = ("gold_bio", "pred_bio") if is_llm else ("gold", "pred")
    overall, per_type = score(preds, gold_key, pred_key)
    im_path = run_dir / "inference_metrics.yaml"
    im = load_yaml(im_path) if im_path.exists() else {}
    if seed is not None and im.get("seed") is not None and int(im["seed"]) != seed:
        sys.exit(f"ABBRUCH: Seed {run_dir.name}: erwartet {seed}, gespeichert {im['seed']}")
    stored_f1 = im.get("test_f1", im.get("f1"))
    if stored_f1 is None:
        print(f"  Hinweis: {run_dir.name} ohne gespeicherten F1; Selbstpruefung uebersprungen")
    elif abs(stored_f1 - overall[2]) > 1e-9:
        sys.exit(f"ABBRUCH: F1-Abweichung in {run_dir.name}: eigen {overall[2]:.8f}, gespeichert {stored_f1:.8f}")
    ev_path = run_dir / "evaluation_metrics.yaml"
    if ev_path.exists():
        ev = load_yaml(ev_path)
        for t in TYPES:
            if abs(ev["per_entity"][t]["f1"] - per_type[t]["f1"]) > 1e-9:
                sys.exit(f"ABBRUCH: Typ-F1-Abweichung {run_dir.name}/{t}")
    macro = statistics.fmean(per_type[t]["f1"] for t in TYPES)
    # Satzzaehler fuer das Bootstrap; ihre Summen muessen die strikten Gesamt- und Typwerte reproduzieren
    sent_micro, sent_types = sentence_counts(preds, gold_key, pred_key)
    if abs(prf(*map(int, sent_micro.sum(0)))[2] - overall[2]) > 1e-12:
        sys.exit(f"ABBRUCH: Satzzaehler {run_dir.name}: Mikro-F1 weicht ab")
    for j, t in enumerate(TYPES):
        if abs(prf(*map(int, sent_types[:, j].sum(0)))[2] - per_type[t]["f1"]) > 1e-12:
            sys.exit(f"ABBRUCH: Satzzaehler {run_dir.name}/{t}: Typ-F1 weicht ab")
    out = dict(name=run_dir.name, precision=overall[0], recall=overall[1], f1=overall[2], macro_f1=macro,
               per_type=per_type, n_sentences=len(preds),
               n_gold=sum(per_type[t]["support"] for t in TYPES),
               latency_mean=im.get("latency_ms_mean"), latency_p50=im.get("latency_ms_p50"),
               latency_p95=im.get("latency_ms_p95"), vram_mb=im.get("vram_peak_mb"),
               throughput=im.get("generation_samples_per_second"), total_params=im.get("total_params"),
               gpu=(im.get("run_metadata", {}).get("cuda", {}).get("devices") or [None])[0],
               commit=im.get("run_metadata", {}).get("git", {}).get("commit"))
    # numpy-Felder und Signaturen bleiben auf dem Laufobjekt (nicht in agg, nicht in summary.json)
    out["sent_micro"], out["sent_types"] = sent_micro, sent_types
    out["sentence_signature"] = hashlib.sha1(json.dumps([r["tokens"] for r in preds]).encode()).hexdigest()
    out["gold_signature"] = hashlib.sha1(json.dumps([r[gold_key] for r in preds]).encode()).hexdigest()
    out["errors"] = error_profile(preds, gold_key, pred_key)
    # Wall-Clock der Testinferenz: Phasenprotokoll (exakt) oder mittlere Latenz x Satzzahl (Schaetzung)
    meta_path = run_dir / "run_metadata.yaml"
    meta = load_yaml(meta_path) if meta_path.exists() else None
    lat = im.get("latency_ms_mean")
    out["inference_seconds_latency_estimate"] = lat * len(preds) / 1000 if lat is not None else None
    phases = parse_phase_history(meta) if meta else None
    if phases and phases["inference"][2] is not None:
        out["inference_seconds"] = phases["inference"][2]
        out["inference_seconds_source"] = "phase_history"
    else:
        out["inference_seconds"] = out["inference_seconds_latency_estimate"]
        out["inference_seconds_source"] = "latency_estimate" if lat is not None else None
    res_path = run_dir / "results.yaml"
    if res_path.exists():
        res = load_yaml(res_path)
        out["train_seconds"] = res.get("train_runtime_seconds")
        out["best_epoch"] = res.get("best_epoch")
        out["val_f1"] = res.get("best_validation_f1", res.get("best_dev_f1"))
        out["epoch_results"] = res.get("epoch_results")
        recov = sorted(run_dir.glob("recovery-*.yaml"))
        # validated_checkpoint der manuellen Wiederherstellung = Wiederaufnahmepunkt der 2. Zuteilung (nur Doku);
        # der protokollierte Trainingsabschnitt beginnt bei results.yaml:resume_checkpoint.
        out["recovery_checkpoint"] = load_yaml(recov[0]).get("validated_checkpoint") if recov else None
        out["resumed"] = bool(res.get("resumed"))
        out["train_lower_bound"] = out["resumed"]
        out["resume_checkpoint"] = Path(res["resume_checkpoint"]).name if res.get("resume_checkpoint") else None
        out["training_profile"] = training_time_profile(run_dir, res, meta, sacct, is_llm)
    if is_llm:
        status = Counter(r["parse_status"] for r in preds)
        failed_no_close = sum(1 for r in preds if r["parse_status"] == "failed"
                              and not r.get("raw_output", "").rstrip().endswith("]"))
        empty = sum(1 for r in preds if not r.get("pred_entities"))   # Saetze ohne uebernommenen Span, egal warum
        # Uebernommene Spans (= B-Tags = Eintraege in pred_entities) und begonnene JSON-Objekte der Rohausgabe
        accepted = sum(len(r.get("pred_entities") or []) for r in preds)
        b_tags = sum(t.startswith("B-") for r in preds for t in r["pred_bio"])
        if accepted != b_tags:
            sys.exit(f"ABBRUCH: Spanzahl {run_dir.name}: pred_entities {accepted} != B-Tags {b_tags}")
        generated = sum(r.get("raw_output", "").count("{") for r in preds)   # inkl. abgeschnittener Objekte (Naeherung)
        gold_empty_ids = {i for i, r in enumerate(preds) if not entities(r["gold_bio"])}
        empty_failed = sum(1 for r in preds if not r.get("pred_entities") and r["parse_status"] == "failed")
        empty_correct = sum(1 for i, r in enumerate(preds) if not r.get("pred_entities")
                            and r["parse_status"] != "failed" and i in gold_empty_ids)
        empty_array = sum(1 for r in preds if r.get("raw_output", "").strip() == "[]")
        unknown = Counter()
        for r in preds:
            if r["parse_diagnostics"].get("unknown_types", 0):
                for t in re.findall(r'"type"\s*:\s*"([^"]*)"', r.get("raw_output", "")):
                    if t not in TYPES:
                        unknown[t] += 1
        failed_recs = [r for r in preds if r["parse_status"] == "failed"]
        lost = sum(len(entities(r["gold_bio"])) for r in failed_recs)
        n_gold_total = sum(len(entities(r["gold_bio"])) for r in preds)
        tp = overall[1] * n_gold_total
        out["truncation"] = dict(failed_sentences=len(failed_recs), lost_gold=lost,
                                 lost_share=lost / n_gold_total,
                                 mean_tokens_failed=(sum(len(r["tokens"]) for r in failed_recs) / len(failed_recs)) if failed_recs else None,
                                 recall_without_failed=tp / (n_gold_total - lost) if n_gold_total > lost else None)
        # Parse-Zaehler aus den Datensaetzen; gespeicherte Werte (kanonische Laeufe) muessen uebereinstimmen
        diag = Counter()
        for r in preds:
            diag.update(r["parse_diagnostics"])
        computed = dict(parse_ok=status["ok"], markdown=status["markdown_stripped"], regex=status["regex_fallback"],
                        failed=status["failed"], failure_rate=status["failed"] / len(preds),
                        wrong_schema=diag["wrong_schema"], invalid_items=diag["invalid_items"],
                        missing_fields=diag["missing_fields"], invalid_offsets=diag["invalid_offsets"],
                        unknown_types=diag["unknown_types"], text_mismatches=diag["text_mismatches"],
                        overlaps=diag["overlaps"])
        im_keys = dict(parse_ok="parse_ok", markdown="parse_markdown_stripped", regex="parse_regex_fallback",
                       failed="parse_failed", failure_rate="parse_failure_rate", wrong_schema="parse_wrong_schema",
                       invalid_items="parse_invalid_items", missing_fields="parse_missing_fields",
                       invalid_offsets="parse_invalid_offsets", unknown_types="parse_unknown_types",
                       text_mismatches="parse_text_mismatches", overlaps="parse_overlaps")
        form = {}
        for k, imk in im_keys.items():
            if imk in im:
                if abs(im[imk] - computed[k]) > (1e-12 if k == "failure_rate" else 0):
                    sys.exit(f"ABBRUCH: Parse-Zaehler {run_dir.name}/{imk}: eigen {computed[k]}, gespeichert {im[imk]}")
                form[k] = im[imk]
            else:
                form[k] = computed[k]
        out["form"] = dict(
            n=len(preds), **form, failed_no_close=failed_no_close, empty=empty,
            accepted_spans=accepted, generated_items=generated, empty_failed=empty_failed,
            empty_correct=empty_correct, empty_other=empty - empty_failed - empty_correct, empty_array=empty_array,
            gold_empty=len(gold_empty_ids),
            status=dict(status), unknown_examples=unknown.most_common(10))
        out["generation_settings"] = im.get("generation_settings")
        out["max_new_tokens"] = im.get("max_new_tokens")
        out["pred_hashes"] = [hashlib.sha1(json.dumps(r["pred_bio"]).encode()).hexdigest() for r in preds]
        if any("regenerated" in r for r in preds):      # Nachinferenz mit hoeherem Ausgabelimit (Step 5)
            idx = [i for i, r in enumerate(preds) if r.get("regenerated")]
            regen = [preds[i] for i in idx]
            sub256 = [dict(gold_bio=r["gold_bio"], pred_bio=r["baseline_256"]["pred_bio"]) for r in regen]
            out["regen"] = dict(
                n_regenerated=len(idx), indices=idx,
                n_limit_reached=sum(1 for r in regen if r.get("token_limit_reached")),
                stop_reasons=dict(Counter(r.get("stop_reason") for r in regen)),
                generated_tokens_mean=statistics.fmean(r["generated_token_count"] for r in regen) if regen else None,
                latency_ms_mean_regenerated=statistics.fmean(r["latency_ms"] for r in regen) if regen else None,
                n_failed_after=status["failed"],
                n_failed_regenerated=sum(1 for r in regen if r["parse_status"] == "failed"),
                n_failed_before_regenerated=sum(1 for r in regen if r["baseline_256"]["parse_status"] == "failed"),
                stored_prefix_match_count=sum(1 for r in regen if r.get("stored_prefix_match")),
                json_complete_count=sum(1 for r in regen if r.get("json_complete")),
                subset_metrics=dict(
                    baseline_256=dict(zip(("precision", "recall", "f1"), score(sub256, "gold_bio", "pred_bio")[0])),
                    output_limit=dict(zip(("precision", "recall", "f1"), score(regen, "gold_bio", "pred_bio")[0]))),
                baseline_pred_hashes={i: hashlib.sha1(json.dumps(preds[i]["baseline_256"]["pred_bio"]).encode()).hexdigest()
                                      for i in idx})
    # Beispiele fuer die Fehlertabelle (kurze Saetze)
    out["examples"] = collect_examples(preds, gold_key, pred_key, is_llm)
    return out


def training_time_profile(run_dir: Path, res, meta, sacct, is_llm: bool):
    """Trainingsdauer eines Laufs aus drei Quellen.
    exakt:     logged_seconds (HF train() des protokollierten, bei Wiederaufnahme nur des letzten Abschnitts),
               last_allocation_seconds und wallclock_span_seconds (Phasenprotokoll),
               sacct_train_seconds (Summe aller Slurm-Zuteilungen des Trainingsjobs; primaere Quelle).
    geschaetzt: seconds_per_step des letzten Abschnitts x Gesamtschritte (estimated_total_seconds)."""
    logged = res.get("train_runtime_seconds")
    resumed = bool(res.get("resumed"))
    resume_step = int(re.search(r"checkpoint-(\d+)", res["resume_checkpoint"]).group(1)) if resumed else 0
    hist = load_json(run_dir / "training_history.json")
    total_steps = max(e["step"] for e in hist if "step" in e)
    if is_llm and total_steps != TOTAL_STEPS_QLORA:
        sys.exit(f"ABBRUCH: Schrittzahl {run_dir.name}: {total_steps} != {TOTAL_STEPS_QLORA}")
    steps_logged = total_steps - resume_step
    sec_per_step = logged / steps_logged
    estimate = sec_per_step * total_steps
    phases = parse_phase_history(meta)
    starts = phases["training_starts"]
    done_ts, last_alloc = phases["training_completed"] or (None, None)
    marks = starts + ([done_ts] if done_ts else [])
    prof = dict(
        resumed=resumed, resume_step=resume_step, total_steps=total_steps, steps_logged=steps_logged,
        logged_seconds=logged, seconds_per_step=sec_per_step, estimated_total_seconds=estimate,
        n_allocations=max(len(starts), 1), allocation_starts=[s.isoformat() for s in starts],
        completed_at=done_ts.isoformat() if done_ts else None,
        allocation_spans_seconds=[(b - a).total_seconds() for a, b in zip(marks, marks[1:])],  # Startabstaende inkl. Wartezeit
        last_allocation_seconds=last_alloc,
        wallclock_span_seconds=(done_ts - starts[0]).total_seconds() if starts and done_ts else None,
        sacct_train_seconds=None, sacct_train_allocations=None, sacct_train_detail=None, sacct_infer_seconds=None,
        exact=["logged_seconds", "last_allocation_seconds", "n_allocations", "wallclock_span_seconds",
               "sacct_train_seconds"],
        estimated=["seconds_per_step", "estimated_total_seconds"])
    if sacct is not None:
        train_rows, infer_rows = sacct.get("train", []), sacct.get("infer", [])
        if train_rows:
            prof["sacct_train_seconds"] = sum(r["elapsed_seconds"] for r in train_rows)
            prof["sacct_train_allocations"] = len(train_rows)
            prof["sacct_train_detail"] = train_rows
            # Selbstpruefung: Slurm-Summe gegen Protokoll (nicht wiederaufgenommen) bzw. Hochrechnung (wiederaufgenommen)
            rel = (abs(prof["sacct_train_seconds"] - estimate) / prof["sacct_train_seconds"] if resumed
                   else abs(prof["sacct_train_seconds"] - logged) / logged)
            if rel >= 0.05:
                print(f"  Hinweis: Slurm-Trainingszeit {run_dir.name} weicht {rel * 100:.1f} % von "
                      f"{'Hochrechnung' if resumed else 'Protokoll'} ab")
            if len(train_rows) != prof["n_allocations"]:
                print(f"  Hinweis: {run_dir.name}: {len(train_rows)} Slurm-Zuteilungen, "
                      f"{prof['n_allocations']} Trainingsstarts im Phasenprotokoll")
        else:
            print(f"  Hinweis: keine Slurm-Zeilen fuer das Training von {run_dir.name}")
        if infer_rows:
            prof["sacct_infer_seconds"] = sum(r["elapsed_seconds"] for r in infer_rows)
    return prof


def collect_examples(preds, gold_key, pred_key, is_llm, max_tokens=22, per_class=4):
    found = {k: [] for k in ("typfehler", "grenzfehler", "auslassung", "zusatz", "unbekannt")}

    def span_text(tokens, e):
        return " ".join(tokens[e[0]:e[1]])

    for rec in preds:
        tokens = rec["tokens"]
        if len(tokens) > max_tokens:
            continue
        gold = entities(rec[gold_key])
        pred = entities(rec[pred_key])
        pset = set(pred)
        gset = set(gold)
        if is_llm and rec["parse_diagnostics"].get("unknown_types", 0) and len(found["unbekannt"]) < per_class:
            found["unbekannt"].append(dict(sentence=" ".join(tokens), raw=rec.get("raw_output", "")[:300],
                                           gold=[(span_text(tokens, e), e[2]) for e in gold]))
        for ge in gold:
            if ge in pset:
                continue
            ov = [pe for pe in pred if pe[0] < ge[1] and ge[0] < pe[1]]
            if not ov and len(found["auslassung"]) < per_class:
                found["auslassung"].append(dict(sentence=" ".join(tokens), gold=(span_text(tokens, ge), ge[2])))
            elif ov:
                same_span = [pe for pe in ov if pe[0] == ge[0] and pe[1] == ge[1]]
                if same_span and len(found["typfehler"]) < per_class:
                    found["typfehler"].append(dict(sentence=" ".join(tokens), gold=(span_text(tokens, ge), ge[2]),
                                                   pred=(span_text(tokens, same_span[0]), same_span[0][2])))
                elif not same_span:
                    same_type = [pe for pe in ov if pe[2] == ge[2]]
                    if same_type and len(found["grenzfehler"]) < per_class:
                        found["grenzfehler"].append(dict(sentence=" ".join(tokens), gold=(span_text(tokens, ge), ge[2]),
                                                         pred=(span_text(tokens, same_type[0]), same_type[0][2])))
        for pe in pred:
            if pe in gset:
                continue
            if not any(pe[0] < ge[1] and ge[0] < pe[1] for ge in gold) and len(found["zusatz"]) < per_class:
                found["zusatz"].append(dict(sentence=" ".join(tokens), pred=(span_text(tokens, pe), pe[2])))
        if all(len(v) >= per_class for v in found.values()):
            break
    return found


def aggregate_group(runs):
    keys = ["precision", "recall", "f1", "macro_f1", "latency_mean", "latency_p50", "latency_p95", "vram_mb",
            "throughput", "train_seconds", "inference_seconds", "inference_seconds_latency_estimate"]
    agg = {}
    for k in keys:
        vals = [r.get(k) for r in runs]
        m, s = mean_std(vals)
        agg[k] = dict(mean=m, std=s, values=vals)
    agg["inference_latency_estimate"] = agg.pop("inference_seconds_latency_estimate")
    sources = [r.get("inference_seconds_source") for r in runs]
    agg["inference_seconds_source"] = sources[0] if len(set(sources)) == 1 else sources
    # Trainingszeitprofil (Step 1): Slurm-Summe fuer alle, Hochrechnung nur fuer wiederaufgenommene Laeufe
    profiles = [r.get("training_profile") for r in runs]
    agg["training_profile"] = profiles
    vals = [p["sacct_train_seconds"] if p else None for p in profiles]
    m, s = mean_std(vals)
    agg["training_sacct_seconds"] = dict(mean=m, std=s, values=vals)
    agg["recovery_checkpoints"] = [r.get("recovery_checkpoint") for r in runs]
    if any(p and p["resumed"] for p in profiles):
        for name, field in (("training_estimate_seconds", "estimated_total_seconds"),
                            ("training_last_segment_seconds", "logged_seconds"),
                            ("training_wallclock_span_seconds", "wallclock_span_seconds")):
            vals = [p[field] for p in profiles]
            m, s = mean_std(vals)
            agg[name] = dict(mean=m, std=s, values=vals)
        agg["training_n_allocations"] = [p["n_allocations"] for p in profiles]
    agg["per_type"] = {}
    for t in TYPES:
        agg["per_type"][t] = {}
        for k in ("precision", "recall", "f1"):
            m, s = mean_std([r["per_type"][t][k] for r in runs])
            agg["per_type"][t][k] = dict(mean=m, std=s)
        agg["per_type"][t]["support"] = runs[0]["per_type"][t]["support"]
    # Fehlerprofil: Mittel der Anteile
    n_gold = runs[0]["errors"]["n_gold"]
    agg["errors"] = {}
    for cls in ("korrekt", "typfehler", "grenzfehler", "grenz_typ", "auslassung"):
        m, s = mean_std([r["errors"]["counts"].get(cls, 0) / n_gold for r in runs])
        agg["errors"][cls] = dict(mean=m, std=s)
    m, s = mean_std([r["errors"]["additions"] for r in runs])
    agg["errors"]["additions"] = dict(mean=m, std=s)
    agg["errors"]["n_gold"] = n_gold
    conf = Counter()
    for r in runs:
        conf.update(r["errors"]["confusion"])
    agg["errors"]["confusion_top"] = [(f"{a}->{b}", c / len(runs)) for (a, b), c in conf.most_common(12)]
    bnd = Counter()
    for r in runs:
        bnd.update(r["errors"]["boundary_by_type"])
    agg["errors"]["boundary_by_type"] = {t: c / len(runs) for t, c in bnd.most_common()}
    add = Counter()
    for r in runs:
        add.update(r["errors"]["additions_by_type"])
    agg["errors"]["additions_by_type"] = {t: c / len(runs) for t, c in add.most_common()}
    if "form" in runs[0]:
        agg["form"] = {}
        for k in runs[0]["form"]:
            if k in ("status", "unknown_examples"):
                continue
            m, s = mean_std([r["form"][k] for r in runs])
            agg["form"][k] = dict(mean=m, std=s, values=[r["form"][k] for r in runs])
        agg["form"]["unknown_examples"] = [r["form"]["unknown_examples"] for r in runs]
    agg["best_epochs"] = [r.get("best_epoch") for r in runs]
    agg["val_f1"] = [r.get("val_f1") for r in runs]
    agg["epoch_results"] = [r.get("epoch_results") for r in runs]
    agg["train_lower_bound"] = any(r.get("train_lower_bound") for r in runs)
    agg["resume_checkpoints"] = [r.get("resume_checkpoint") for r in runs]
    if "truncation" in runs[0]:
        agg["truncation"] = {k: mean_std([r["truncation"][k] for r in runs]) for k in runs[0]["truncation"]}
    agg["gpus"] = sorted({r["gpu"] for r in runs if r["gpu"]})
    agg["commits"] = sorted({r["commit"] for r in runs if r["commit"]})
    agg["total_params_reported"] = [r.get("total_params") for r in runs]
    if "regen" in runs[0]:
        agg["regen"] = {}
        for k in ("n_regenerated", "n_limit_reached", "generated_tokens_mean", "latency_ms_mean_regenerated",
                  "n_failed_after", "n_failed_regenerated", "n_failed_before_regenerated",
                  "stored_prefix_match_count", "json_complete_count"):
            vals = [r["regen"][k] for r in runs]
            m, s = mean_std(vals)
            agg["regen"][k] = dict(mean=m, std=s, values=vals)
        agg["regen"]["stop_reasons"] = [r["regen"]["stop_reasons"] for r in runs]
        agg["regen"]["subset_metrics"] = {
            ev: {m: mean_std([r["regen"]["subset_metrics"][ev][m] for r in runs]) for m in ("precision", "recall", "f1")}
            for ev in ("baseline_256", "output_limit")}
    return agg


def paired_limit_stats(base_runs, runs):
    """Seedweise Paarung Basis (Limit 256) gegen Nachinferenz (Limit 1024): identische Saetze und Referenz,
    nicht regenerierte Saetze mit unveraenderter Vorhersage, gespeicherte 256er-Vorhersage der regenerierten
    Saetze gleich der Basis; sonst ABBRUCH. Rueckgabe: Kennzahlen je Seed und mean/std darueber."""
    per_seed = []
    for b, r in zip(base_runs, runs):
        if (b["sentence_signature"], b["gold_signature"]) != (r["sentence_signature"], r["gold_signature"]):
            sys.exit(f"ABBRUCH: 1024-Lauf {r['name']}: Saetze oder Referenz weichen von {b['name']} ab")
        rg = r.get("regen")
        if rg is None:
            sys.exit(f"ABBRUCH: 1024-Lauf {r['name']} ohne regenerated-Markierungen")
        regen = set(rg["indices"])
        changed = sum(1 for i, (hb, hr) in enumerate(zip(b["pred_hashes"], r["pred_hashes"]))
                      if i not in regen and hb != hr)
        if changed:
            sys.exit(f"ABBRUCH: 1024-Lauf {r['name']}: {changed} nicht regenerierte Saetze mit veraenderter Vorhersage")
        mism = sum(1 for i, h in rg["baseline_pred_hashes"].items() if h != b["pred_hashes"][i])
        if mism:
            sys.exit(f"ABBRUCH: 1024-Lauf {r['name']}: gespeicherte 256er-Vorhersage weicht in {mism} Saetzen ab")
        per_seed.append(dict(
            run=r["name"], f1_256=b["f1"], f1_1024=r["f1"], delta_f1=r["f1"] - b["f1"],
            delta_precision=r["precision"] - b["precision"], delta_recall=r["recall"] - b["recall"],
            delta_macro_f1=r["macro_f1"] - b["macro_f1"],
            omission_256=b["errors"]["counts"].get("auslassung", 0) / b["errors"]["n_gold"],
            omission_1024=r["errors"]["counts"].get("auslassung", 0) / r["errors"]["n_gold"],
            failed_256=b["form"]["failed"], failed_1024=r["form"]["failed"],
            n_regenerated=rg["n_regenerated"], n_limit_reached=rg["n_limit_reached"],
            subset_f1_256=rg["subset_metrics"]["baseline_256"]["f1"],
            subset_f1_1024=rg["subset_metrics"]["output_limit"]["f1"]))
    stats = {k: dict(zip(("mean", "std"), mean_std([p[k] for p in per_seed]))) for k in per_seed[0] if k != "run"}
    return dict(per_seed=per_seed, **stats)


# --------------------------------------------------------------------------- CrossNER
def analyse_crossner():
    agg_file = load_yaml(RES / "crossner" / "aggregate" / "summary.yaml")
    excluded = set(load_yaml(CFG / "crossner_overlap_exclusions.yaml")["excluded_sample_ids"])
    with open(RES / "crossner" / "aggregate" / "seed_metrics.csv", encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    with open(RES / "crossner" / "aggregate" / "per_entity_metrics.csv", encoding="utf-8", newline="") as handle:
        entity_rows = list(csv.DictReader(handle))
    out = dict(aggregate={}, sensitivity={}, parse={}, latency={}, gpus={}, sentence_stats={}, per_seed={},
               seed_table={}, per_type_domain={}, samples={}, vram={})
    for key, name in CROSS_AGG_NAMES.items():
        out["aggregate"][key] = {row["domain"]: row for row in agg_file["summary"] if row["model"] == name}
    for key, runs in CROSS_RUNS.items():
        is_llm = key.startswith("qwen")
        pooled_full, pooled_excl, fails, no_close, lat_mean, lat_median, n_total = [], [], [], [], [], [], []
        gpus = set()
        tokens_per_sentence, ents_per_sentence = [], []
        per_domain_seed, per_domain_seed_types, resources = {}, {}, {}
        for run in runs:
            records = []
            for dom, _ in DOMAINS:
                dom_dir = RES / "crossner" / run / dom
                recs = load_json(dom_dir / "test_predictions.json")
                records.extend(recs)
                if out["samples"].setdefault(dom, len(recs)) != len(recs):
                    sys.exit(f"ABBRUCH: CrossNER {run}/{dom}: Satzzahl weicht von anderen Laeufen ab")
                im = load_yaml(dom_dir / "inference_metrics.yaml")
                dev = (im.get("run_metadata", {}).get("cuda", {}).get("devices") or [None])[0]
                if dev:
                    gpus.add(dev)
                resources[(run, dom)] = dict(vram_peak_mb=im.get("vram_peak_mb"), latency_ms_mean=im.get("latency_ms_mean"),
                                             latency_ms_p50=im.get("latency_ms_p50"), latency_ms_p95=im.get("latency_ms_p95"))
                ov, pt = score(recs, "gold_bio", "pred_bio", COARSE)
                per_domain_seed[(run, dom)] = ov
                per_domain_seed_types[(run, dom)] = pt
            ov_full, _ = score(records, "gold_bio", "pred_bio", COARSE)
            ov_excl, _ = score([r for r in records if r["sample_id"] not in excluded], "gold_bio", "pred_bio", COARSE)
            pooled_full.append(ov_full)
            pooled_excl.append(ov_excl)
            n_total.append(len(records))
            lats = [r["latency_ms"] for r in records]
            lat_mean.append(statistics.fmean(lats))
            lat_median.append(statistics.median(lats))
            if is_llm:
                fails.append(sum(1 for r in records if r["parse_status"] == "failed"))
                no_close.append(sum(1 for r in records if r["parse_status"] == "failed"
                                    and not r.get("raw_output", "").rstrip().endswith("]")))
            if not tokens_per_sentence:
                tokens_per_sentence = [len(r["tokens"]) for r in records]
                ents_per_sentence = [len(entities(r["gold_bio"])) for r in records]
        agg_pooled = out["aggregate"][key]["pooled_micro"]["f1_mean"]
        own_pooled = statistics.fmean(p[2] for p in pooled_full)
        if abs(agg_pooled - own_pooled) > 1e-6:
            sys.exit(f"ABBRUCH: CrossNER gepoolt {key}: eigen {own_pooled:.6f}, Aggregat {agg_pooled:.6f}")
        out["sensitivity"][key] = dict(
            full=dict(zip(("precision", "recall", "f1"), [mean_std([p[i] for p in pooled_full]) for i in range(3)])),
            excluded=dict(zip(("precision", "recall", "f1"), [mean_std([p[i] for p in pooled_excl]) for i in range(3)])),
            n_full=n_total[0], n_excluded=n_total[0] - len(excluded))
        out["latency"][key] = dict(mean=mean_std(lat_mean), median=mean_std(lat_median))
        out["gpus"][key] = sorted(gpus)
        if is_llm:
            out["parse"][key] = dict(failed=mean_std(fails), failed_no_close=mean_std(no_close), n=n_total[0],
                                     rate=mean_std([f / n for f, n in zip(fails, n_total)]))
        out["sentence_stats"][key] = dict(tokens_mean=statistics.fmean(tokens_per_sentence),
                                          entities_mean=statistics.fmean(ents_per_sentence),
                                          tokens_max=max(tokens_per_sentence))
        out["per_seed"][key] = {f"{run}/{dom}": ov for (run, dom), ov in per_domain_seed.items()}
        vram_vals = [v["vram_peak_mb"] for v in resources.values() if v["vram_peak_mb"] is not None]
        out["vram"][key] = dict(max_mb=max(vram_vals) if vram_vals else None,
                                mean_mb=statistics.fmean(vram_vals) if vram_vals else None,
                                per_run_domain={f"{run}/{dom}": v for (run, dom), v in resources.items()})
        # Seed-Einzelwerte und Typwerte aus den Aggregat-CSVs, jede Zeile gegen die eigene strikte Rechnung geprueft
        name = CROSS_AGG_NAMES[key]
        seed_to_run = dict(zip(SEEDS, runs))
        table = {}
        for row in seed_rows:
            if row["model"] != name:
                continue
            seed, dom = int(row["seed"]), row["domain"]
            vals = {k: float(row[k]) for k in ("precision", "recall", "f1")}
            own = (per_domain_seed.get((seed_to_run[seed], dom)) if dom in out["samples"]
                   else pooled_full[SEEDS.index(seed)] if dom == "pooled_micro" else None)
            if own is not None:
                if any(abs(vals[k] - own[i]) > 1e-9 for i, k in enumerate(("precision", "recall", "f1"))):
                    sys.exit(f"ABBRUCH: seed_metrics.csv {name}/{seed}/{dom}: F1 {vals['f1']:.8f} != eigen {own[2]:.8f}")
                if dom in out["samples"] and int(row["samples"]) != out["samples"][dom]:
                    sys.exit(f"ABBRUCH: seed_metrics.csv {name}/{seed}/{dom}: Satzzahl {row['samples']}")
            table.setdefault(dom, {})[seed] = vals
        if any(len(table.get(dom, {})) != len(SEEDS) for dom in list(out["samples"]) + ["pooled_micro", "domain_mean"]):
            sys.exit(f"ABBRUCH: seed_metrics.csv {name}: Zeilen fehlen")
        out["seed_table"][key] = table
        types = {}
        for row in entity_rows:
            if row["model"] != name:
                continue
            seed, dom, t = int(row["seed"]), row["domain"], row["entity_type"]
            pt = per_domain_seed_types[(seed_to_run[seed], dom)][t]
            if abs(float(row["f1"]) - pt["f1"]) > 1e-9 or int(float(row["support"])) != pt["support"]:
                sys.exit(f"ABBRUCH: per_entity_metrics.csv {name}/{seed}/{dom}/{t} weicht ab")
            types.setdefault(dom, {}).setdefault(t, []).append(
                dict(seed=seed, precision=float(row["precision"]), recall=float(row["recall"]), f1=float(row["f1"]),
                     support=int(float(row["support"]))))
        out["per_type_domain"][key] = {
            dom: {t: dict(f1=mean_std([v["f1"] for v in vals]), precision=mean_std([v["precision"] for v in vals]),
                          recall=mean_std([v["recall"] for v in vals]), support=vals[0]["support"], n_seeds=len(vals))
                  for t, vals in per_dom.items()}
            for dom, per_dom in types.items()}
    # Sensitivitaet des Ausgabelimits: 1024-Rauchtest gegen Hauptlauf auf denselben Saetzen
    limit = {}
    for dom in ("music", "politics"):
        smoke = load_json(RES / "crossner_output_limit_1024" / "smoke" / "qwen35-27b-qlora-3ep" / dom
                          / "test_predictions.json")
        ids = {r["sample_id"] for r in smoke}
        main = [r for r in load_json(RES / "crossner" / "qwen35-27b-qlora-3ep" / dom / "test_predictions.json")
                if r["sample_id"] in ids]
        if len(main) != len(smoke):
            sys.exit("ABBRUCH: 1024-Rauchtest: Satz-IDs nicht im Hauptlauf gefunden")
        limit[dom] = dict(
            n=len(smoke),
            limit256=dict(zip(("precision", "recall", "f1"), score(main, "gold_bio", "pred_bio", COARSE)[0]),
                          failed=sum(1 for r in main if r["parse_status"] == "failed"),
                          latency_mean=statistics.fmean(r["latency_ms"] for r in main)),
            limit1024=dict(zip(("precision", "recall", "f1"), score(smoke, "gold_bio", "pred_bio", COARSE)[0]),
                           failed=sum(1 for r in smoke if r["parse_status"] == "failed"),
                           latency_mean=statistics.fmean(r["latency_ms"] for r in smoke)),
            tokens_mean=statistics.fmean(len(r["tokens"]) for r in smoke))
    out["output_limit"] = limit
    out["output_limit_full"] = analyse_crossner_limit(out)
    return out


def analyse_crossner_limit(C):
    """Vollstaendige CrossNER-Auswertung mit Ausgabelimit 1024 (aggregate/comparison.yaml des Moduls
    src/evaluate/crossner_output_limit.py). Fehlt die Datei oder ist sie unvollstaendig, nur Hinweis."""
    path = RES / "crossner_output_limit_1024" / "aggregate" / "comparison.yaml"
    if not path.exists():
        print(f"  Hinweis: {path.relative_to(RES)} fehlt; tab-limit.tex wird nicht erzeugt")
        return dict(available=False, complete=False)
    rep = load_yaml(path)
    status = dict(available=True, complete=bool(rep.get("complete")), max_new_tokens=rep.get("max_new_tokens"),
                  successful_evaluations=rep.get("successful_evaluations"),
                  expected_evaluations=rep.get("expected_evaluations"))
    if not status["complete"]:
        print(f"  Hinweis: 1024-Auswertung unvollstaendig ({status['successful_evaluations']}/"
              f"{status['expected_evaluations']}); tab-limit.tex wird nicht erzeugt")
        return status
    limit_eval = rep.get("evaluation") or f"output_limit_{rep.get('max_new_tokens', 1024)}"
    rows = {(r["evaluation"], r["population"], r["domain"]): r for r in rep["summary"]}
    doms = [d for d, _ in DOMAINS] + ["pooled_micro", "domain_mean"]
    # Selbstpruefung: Referenzzeilen (256 bzw. Encoder) muessen den Hauptlauf reproduzieren (gleiche Saetze, gleicher Scorer)
    for dom in doms:
        for ev, key in (("baseline_256", "qwen27-qlora"), ("encoder_baseline", "deberta-large")):
            r = rows.get((ev, "full", dom))
            if r is None or rows.get((limit_eval, "full", dom)) is None:
                sys.exit(f"ABBRUCH: comparison.yaml ohne Zeile {ev}/full/{dom}")
            if abs(r["f1_mean"] - C["aggregate"][key][dom]["f1_mean"]) > 1e-6:
                sys.exit(f"ABBRUCH: comparison.yaml {ev}/{dom}: F1 {r['f1_mean']:.6f} != Hauptlauf "
                         f"{C['aggregate'][key][dom]['f1_mean']:.6f}")
    for ev, key in (("baseline_256", "qwen27-qlora"), ("encoder_baseline", "deberta-large")):
        r = rows.get((ev, "deduplicated", "pooled_micro"))
        if r is not None and abs(r["f1_mean"] - C["sensitivity"][key]["excluded"]["f1"][0]) > 1e-6:
            sys.exit(f"ABBRUCH: comparison.yaml {ev}/deduplicated/pooled_micro weicht von der Sensitivitaet ab")

    def pick(r):
        return {k: v for k, v in r.items() if k.endswith("_mean") or k.endswith("_std")
                or k in ("sample_count", "successful_seeds", "complete")}

    sizes = {"256": "baseline_256", str(status["max_new_tokens"]): limit_eval, "encoder": "encoder_baseline"}
    for pop in ("full", "deduplicated"):
        status[pop] = {dom: {size: pick(rows[(ev, pop, dom)]) for size, ev in sizes.items() if (ev, pop, dom) in rows}
                       for dom in doms}
    status["evaluation"] = limit_eval
    return status


# --------------------------------------------------------------------------- Satz-Bootstrap
def f1_of(S):
    """F1 aus (tp, fp, fn) in der letzten Achse; Nullfaelle wie prf()."""
    tp, fp, fn = S[..., 0], S[..., 1], S[..., 2]
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)


def bootstrap_intervals(G, pair_keys, B=BOOT_B, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """Gepaartes Satz-Bootstrap: dieselben B Stichproben (Ziehen mit Zuruecklegen ueber die n Testsaetze)
    fuer alle Laeufe und Konfigurationen. Je Replikat werden die Satzzaehler gewichtet aufsummiert
    (Ganzzahlen in float64, daher exakt und plattformunabhaengig reproduzierbar) und daraus Mikro-F1
    und Makro-F1 berechnet. Intervalle: Perzentile 2,5/97,5 des Seed-Mittels je Replikat; Differenzen
    seedgleich gepaart (42/123/456), bei einem Einzellauf (Zero-Shot) dieser gegen jeden Seed."""
    rng = np.random.default_rng(seed)
    n = next(iter(G.values()))["runs"][0]["n_sentences"]
    nt = len(TYPES)
    width = 3 + 3 * nt
    M, R = {}, {}
    for key, g in G.items():
        M[key] = np.concatenate([np.concatenate([r["sent_micro"], r["sent_types"].reshape(n, 3 * nt)], axis=1)
                                 for r in g["runs"]], axis=1).astype(np.float64)
        R[key] = len(g["runs"])
        point = f1_of(M[key].sum(0).reshape(R[key], width)[:, :3])     # ungewichtet = Laufwerte
        if any(abs(point[j] - r["f1"]) > 1e-12 for j, r in enumerate(g["runs"])):
            sys.exit(f"ABBRUCH: Bootstrap-Punktschaetzung {key} weicht von den Laufwerten ab")
    F1 = {k: np.empty((B, R[k])) for k in G}
    MF1 = {k: np.empty((B, R[k])) for k in G}
    for lo in range(0, B, chunk):
        b = min(chunk, B - lo)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float64)
        for key in G:
            S = (W @ M[key]).reshape(b, R[key], width)
            F1[key][lo:lo + b] = f1_of(S[:, :, :3])
            MF1[key][lo:lo + b] = f1_of(S[:, :, 3:].reshape(b, R[key], nt, 3)).mean(axis=2)

    def ci(x):
        lo_, hi_ = np.percentile(x, [2.5, 97.5])
        return [float(lo_) * 100, float(hi_) * 100]

    configs = {}
    for key, g in G.items():
        a = g["agg"]
        configs[key] = dict(
            micro=dict(point=a["f1"]["mean"] * 100, ci_seed_mean=ci(F1[key].mean(1)),
                       ci_per_seed=[ci(F1[key][:, j]) for j in range(R[key])]),
            macro=dict(point=a["macro_f1"]["mean"] * 100, ci_seed_mean=ci(MF1[key].mean(1)),
                       ci_per_seed=[ci(MF1[key][:, j]) for j in range(R[key])]))
    pairs = {}
    for pk in pair_keys:
        k1, k2 = pk.split("->")
        single = R[k1] == 1 or R[k2] == 1
        lo_, hi_ = ci((F1[k2] - F1[k1]).mean(1))        # Broadcasting, wenn eine Seite ein Einzellauf ist
        mlo, mhi = ci((MF1[k2] - MF1[k1]).mean(1))
        pairs[pk] = dict(diff=(G[k2]["agg"]["f1"]["mean"] - G[k1]["agg"]["f1"]["mean"]) * 100, ci=[lo_, hi_],
                         excludes_zero=bool(lo_ > 0 or hi_ < 0),
                         macro_diff=(G[k2]["agg"]["macro_f1"]["mean"] - G[k1]["agg"]["macro_f1"]["mean"]) * 100,
                         macro_ci=[mlo, mhi], macro_excludes_zero=bool(mlo > 0 or mhi < 0),
                         pairing="single-run vs each seed" if single else "seed-matched")
    settings = dict(B=B, seed=seed, chunk=chunk, n_sentences=n, interval="Perzentil 2,5/97,5",
                    note="Satz-Bootstrap mit identischer Stichprobe fuer alle Laeufe; erfasst Testsplit-Unsicherheit, "
                         "nicht Seed-Streuung")
    return dict(settings=settings, configs=configs, pairs=pairs)


def crossner_bootstrap(C, B=BOOT_B, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """Gepaartes, nach Domaene geschichtetes Satz-Bootstrap auf PER/ORG/LOC.

    Pro Domaene bleiben Satzzahl und Paarung aller Modelle/Seeds erhalten.
    Gepoolter Mikro-F1 wird aus summierten Entitaetszaehlern berechnet,
    anschliessend ueber die drei festen Seeds gemittelt. Keine erneute
    Seed-Ziehung, Dokumentcluster oder Korrektur fuer Mehrfachvergleiche.
    """
    configs = [("deberta-large", RES / "crossner", CROSS_RUNS["deberta-large"]),
               ("qwen27-qlora", RES / "crossner", CROSS_RUNS["qwen27-qlora"])]
    if C["output_limit_full"].get("complete"):
        configs.append(("qwen27-qlora-1024", RES / "crossner_output_limit_1024",
                        CROSS_RUNS["qwen27-qlora"]))
    rng = np.random.default_rng(seed)
    pooled = np.zeros((B, len(configs), len(SEEDS), 3), dtype=np.float64)
    pooled_point = np.zeros(pooled.shape[1:], dtype=np.float64)
    domains, sizes = {}, {}

    def summarise(counts, point_counts):
        values = f1_of(counts).mean(axis=2) * 100
        point = f1_of(point_counts).mean(axis=1) * 100
        pairs = {}
        for j in range(1, len(configs)):
            lo, hi = np.percentile(values[:, j] - values[:, 0], [2.5, 97.5])
            pairs[configs[j][0]] = dict(diff=float(point[j] - point[0]),
                                       ci=[float(lo), float(hi)],
                                       excludes_zero=bool(lo > 0 or hi < 0))
        return dict(configs={cfg[0]: float(point[j]) for j, cfg in enumerate(configs)},
                    pairs=pairs)

    for dom, _ in DOMAINS:
        reference = None
        matrices = []
        for key, root, runs in configs:
            per_seed = []
            for run in runs:
                records = load_json(root / run / dom / "test_predictions.json")
                signature = [(r["sample_id"], r["tokens"], r["gold_bio"]) for r in records]
                if not records or len({r["sample_id"] for r in records}) != len(records):
                    raise ValueError(f"CrossNER {key}/{run}/{dom}: leere oder doppelte Satz-IDs")
                if any(len(r["tokens"]) != len(r[k]) for r in records for k in ("gold_bio", "pred_bio")):
                    raise ValueError(f"CrossNER {key}/{run}/{dom}: ungleiche Token-/Tag-Laengen")
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise ValueError(f"CrossNER {key}/{run}/{dom}: Satzfolge oder Referenz weicht ab")
                counts, _ = sentence_counts(records, "gold_bio", "pred_bio", COARSE)
                per_seed.append(counts)
            matrices.append(np.stack(per_seed, axis=1))
        matrix = np.stack(matrices, axis=1).astype(np.float64)
        n = len(matrix)
        sizes[dom] = n
        point = matrix.sum(axis=0)
        samples = np.empty_like(pooled)
        for start in range(0, B, chunk):
            b = min(chunk, B - start)
            weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                                for _ in range(b)]).astype(np.float64)
            samples[start:start + b] = (weights @ matrix.reshape(n, -1)).reshape(b, *point.shape)
        domains[dom] = summarise(samples, point)
        pooled += samples
        pooled_point += point
    domains["pooled_micro"] = summarise(pooled, pooled_point)
    # Unabhaengiger Abgleich der Punktschaetzungen gegen die gespeicherten Aggregate.
    for dom, result in domains.items():
        for key, actual in result["configs"].items():
            expected = (C["output_limit_full"]["full"][dom]["1024"]["f1_mean"]
                        if key.endswith("1024") else C["aggregate"][key][dom]["f1_mean"])
            if abs(actual - expected * 100) > 1e-4:
                raise ValueError(f"CrossNER Bootstrap {key}/{dom}: Punktschaetzung weicht vom Aggregat ab")
    return dict(settings=dict(B=B, seed=seed, chunk=chunk, n_by_domain=sizes,
                              reference="deberta-large", interval="Perzentil 2,5/97,5",
                              note="Nach Domaene geschichtetes, gepaartes Satz-Bootstrap; feste Seeds; "
                                   "keine Dokumentcluster oder Korrektur fuer Mehrfachvergleiche"),
                domains=domains)


def tab_crossner_ci(C):
    keys = ["qwen27-qlora"]
    if C["output_limit_full"].get("complete"):
        keys.append("qwen27-qlora-1024")
    headers = ["\\textbf{Domäne}"]
    for key in keys:
        limit = "1\\,024" if key.endswith("1024") else "256"
        headers.extend([f"$\\Delta$F$_1$ ({limit})", "95-\\%-Intervall"])
    lines = ["\\begin{tabular}{l" + "rr" * len(keys) + "}", "    \\toprule",
             "    " + " & ".join(headers) + r" \\", "    \\midrule"]
    for dom, label in DOMAINS + [("pooled_micro", "Gepoolt")]:
        if dom == "pooled_micro":
            lines.append("    \\midrule")
        cells = [label]
        for key in keys:
            p = C["bootstrap"]["domains"][dom]["pairs"][key]
            cells.extend([signed(p["diff"]),
                          f"[{signed(p['ci'][0])}; {signed(p['ci'][1])}]"])
        lines.append("    " + " & ".join(cells) + r" \\")
    return "\n".join(lines + ["    \\bottomrule", "\\end{tabular}", ""])


# --------------------------------------------------------------------------- Tabellen
def tab_haupt(G):
    rows = []
    for key in TABLE_ORDER:
        g = G[key]
        a = g["agg"]
        cells = [pm(a[k]["mean"] * 100, None if a[k]["std"] is None else a[k]["std"] * 100)
                 for k in ("precision", "recall", "f1")]
        cells.append(pm(a["macro_f1"]["mean"] * 100, None if a["macro_f1"]["std"] is None else a["macro_f1"]["std"] * 100))
        rows.append(f"    {g['label']} & {g['regime']} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows[:2]) + "\n    \\midrule\n" + "\n".join(rows[2:5]) + "\n    \\midrule\n" + "\n".join(rows[5:])
    return ("\\begin{tabular}{llcccc}\n    \\toprule\n    \\textbf{Modell} & \\textbf{Regime} & \\textbf{Precision} &\n"
            "    \\textbf{Recall} & \\textbf{Mikro-F$_1$} & \\textbf{Makro-F$_1$} \\\\\n    \\midrule\n"
            + body + "\n    \\bottomrule\n\\end{tabular}\n")


def tab_typen(G):
    cols = ["deberta-base", "deberta-large", "qwen08-qlora", "qwen4-qlora", "qwen27-qlora", "qwen27-zs"]
    lines = ["\\begin{tabular}{lrcccccc}", "    \\toprule",
             "    & & \\multicolumn{2}{c}{\\textbf{DeBERTa-v3}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5 QLoRA}} &",
             "      \\textbf{Qwen3.5} \\\\",
             "    \\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}",
             "    \\textbf{Typ} & \\textbf{Support} & \\textbf{base} & \\textbf{large} &",
             "      \\textbf{0.8B} & \\textbf{4B} & \\textbf{27B} & \\textbf{27B Zero-Shot} \\\\",
             "    \\midrule"]
    for gi, (title, types) in enumerate(TYPE_GROUPS):
        if gi:
            lines.append("    \\midrule")
        lines.append(f"    \\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for t in types:
            sup = G["deberta-base"]["agg"]["per_type"][t]["support"]
            vals = [de(G[c]["agg"]["per_type"][t]["f1"]["mean"] * 100, 1) for c in cols]
            lines.append(f"    {t:5s} & {int(sup):,}".replace(",", "\\,") + " & " + " & ".join(vals) + " \\\\")
    lines.append("    \\midrule")
    n_gold = G["deberta-base"]["agg"]["errors"]["n_gold"]
    lines.append(f"    Mikro-F$_1$ (alle) & {n_gold:,}".replace(",", "\\,") + " & "
                 + " & ".join(de(G[c]["agg"]["f1"]["mean"] * 100, 1) for c in cols) + " \\\\")
    lines.append("    Makro-F$_1$ & -- & " + " & ".join(de(G[c]["agg"]["macro_f1"]["mean"] * 100, 1) for c in cols) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_groesse(G):
    def cell(key):
        a = G[key]["agg"]["f1"]
        return pm(a["mean"] * 100, None if a["std"] is None else a["std"] * 100)

    def delta(k2, k1):
        return signed((G[k2]["agg"]["f1"]["mean"] - G[k1]["agg"]["f1"]["mean"]) * 100)

    lines = ["\\begin{tabular}{llccccc}", "    \\toprule",
             "    \\textbf{Familie} & \\textbf{Regime} & \\textbf{S} & \\textbf{M} &",
             "    \\textbf{L} & $\\Delta$(M$-$S) & $\\Delta$(L$-$M) \\\\", "    \\midrule",
             f"    DeBERTa-v3 & Fine-Tuning & {cell('deberta-base')} & {cell('deberta-large')} & -- & "
             f"{delta('deberta-large', 'deberta-base')} & -- \\\\",
             f"    Qwen3.5 & Zero-Shot & {cell('qwen08-zs')} & {cell('qwen4-zs')} & {cell('qwen27-zs')} & "
             f"{delta('qwen4-zs', 'qwen08-zs')} & {delta('qwen27-zs', 'qwen4-zs')} \\\\",
             f"    Qwen3.5 & QLoRA & {cell('qwen08-qlora')} & {cell('qwen4-qlora')} & {cell('qwen27-qlora')} & "
             f"{delta('qwen4-qlora', 'qwen08-qlora')} & {delta('qwen27-qlora', 'qwen4-qlora')} \\\\",
             "    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def note_rows(text, ncols, width=100):
    """Tabellenfussnote als eine oder mehrere \\multicolumn-Zeilen (Umbruch an Wortgrenzen, damit
    keine Zeile breiter als die Tabelle wird; die Zeichenzahl zaehlt TeX-Makros mit, ist also konservativ)."""
    rows, cur = [], ""
    for word in text.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            rows.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    rows.append(cur)
    return [f"    \\multicolumn{{{ncols}}}{{@{{}}l}}{{\\scriptsize {r}}} \\\\" for r in rows]


def day_list(dates, with_year=True):
    """'7., 12., 15. und 18.~August 2026' (ein Monat) bzw. '31.~Juli, 2.~August 2026' (Monatswechsel)."""
    days = sorted({(d.year, d.month, d.day) for d in dates})
    if len({(y, m) for y, m, _ in days}) == 1:
        y, m, _ = days[0]
        items = [f"{d}." for _, _, d in days]
        joined = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " und " + items[-1]
        return f"{joined}~{MONTHS[m - 1]}" + (f" {y}" if with_year else "")
    items = [f"{d}.~{MONTHS[m - 1]}" + (f" {y}" if with_year else "") for y, m, d in days]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " und " + items[-1]


def training_footnote(a, sacct_ok):
    """Fussnote a der Effizienztabelle, aus den Trainingsprofilen der wiederaufgenommenen Gruppe erzeugt."""
    profs = a["training_profile"]
    starts = [iso(s) for p in profs for s in p["allocation_starts"]]
    done = [iso(p["completed_at"]) for p in profs if p["completed_at"]]
    n_alloc = sorted({p["n_allocations"] for p in profs})
    n_word = NUMBER_WORDS.get(n_alloc[0], str(n_alloc[0])) if len(n_alloc) == 1 else f"{n_alloc[0]}--{n_alloc[-1]}"
    resume = " bzw. ".join(fmt_int(s) for s in dict.fromkeys(p["resume_step"] for p in profs))
    last_h = de(a["training_last_segment_seconds"]["mean"] / 3600, 1)
    rate = de(statistics.fmean(p["seconds_per_step"] for p in profs), 1)
    total = fmt_int(profs[0]["total_steps"])
    est_h = de(a["training_estimate_seconds"]["mean"] / 3600, 0)
    dates = f"(Start {day_list(starts)}, Abschluss {day_list(done, with_year=False)})"
    nodes = {r["node"] for p in profs for r in (p["sacct_train_detail"] or [])}
    seeds_word = NUMBER_WORDS.get(len(profs), str(len(profs))).lower()
    same_node = f" Die {seeds_word} Seeds liefen gleichzeitig auf einem Knoten." if len(nodes) == 1 and len(profs) > 1 else ""
    if sacct_ok:
        return (f"{n_word} Slurm-Zuteilungen {dates} wegen des 3-Tage-Limits, Summe laut Slurm-Abrechnung; "
                f"Hochrechnung \\\"uber die Schrittrate des letzten Abschnitts: {est_h}~h "
                f"(Abschnitt~\\ref{{sec:effizienzmetriken}}).{same_node}")
    return (f"Hochrechnung: {n_word} Slurm-Zuteilungen {dates} wegen des 3-Tage-Limits; der Trainer "
            f"protokollierte nur den letzten Abschnitt ab Schritt {resume} (Mittel {last_h}~h). Dessen Schrittrate "
            f"({rate}~s je Schritt) auf {total} Schritte hochgerechnet (Abschnitt~\\ref{{sec:effizienzmetriken}}).")


def tab_effizienz(G):
    sacct_ok = all(G[k]["agg"]["training_sacct_seconds"]["mean"] is not None for k in TRAINED)
    lines = ["\\begin{tabular}{llrrrrrrrr}", "    \\toprule",
             "    \\textbf{Modell} & \\textbf{Regime} & \\textbf{GPU} &",
             "    \\textbf{Training} & \\textbf{Testinferenz} & \\multicolumn{3}{c}{\\textbf{Latenz [ms]}} &",
             "    \\textbf{Durchsatz} & \\textbf{Speicher} \\\\", "    \\cmidrule(lr){6-8}",
             "    & & & \\textbf{[h]} & \\textbf{gesamt [h]} & \\textbf{Mittel} & \\textbf{Median} & \\textbf{P95} &",
             "    \\textbf{[S\\\"atze/s]} & \\textbf{[GiB]} \\\\", "    \\midrule"]
    for i, key in enumerate(TABLE_ORDER):
        g = G[key]
        a = g["agg"]
        if i in (2, 5):
            lines.append("    \\midrule")
        enc = key.startswith("deberta")
        if a["train_seconds"]["mean"] is None:
            train = "--"
        elif sacct_ok:                       # primaere Quelle: Summe der Slurm-Zuteilungen
            train = de(a["training_sacct_seconds"]["mean"] / 3600, 1)
            if a.get("train_lower_bound"):
                train += "$^{a}$"
        elif a.get("train_lower_bound"):     # Rueckfall ohne Abrechnung: Hochrechnung ueber die Schrittrate
            train = de(a["training_estimate_seconds"]["mean"] / 3600, 1) + "$^{a}$"
        else:
            train = de(a["train_seconds"]["mean"] / 3600, 1)
        inf = de(a["inference_seconds"]["mean"] / 3600, 2 if enc else 1)
        if a["inference_seconds_source"] != "phase_history":
            inf += "$^{c}$"
        nd = 1 if enc else 0
        lat = de_sep(a["latency_mean"]["mean"], nd)
        med = de_sep(a["latency_p50"]["mean"], nd) if a["latency_p50"]["mean"] is not None else "--$^{b}$"
        p95 = de_sep(a["latency_p95"]["mean"], nd)
        thr = "--$^{d}$" if enc else de(a["throughput"]["mean"], 2)
        mem = de(a["vram_mb"]["mean"] / 1024, 1)
        lines.append(f"    {g['label']} & {g['regime']} & {g['gpu']} & {train} & {inf} & {lat} & {med} & {p95} & "
                     f"{thr} & {mem} \\\\")
    n_sent = fmt_int(G["qwen27-zs"]["runs"][0]["n_sentences"])
    lines.append("    \\bottomrule")
    lines += note_rows("$^{a}$ " + training_footnote(G["qwen27-qlora"]["agg"], sacct_ok), 10)
    lines += note_rows("$^{b}$ F\\\"ur die Zero-Shot-L\\\"aufe wurde kein Median protokolliert.", 10)
    lines += note_rows("$^{c}$ Zero-Shot-L\\\"aufe ohne Phasenprotokoll: mittlere Latenz $\\times$ "
                       f"{n_sent} S\\\"atze (Sch\\\"atzung).", 10)
    lines += note_rows("$^{d}$ Durchsatz nur f\\\"ur die generative Inferenz protokolliert.", 10)
    lines += ["\\end{tabular}", ""]
    return "\n".join(lines)


def tab_formtreue(G):
    cols = ["qwen08-zs", "qwen4-zs", "qwen27-zs", "qwen08-qlora", "qwen4-qlora", "qwen27-qlora"]

    def val(key, field, pct=False):
        v = G[key]["agg"]["form"][field]["mean"]
        if pct:
            return de(v * 100, 2)
        return f"{round(v):,}".replace(",", "\\,")

    lines = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{5.8cm}rrrrrr}", "    \\toprule",
             "    & \\multicolumn{3}{c}{\\textbf{Zero-Shot}} &", "      \\multicolumn{3}{c}{\\textbf{QLoRA}} \\\\",
             "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "    \\textbf{Kennzahl} & \\textbf{0.8B} & \\textbf{4B} & \\textbf{27B} &",
             "      \\textbf{0.8B} & \\textbf{4B} & \\textbf{27B} \\\\", "    \\midrule"]
    rows = [("Nicht parsebare Antworten", lambda k: val(k, "failed")),
            ("\\quad Anteil an allen Antworten [\\%]", lambda k: val(k, "failure_rate", pct=True)),
            ("\\quad davon ohne schlie\\ss{}ende Klammer", lambda k: val(k, "failed_no_close")),
            ("Referenzentit\\\"aten in nicht parsebaren Antworten [\\%]",
             lambda k: de(G[k]["agg"]["truncation"]["lost_share"][0] * 100, 2)),
            ("Parse direkt", lambda k: val(k, "parse_ok")),
            ("\\quad nach Entfernen eines Codeblocks", lambda k: val(k, "markdown")),
            ("\\quad durch Extraktion aus Text", lambda k: val(k, "regex")),
            ("Falsche Struktur (kein Array)", lambda k: val(k, "wrong_schema")),
            ("Nicht-Objekte", lambda k: val(k, "invalid_items")),
            ("Fehlende Felder", lambda k: val(k, "missing_fields")),
            ("Unzul\\\"assige Indizes", lambda k: val(k, "invalid_offsets")),
            ("Unbekannte Typen", lambda k: val(k, "unknown_types")),
            ("Text-Abweichungen", lambda k: val(k, "text_mismatches")),
            ("\\\"Uberlappungen", lambda k: val(k, "overlaps")),
            ("Begonnene Elemente (Rohausgabe)", lambda k: val(k, "generated_items")),
            ("\\\"Ubernommene Spans", lambda k: val(k, "accepted_spans"))]
    for label, fn in rows:
        lines.append(f"    {label} & " + " & ".join(fn(k) for k in cols) + " \\\\")
    lines.append("    \\midrule")
    lines.append("    Leere Vorhersagen & " + " & ".join(val(k, "empty") for k in cols) + " \\\\")
    lines.append("    \\quad davon nicht parsebar & " + " & ".join(val(k, "empty_failed") for k in cols) + " \\\\")
    lines.append("    \\quad davon ohne Referenzentit\\\"at & " + " & ".join(val(k, "empty_correct") for k in cols) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_fehlertypen(G):
    keys = ["deberta-base", "deberta-large", "qwen08-qlora", "qwen4-qlora", "qwen27-qlora", "qwen27-zs"]
    lines = ["\\begin{tabular}{llrrrrrr}", "    \\toprule",
             "    \\textbf{Modell} & \\textbf{Regime} & \\textbf{Korrekt} &",
             "    \\textbf{\\shortstack[c]{Typ-\\\\fehler}} & \\textbf{\\shortstack[c]{Grenz-\\\\fehler}} &",
             "    \\textbf{\\shortstack[c]{Grenz- und\\\\Typfehler}} &",
             "    \\textbf{\\shortstack[c]{Aus-\\\\lassung}} & \\textbf{Zus\\\"atze} \\\\", "    \\midrule"]
    for key in keys:
        g = G[key]
        e = g["agg"]["errors"]
        cells = [de(e[c]["mean"] * 100, 1) for c in ("korrekt", "typfehler", "grenzfehler", "grenz_typ", "auslassung")]
        cells.append(f"{round(e['additions']['mean']):,}".replace(",", "\\,"))
        lines.append(f"    {g['label']} & {g['regime']} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_crossner(C):
    agg = C["aggregate"]
    sens = C["sensitivity"]
    lines = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.6cm}rcccccc}", "    \\toprule",
             "    & & \\multicolumn{3}{c}{\\textbf{DeBERTa-v3-large}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B (QLoRA)}} \\\\",
             "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "    \\textbf{Dom\\\"ane} & \\textbf{S\\\"atze} & \\textbf{P} & \\textbf{R} & \\textbf{F$_1$} &",
             "      \\textbf{P} & \\textbf{R} & \\textbf{F$_1$} \\\\", "    \\midrule"]
    counts = C["samples"]

    def trio(key, dom):
        r = agg[key][dom]
        return (f"{de(r['precision_mean'] * 100, 1)} & {de(r['recall_mean'] * 100, 1)} & "
                f"{pm(r['f1_mean'] * 100, r['f1_std'] * 100, 1)}")

    for dom, label in DOMAINS:
        lines.append(f"    {label} & {counts[dom]} & {trio('deberta-large', dom)} & {trio('qwen27-qlora', dom)} \\\\")
    lines.append("    \\midrule")
    lines.append(f"    Gepoolt & {fmt_int(sum(counts.values()))} & {trio('deberta-large', 'pooled_micro')} & "
                 f"{trio('qwen27-qlora', 'pooled_micro')} \\\\")
    dm_d = agg["deberta-large"]["domain_mean"]
    dm_q = agg["qwen27-qlora"]["domain_mean"]
    lines.append(f"    Dom\\\"anenmittel & -- & -- & -- & {pm(dm_d['f1_mean'] * 100, dm_d['f1_std'] * 100, 1)} & -- & -- & "
                 f"{pm(dm_q['f1_mean'] * 100, dm_q['f1_std'] * 100, 1)} \\\\")
    sd = sens["deberta-large"]["excluded"]
    sq = sens["qwen27-qlora"]["excluded"]
    lines.append(f"    Sensitivit\\\"at (ohne 21 S\\\"atze) & {sens['deberta-large']['n_excluded']:,}".replace(",", "\\,")
                 + f" & {de(sd['precision'][0] * 100, 1)} & {de(sd['recall'][0] * 100, 1)} & {pm(sd['f1'][0] * 100, sd['f1'][1] * 100, 1)}"
                 + f" & {de(sq['precision'][0] * 100, 1)} & {de(sq['recall'][0] * 100, 1)} & {pm(sq['f1'][0] * 100, sq['f1'][1] * 100, 1)} \\\\")
    lines.append("    \\midrule")
    bd = agg["deberta-large"]["pooled_micro"]
    bq = agg["qwen27-qlora"]["pooled_micro"]
    lines.append(f"    MultiNERD-Referenz (PER/ORG/LOC) & 16\\,454 & -- & -- & {pm(bd['baseline_f1_mean'] * 100, bd['baseline_f1_std'] * 100, 1)}"
                 f" & -- & -- & {pm(bq['baseline_f1_mean'] * 100, bq['baseline_f1_std'] * 100, 1)} \\\\")
    lines.append(f"    $\\Delta$F$_1$ (gepoolt $-$ Referenz) & & & & {signed(bd['delta_f1_mean'] * 100, 1)} & & & "
                 f"{signed(bq['delta_f1_mean'] * 100, 1)} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_app_seeds(G):
    lines = ["\\begin{tabular}{llccc}", "    \\toprule",
             "    \\textbf{Modell} & \\textbf{Regime} & \\textbf{Seed 42} & \\textbf{Seed 123} & \\textbf{Seed 456} \\\\",
             "    \\midrule"]
    for key in ["deberta-base", "deberta-large", "qwen08-qlora", "qwen4-qlora", "qwen27-qlora"]:
        g = G[key]
        a = g["agg"]
        test = " & ".join(de(v * 100, 2) for v in a["f1"]["values"])
        lines.append(f"    {g['label']} & {g['regime']} & {test} \\\\")
        val = " & ".join(f"{de(v * 100, 2)} ({int(e)})" for v, e in zip(a["val_f1"], a["best_epochs"]))
        lines.append(f"    \\quad Validierung (Epoche) & & {val} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_app_typen_prf(G):
    cols = ["deberta-large", "qwen27-qlora", "qwen27-zs"]
    lines = ["\\begin{tabular}{lrrrrrrrrrr}", "    \\toprule",
             "    & & \\multicolumn{3}{c}{\\textbf{DeBERTa-v3-large}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B QLoRA}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B Zero-Shot}} \\\\",
             "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
             "    \\textbf{Typ} & \\textbf{Support} & P & R & F$_1$ & P & R & F$_1$ & P & R & F$_1$ \\\\",
             "    \\midrule"]
    for t in TYPES:
        sup = int(G["deberta-base"]["agg"]["per_type"][t]["support"])
        cells = []
        for c in cols:
            pt = G[c]["agg"]["per_type"][t]
            cells += [de(pt["precision"]["mean"] * 100, 1), de(pt["recall"]["mean"] * 100, 1), de(pt["f1"]["mean"] * 100, 1)]
        lines.append(f"    {t:5s} & {sup:,}".replace(",", "\\,") + " & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_app_crossner_seeds(C):
    st = C["seed_table"]
    seeds = " & ".join(f"\\textbf{{{s}}}" for s in SEEDS)
    lines = ["\\begin{tabular}{lrrrrrrr}", "    \\toprule",
             "    & & \\multicolumn{3}{c}{\\textbf{DeBERTa-v3-large}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B (QLoRA)}} \\\\",
             "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             f"    \\textbf{{Dom\\\"ane}} & \\textbf{{S\\\"atze}} & {seeds} & {seeds} \\\\", "    \\midrule"]

    def cells(dom):
        return " & ".join(de(st[key][dom][s]["f1"] * 100, 2) for key in ("deberta-large", "qwen27-qlora") for s in SEEDS)

    for dom, label in DOMAINS:
        lines.append(f"    {label} & {C['samples'][dom]} & {cells(dom)} \\\\")
    lines.append("    \\midrule")
    lines.append(f"    Gepoolt & {fmt_int(sum(C['samples'].values()))} & {cells('pooled_micro')} \\\\")
    lines.append(f"    Dom\\\"anenmittel & -- & {cells('domain_mean')} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_app_crossner_typen(C):
    pt = C["per_type_domain"]
    types = " & ".join(COARSE)
    lines = ["\\begin{tabular}{lrrrrrrrrr}", "    \\toprule",
             "    & \\multicolumn{3}{c}{\\textbf{Support}} & \\multicolumn{3}{c}{\\textbf{DeBERTa-v3-large}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B (QLoRA)}} \\\\",
             "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             f"    \\textbf{{Dom\\\"ane}} & {types} & {types} & {types} \\\\", "    \\midrule"]
    for dom, label in DOMAINS:
        sup = " & ".join(fmt_int(pt["deberta-large"][dom][t]["support"]) for t in COARSE)
        if any(pt["qwen27-qlora"][dom][t]["support"] != pt["deberta-large"][dom][t]["support"] for t in COARSE):
            sys.exit(f"ABBRUCH: CrossNER-Support {dom} unterscheidet sich zwischen den Modellen")
        vals = " & ".join(de(pt[key][dom][t]["f1"][0] * 100, 1) for key in ("deberta-large", "qwen27-qlora") for t in COARSE)
        lines.append(f"    {label} & {sup} & {vals} \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_limit(C):
    """CrossNER mit Ausgabelimit 1024 gegen die Hauptlaeufe (256) auf denselben Saetzen; nur bei complete."""
    L = C["output_limit_full"]
    big = str(L["max_new_tokens"])
    lines = ["\\begin{tabular}{llrrrrrrr}", "    \\toprule",
             "    \\textbf{Dom\\\"ane} & \\textbf{Limit} & \\textbf{P} & \\textbf{R} & \\textbf{F$_1$} & $\\Delta$\\textbf{F$_1$} &",
             "      \\textbf{\\shortstack[c]{nicht pars-\\\\bar [\\%]}} & \\textbf{\\shortstack[c]{Limit er-\\\\reicht [\\%]}} &",
             "      \\textbf{Latenz [s]} \\\\", "    \\midrule"]

    def pct(r, field, nd=1):
        m, s = r.get(field + "_mean"), r.get(field + "_std")
        return "--" if m is None else pm(m * 100, None if s is None else s * 100, nd)

    def row(label, dom, size):
        r = L["full"][dom][size]
        if size == "256":
            delta, hit = "--", "--$^{a}$"
        else:
            m, s = r.get("delta_f1_vs_256_mean"), r.get("delta_f1_vs_256_std")
            delta = "--" if m is None else signed(m * 100, 1) + ("" if s is None else f"\\,$\\pm$\\,{de(s * 100, 1)}")
            hit = pct(r, "output_limit_rate")
        lat = "--" if r.get("latency_ms_mean") is None else de(r["latency_ms_mean"] / 1000, 1)
        return (f"    {label} & {fmt_int(int(size))} & {de(r['precision_mean'] * 100, 1)} & {de(r['recall_mean'] * 100, 1)} & "
                f"{pct(r, 'f1')} & {delta} & {pct(r, 'parse_failure_rate')} & {hit} & {lat} \\\\")

    for dom, label in DOMAINS:
        lines.append(row(label, dom, "256"))
        lines.append(row("", dom, big))
    lines.append("    \\midrule")
    lines.append(row("Gepoolt", "pooled_micro", "256"))
    lines.append(row("", "pooled_micro", big))
    lines.append("    \\bottomrule")
    lines += note_rows("$^{a}$ F\\\"ur die Hauptl\\\"aufe (Limit 256) wurden Ausgabel\\\"ange und Stoppgrund nicht "
                       "protokolliert; Limit-Treffer daher unbekannt.", 9)
    lines += ["\\end{tabular}", ""]
    return "\n".join(lines)


def tab_limit_multinerd(G, L):
    """MultiNERD: Basislaeufe (Limit 256) gegen Nachinferenz der abgeschnittenen Saetze mit Limit 1024."""
    lines = ["\\begin{tabular}{llrrrrrrr}", "    \\toprule",
             "    \\textbf{Modell} & \\textbf{Limit} & \\textbf{P} & \\textbf{R} & \\textbf{Mikro-F$_1$} &",
             "      $\\Delta$\\textbf{F$_1$} & \\textbf{\\shortstack[c]{neu\\\\erzeugt}} &",
             "      \\textbf{\\shortstack[c]{weiterhin\\\\abgeschnitten}} & \\textbf{\\shortstack[c]{Auslas-\\\\sungen [\\%]}} \\\\",
             "    \\midrule"]
    for key, base in LIMIT_GROUPS.items():
        if key not in L:
            continue
        st = L[key]
        big = fmt_int(G[key]["runs"][0].get("max_new_tokens") or 1024)
        for label, g, size in ((G[base]["label"], G[base], "256"), ("", G[key], big)):
            a = g["agg"]
            cells = [de(a["precision"]["mean"] * 100, 1), de(a["recall"]["mean"] * 100, 1),
                     pm(a["f1"]["mean"] * 100, None if a["f1"]["std"] is None else a["f1"]["std"] * 100, 2)]
            if size == "256":
                cells += ["--", "--", "--"]
            else:
                d = signed(st["delta_f1"]["mean"] * 100, 2)
                if st["delta_f1"]["std"] is not None:
                    d += f"\\,$\\pm$\\,{de(st['delta_f1']['std'] * 100, 2)}"
                cells += [d, fmt_int(st["n_regenerated"]["mean"]), fmt_int(st["n_limit_reached"]["mean"])]
            cells.append(de(a["errors"]["auslassung"]["mean"] * 100, 2))
            lines.append(f"    {label} & {size} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


CI_PAIRS = ["qwen27-qlora->deberta-large", "deberta-base->qwen27-qlora", "deberta-base->deberta-large",
            "qwen4-qlora->qwen27-qlora", "qwen08-qlora->qwen4-qlora", "qwen08-qlora->deberta-base",
            "deberta-base->qwen4-qlora"]
CI_PAIRS_1024 = ["qwen27-qlora-1024->deberta-large", "deberta-base->qwen27-qlora-1024",
                 "qwen08-qlora-1024->deberta-base"]


def tab_ci(G, boot):
    def name(key, with_regime=False):
        g = G[key]
        return g["label"] + (f" ({g['regime']})" if with_regime or not key.startswith("deberta") else "")

    def interval(lo, hi, nd=2):
        return f"[{signed(lo, nd)}; {signed(hi, nd)}]" + ("$^{\\dagger}$" if lo <= 0 <= hi else "")

    lines = ["\\begin{tabular}{lrlrl}", "    \\toprule",
             "    \\textbf{Vergleich (A $-$ B)} & $\\Delta$\\,\\textbf{Mikro-F$_1$} & \\textbf{95\\,\\%-KI} &",
             "      $\\Delta$\\,\\textbf{Makro-F$_1$} & \\textbf{95\\,\\%-KI} \\\\", "    \\midrule"]
    for pk in [p for p in CI_PAIRS + CI_PAIRS_1024 if p in boot["pairs"]]:
        k1, k2 = pk.split("->")
        p = boot["pairs"][pk]
        lines.append(f"    {name(k2)} $-$ {name(k1)} & {signed(p['diff'], 2)} & {interval(*p['ci'])} & "
                     f"{signed(p['macro_diff'], 2)} & {interval(*p['macro_ci'])} \\\\")
    lines += ["    \\midrule",
              "    \\textbf{Konfiguration} & \\textbf{Mikro-F$_1$} & \\textbf{95\\,\\%-KI} &",
              "      \\textbf{Makro-F$_1$} & \\textbf{95\\,\\%-KI} \\\\", "    \\midrule"]
    for key in TABLE_ORDER + [k for k in LIMIT_GROUPS if k in G]:
        c = boot["configs"][key]
        mi, ma = c["micro"]["ci_seed_mean"], c["macro"]["ci_seed_mean"]
        lines.append(f"    {name(key, True)} & {de(c['micro']['point'], 2)} & [{de(mi[0], 2)}; {de(mi[1], 2)}] & "
                     f"{de(c['macro']['point'], 2)} & [{de(ma[0], 2)}; {de(ma[1], 2)}] \\\\")
    lines.append("    \\bottomrule")
    s = boot["settings"]
    lines += note_rows(f"$^{{\\dagger}}$ Intervall schlie\\ss{{}}t 0 ein. Satz-Bootstrap, $B = {fmt_int(s['B'])}$, "
                       "gleiche Stichprobe f\\\"ur alle L\\\"aufe, seedgleiche Paare; Konfigurationen: Intervall des "
                       "Seed-Mittels.", 5)
    lines += ["\\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Abbildungsdaten
def fig_data(G):
    typen = ["type support f1_large f1_q27 f1_zs27"]
    for t in TYPES:
        typen.append(f"{t} {int(G['deberta-base']['agg']['per_type'][t]['support'])} "
                     f"{G['deberta-large']['agg']['per_type'][t]['f1']['mean'] * 100:.2f} "
                     f"{G['qwen27-qlora']['agg']['per_type'][t]['f1']['mean'] * 100:.2f} "
                     f"{G['qwen27-zs']['agg']['per_type'][t]['f1']['mean'] * 100:.2f}")
    groesse = ["series params f1 sd"]
    for key in TABLE_ORDER:
        g = G[key]
        a = g["agg"]["f1"]
        series = {"Fine-Tuning": "deberta", "Zero-Shot": "zeroshot", "QLoRA": "qlora"}[g["regime"]]
        groesse.append(f"{series} {g['params']:.4g} {a['mean'] * 100:.2f} {(a['std'] or 0) * 100:.2f}")
    tradeoff = ["key short regime family latency_ms vram_gb f1 sd gpu"]
    for key in TABLE_ORDER:
        g = G[key]
        a = g["agg"]
        short = g["label"].replace("DeBERTa-v3-", "D-").replace("Qwen3.5-", "Q")
        tradeoff.append(f"{key} {short} {g['regime']} {g['family']} {a['latency_mean']['mean']:.2f} "
                        f"{a['vram_mb']['mean'] / 1024:.2f} {a['f1']['mean'] * 100:.2f} {(a['f1']['std'] or 0) * 100:.2f} {g['gpu']}")
    return "\n".join(typen) + "\n", "\n".join(groesse) + "\n", "\n".join(tradeoff) + "\n"


# --------------------------------------------------------------------------- Hauptprogramm
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("MultiNERD-Laeufe auswerten ...")
    sacct = load_sacct(SACCT)
    if sacct is None:
        print(f"  Hinweis: Slurm-Abrechnung {SACCT.name} fehlt; Trainingszeiten aus Protokoll bzw. Hochrechnung")
    G = {}
    for key, meta in TRAINED.items():
        runs = [analyse_multinerd_run(RES / "multinerd" / r, not key.startswith("deberta"), seed, sacct_for(sacct, key, seed))
                for r, seed in zip(meta["runs"], SEEDS)]
        G[key] = dict(meta, agg=aggregate_group(runs), runs=runs)
        print(f"  {key}: F1 {G[key]['agg']['f1']['mean'] * 100:.2f} +- {G[key]['agg']['f1']['std'] * 100:.2f}")
    for key, meta in ZEROSHOT.items():
        run = analyse_multinerd_run(RES / "multinerd" / meta["run"], True)
        G[key] = dict(meta, agg=aggregate_group([run]), runs=[run])
        print(f"  {key}: F1 {G[key]['agg']['f1']['mean'] * 100:.2f}")
    # Nachinferenz mit Ausgabelimit 1024 (Step 5): nur vollstaendige Dreiergruppen, seedweise an die Basis gepaart
    L = {}
    if not LIMIT_ROOT.exists():
        print(f"  Hinweis: {LIMIT_ROOT.name} fehlt; 1024-Gruppen uebersprungen")
    else:
        for key, base in LIMIT_GROUPS.items():
            dirs = [LIMIT_ROOT / r for r in TRAINED[base]["runs"]]
            present = sum((d / "test_predictions.json").exists() for d in dirs)
            if present != len(dirs):
                print(f"  Hinweis: {key} unvollstaendig ({present}/{len(dirs)} Laeufe); uebersprungen")
                continue
            runs = [analyse_multinerd_run(d, True, seed) for d, seed in zip(dirs, SEEDS)]
            G[key] = dict(TRAINED[base], regime="QLoRA, Limit 1\\,024", agg=aggregate_group(runs), runs=runs)
            L[key] = paired_limit_stats(G[base]["runs"], runs)
            print(f"  {key}: F1 {G[key]['agg']['f1']['mean'] * 100:.2f} +- {G[key]['agg']['f1']['std'] * 100:.2f} "
                  f"(Delta zu 256: {L[key]['delta_f1']['mean'] * 100:+.2f})")
    print("CrossNER auswerten ...")
    C = analyse_crossner()

    # Satzstatistik MultiNERD-Test
    preds = load_json(RES / "multinerd" / "deberta-v3-large-canonical" / "test_predictions.json")
    mn_tokens = statistics.fmean(len(r["tokens"]) for r in preds)
    mn_ents = statistics.fmean(len(entities(r["gold"])) for r in preds)
    mn_coarse = statistics.fmean(sum(1 for e in entities(r["gold"]) if e[2] in COARSE) for r in preds)
    mn_gold_empty = sum(1 for r in preds if not entities(r["gold"]))
    greedy_dir = RES / "multinerd" / "qwen35-08b-zeroshot-greedy-v1"
    greedy_preds = load_json(greedy_dir / "test_predictions.json")
    greedy_im = load_yaml(greedy_dir / "inference_metrics.yaml")
    greedy = dict(f1=greedy_im["f1"], precision=greedy_im["precision"], recall=greedy_im["recall"],
                  empty_arrays=sum(1 for r in greedy_preds if r["raw_output"].strip() == "[]"),
                  n=len(greedy_preds), parse_failure_rate=greedy_im["parse_failure_rate"])

    # Deskriptive Differenzen; Unsicherheit ausschliesslich aus dem gepaarten Bootstrap.
    def compare(k1, k2):
        a1, a2 = G[k1]["agg"]["f1"], G[k2]["agg"]["f1"]
        diff = (a2["mean"] - a1["mean"]) * 100
        return dict(diff=diff)

    comparisons = {f"{k1}->{k2}": compare(k1, k2) for k1, k2 in [
        ("deberta-base", "deberta-large"), ("qwen08-qlora", "qwen4-qlora"), ("qwen4-qlora", "qwen27-qlora"),
        ("qwen27-qlora", "deberta-large"), ("deberta-base", "qwen4-qlora"), ("qwen08-qlora", "deberta-base"),
        ("qwen27-zs", "qwen27-qlora"), ("qwen4-zs", "qwen4-qlora"), ("qwen08-zs", "qwen08-qlora"),
        ("qwen08-qlora", "deberta-large"), ("deberta-base", "qwen27-qlora"),
        ("qwen27-qlora-1024", "deberta-large"), ("deberta-base", "qwen27-qlora-1024"),
        ("deberta-base", "qwen4-qlora-1024"), ("qwen08-qlora-1024", "deberta-base"),
        ("qwen27-qlora", "qwen27-qlora-1024"), ("qwen4-qlora", "qwen4-qlora-1024"),
        ("qwen08-qlora", "qwen08-qlora-1024")] if k1 in G and k2 in G}

    # Gepaartes Satz-Bootstrap: setzt identische Satzfolge und Referenz in allen Laeufen voraus
    sigs = {(r["sentence_signature"], r["gold_signature"]) for g in G.values() for r in g["runs"]}
    if len(sigs) != 1:
        sys.exit("ABBRUCH: Testsaetze oder Referenz nicht in allen Laeufen identisch; Paarung unmoeglich")
    print(f"Satz-Bootstrap (B = {BOOT_B}, Saat {BOOT_SEED}) ...")
    boot = bootstrap_intervals(G, list(comparisons))
    for key, comp in comparisons.items():
        k1, k2 = key.split("->")
        comp["single_run"] = len(G[k1]["runs"]) == 1 or len(G[k2]["runs"]) == 1
        if comp["single_run"]:
            comp["uncertainty_note"] = ("Zero-Shot: Einzellauf; Bootstrap bedingt auf die gespeicherten "
                                        "Ausgaben, keine Messung der Sampling-Streuung")
        comp["bootstrap"] = boot["pairs"][key]
    p = boot["pairs"]["qwen27-qlora->deberta-large"]
    print(f"  DeBERTa-large - Qwen27B (QLoRA): {p['diff']:+.2f} [{p['ci'][0]:+.2f}; {p['ci'][1]:+.2f}], "
          f"Makro {p['macro_diff']:+.2f} [{p['macro_ci'][0]:+.2f}; {p['macro_ci'][1]:+.2f}]")

    print(f"CrossNER-Satz-Bootstrap (B = {BOOT_B}, nach Domaene geschichtet) ...")
    C["bootstrap"] = crossner_bootstrap(C)
    print("Tabellen und Abbildungsdaten schreiben ...")
    write(OUT / "tab-haupt.tex", tab_haupt(G))
    write(OUT / "tab-typen.tex", tab_typen(G))
    write(OUT / "tab-groesse.tex", tab_groesse(G))
    write(OUT / "tab-effizienz.tex", tab_effizienz(G))
    write(OUT / "tab-formtreue.tex", tab_formtreue(G))
    write(OUT / "tab-fehlertypen.tex", tab_fehlertypen(G))
    write(OUT / "tab-crossner.tex", tab_crossner(C))
    write(OUT / "tab-crossner-ci.tex", tab_crossner_ci(C))
    write(OUT / "tab-app-seeds.tex", tab_app_seeds(G))
    write(OUT / "tab-app-typen-prf.tex", tab_app_typen_prf(G))
    write(OUT / "tab-ci.tex", tab_ci(G, boot))
    write(OUT / "tab-app-crossner-seeds.tex", tab_app_crossner_seeds(C))
    write(OUT / "tab-app-crossner-typen.tex", tab_app_crossner_typen(C))
    if C["output_limit_full"].get("complete"):
        write(OUT / "tab-limit.tex", tab_limit(C))
    if L:
        write(OUT / "tab-limit-multinerd.tex", tab_limit_multinerd(G, L))
    d_typen, d_groesse, d_tradeoff = fig_data(G)
    write(OUT / "fig-typen.dat", d_typen)
    write(OUT / "fig-groesse.dat", d_groesse)
    write(OUT / "fig-tradeoff.dat", d_tradeoff)

    # Trainingszeit 27B (Step 1): drei Quellen nebeneinander; die fruehere Hochrechnung der Arbeit benutzte
    # faelschlich den Recovery-Checkpoint (Wiederaufnahmepunkt der 2. Zuteilung) als Beginn des Protokollabschnitts.
    profs = G["qwen27-qlora"]["agg"]["training_profile"]
    recovery_steps = [int(re.search(r"checkpoint-(\d+)", rc).group(1)) if rc else None
                      for rc in G["qwen27-qlora"]["agg"]["recovery_checkpoints"]]
    training_time_27b = dict(
        total_steps=[p["total_steps"] for p in profs], resume_steps=[p["resume_step"] for p in profs],
        steps_logged=[p["steps_logged"] for p in profs],
        last_segment_h=[p["logged_seconds"] / 3600 for p in profs],
        last_allocation_h=[p["last_allocation_seconds"] / 3600 for p in profs],
        seconds_per_step=[p["seconds_per_step"] for p in profs],
        estimate_h=[p["estimated_total_seconds"] / 3600 for p in profs],
        sacct_h=[p["sacct_train_seconds"] / 3600 if p["sacct_train_seconds"] else None for p in profs],
        sacct_allocations=[p["sacct_train_allocations"] for p in profs],
        sacct_infer_h=[p["sacct_infer_seconds"] / 3600 if p["sacct_infer_seconds"] else None for p in profs],
        wallclock_span_h=[p["wallclock_span_seconds"] / 3600 for p in profs],
        n_allocations=[p["n_allocations"] for p in profs],
        allocation_starts=[p["allocation_starts"] for p in profs],
        completed_at=[p["completed_at"] for p in profs],
        recovery_checkpoints=G["qwen27-qlora"]["agg"]["recovery_checkpoints"],
        previous_thesis_estimate_h=[p["logged_seconds"] / (p["total_steps"] - rs) * p["total_steps"] / 3600
                                    if rs is not None else None for p, rs in zip(profs, recovery_steps)],
        previous_thesis_estimate_note="Hochrechnung mit dem Recovery-Checkpoint als Beginn des Protokollabschnitts "
                                      "(fehlerhaft; Beginn ist results.yaml:resume_checkpoint)",
        exact=profs[0]["exact"], estimated=profs[0]["estimated"],
        primary_source="sacct" if sacct is not None else "estimate")
    inference_wallclock_check = {}
    for g in G.values():
        for r in g["runs"]:
            p = r.get("training_profile")
            wall, est = r.get("inference_seconds"), r.get("inference_seconds_latency_estimate")
            inference_wallclock_check[r["name"]] = dict(
                wall_h=wall / 3600 if wall else None, latency_estimate_h=est / 3600 if est else None,
                diff_h=(wall - est) / 3600 if wall and est else None, source=r.get("inference_seconds_source"),
                sacct_h=p["sacct_infer_seconds"] / 3600 if p and p.get("sacct_infer_seconds") else None)

    summary = dict(
        groups={k: dict(label=v["label"], regime=v["regime"], gpu=v["gpu"], params=v["params"], agg=v["agg"],
                        examples=[r["examples"] for r in v["runs"]],
                        generation_settings=v["runs"][0].get("generation_settings"))
                for k, v in G.items()},
        comparisons=comparisons,
        crossner=C,
        multinerd_sentence_stats=dict(tokens_mean=mn_tokens, entities_mean=mn_ents, coarse_entities_mean=mn_coarse,
                                      n_sentences=len(preds), gold_empty_sentences=mn_gold_empty),
        greedy_control_08b=greedy,
        training_time_27b=training_time_27b,
        inference_wallclock_check=inference_wallclock_check,
        bootstrap=boot,
        output_limit_multinerd=dict(available=list(L), missing=[k for k in LIMIT_GROUPS if k not in L], per_size=L),
    )

    def default(o):
        if isinstance(o, Counter):
            return {f"{k[0]}->{k[1]}" if isinstance(k, tuple) else k: v for k, v in o.items()}
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=default), encoding="utf-8")
    print(f"  geschrieben: thesis/generated/summary.json")
    print("Fertig.")


if __name__ == "__main__":
    main()
