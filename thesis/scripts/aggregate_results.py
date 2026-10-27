#!/usr/bin/env python3
"""Aggregiert die Laufartefakte unter ba-ner/results/ zu den Tabellen,
Abbildungsdaten und Kennzahlen fuer Kapitel 5 der Bachelorarbeit.

Aufruf aus dem Repo-Stamm:
    python3 thesis/scripts/aggregate_results.py
Benoetigt PyYAML (pip3 install --user pyyaml).

Eingaben:  ba-ner/results/{multinerd,crossner,crossner_output_limit_1024}
           ba-ner/configs/crossner_overlap_exclusions.yaml
Ausgaben:  thesis/generated/summary.json  (alle Kennzahlen)
           thesis/generated/tab-*.tex     (tabular-Fragmente fuer \\input)
           thesis/generated/fig-*.dat     (Datentabellen fuer pgfplots)

Alle F1-, Precision- und Recall-Werte werden strikt auf Entitaetsebene nach
dem IOB2-Schema berechnet (wie seqeval im strikten Modus) und gegen die
gespeicherten Werte der Laeufe geprueft. Mittelwerte und
Stichprobenstandardabweichungen (ddof=1) ueber die Seeds 42, 123, 456.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

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


# --------------------------------------------------------------------------- MultiNERD-Laeufe
def analyse_multinerd_run(run_dir: Path, is_llm: bool):
    preds = load_json(run_dir / "test_predictions.json")
    gold_key, pred_key = ("gold_bio", "pred_bio") if is_llm else ("gold", "pred")
    overall, per_type = score(preds, gold_key, pred_key)
    im = load_yaml(run_dir / "inference_metrics.yaml")
    stored_f1 = im.get("test_f1", im.get("f1"))
    if abs(stored_f1 - overall[2]) > 1e-9:
        sys.exit(f"ABBRUCH: F1-Abweichung in {run_dir.name}: eigen {overall[2]:.8f}, gespeichert {stored_f1:.8f}")
    ev_path = run_dir / "evaluation_metrics.yaml"
    if ev_path.exists():
        ev = load_yaml(ev_path)
        for t in TYPES:
            if abs(ev["per_entity"][t]["f1"] - per_type[t]["f1"]) > 1e-9:
                sys.exit(f"ABBRUCH: Typ-F1-Abweichung {run_dir.name}/{t}")
    macro = statistics.fmean(per_type[t]["f1"] for t in TYPES)
    out = dict(name=run_dir.name, precision=overall[0], recall=overall[1], f1=overall[2], macro_f1=macro,
               per_type=per_type, n_sentences=len(preds),
               n_gold=sum(per_type[t]["support"] for t in TYPES),
               latency_mean=im.get("latency_ms_mean"), latency_p50=im.get("latency_ms_p50"),
               latency_p95=im.get("latency_ms_p95"), vram_mb=im.get("vram_peak_mb"),
               throughput=im.get("generation_samples_per_second"), total_params=im.get("total_params"),
               gpu=(im.get("run_metadata", {}).get("cuda", {}).get("devices") or [None])[0],
               commit=im.get("run_metadata", {}).get("git", {}).get("commit"))
    out["errors"] = error_profile(preds, gold_key, pred_key)
    res_path = run_dir / "results.yaml"
    if res_path.exists():
        res = load_yaml(res_path)
        out["train_seconds"] = res.get("train_runtime_seconds")
        out["best_epoch"] = res.get("best_epoch")
        out["val_f1"] = res.get("best_validation_f1", res.get("best_dev_f1"))
        out["epoch_results"] = res.get("epoch_results")
        recov = list(run_dir.glob("recovery-*.yaml"))
        out["train_lower_bound"] = bool(recov)
        out["resume_checkpoint"] = load_yaml(recov[0]).get("validated_checkpoint") if recov else None
    if is_llm:
        status = Counter(r["parse_status"] for r in preds)
        failed_no_close = sum(1 for r in preds if r["parse_status"] == "failed"
                              and not r.get("raw_output", "").rstrip().endswith("]"))
        empty = sum(1 for r in preds if not r.get("pred_entities"))
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
        out["form"] = dict(
            n=len(preds), parse_ok=im["parse_ok"], markdown=im["parse_markdown_stripped"],
            regex=im["parse_regex_fallback"], failed=im["parse_failed"], failure_rate=im["parse_failure_rate"],
            wrong_schema=im["parse_wrong_schema"], invalid_items=im["parse_invalid_items"],
            missing_fields=im["parse_missing_fields"], invalid_offsets=im["parse_invalid_offsets"],
            unknown_types=im["parse_unknown_types"], text_mismatches=im["parse_text_mismatches"],
            overlaps=im["parse_overlaps"], failed_no_close=failed_no_close, empty=empty,
            status=dict(status), unknown_examples=unknown.most_common(10))
        out["generation_settings"] = im.get("generation_settings")
        out["max_new_tokens"] = im.get("max_new_tokens")
    # Beispiele fuer die Fehlertabelle (kurze Saetze)
    out["examples"] = collect_examples(preds, gold_key, pred_key, is_llm)
    return out


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
            "throughput", "train_seconds"]
    agg = {}
    for k in keys:
        vals = [r.get(k) for r in runs]
        m, s = mean_std(vals)
        agg[k] = dict(mean=m, std=s, values=vals)
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
    return agg


# --------------------------------------------------------------------------- CrossNER
def analyse_crossner():
    agg_file = load_yaml(RES / "crossner" / "aggregate" / "summary.yaml")
    excluded = set(load_yaml(CFG / "crossner_overlap_exclusions.yaml")["excluded_sample_ids"])
    out = dict(aggregate={}, sensitivity={}, parse={}, latency={}, gpus={}, sentence_stats={}, per_seed={})
    for key, name in CROSS_AGG_NAMES.items():
        out["aggregate"][key] = {row["domain"]: row for row in agg_file["summary"] if row["model"] == name}
    for key, runs in CROSS_RUNS.items():
        is_llm = key.startswith("qwen")
        pooled_full, pooled_excl, fails, no_close, lat_mean, lat_median, n_total = [], [], [], [], [], [], []
        gpus = set()
        tokens_per_sentence, ents_per_sentence = [], []
        per_domain_seed = {}
        for run in runs:
            records = []
            for dom, _ in DOMAINS:
                dom_dir = RES / "crossner" / run / dom
                recs = load_json(dom_dir / "test_predictions.json")
                records.extend(recs)
                im = load_yaml(dom_dir / "inference_metrics.yaml")
                dev = (im.get("run_metadata", {}).get("cuda", {}).get("devices") or [None])[0]
                if dev:
                    gpus.add(dev)
                ov, _ = score(recs, "gold_bio", "pred_bio", COARSE)
                per_domain_seed[(run, dom)] = ov
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
    return out


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
            "    \\textbf{Recall} & \\textbf{Micro-F$_1$} & \\textbf{Makro-F$_1$} \\\\\n    \\midrule\n"
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
    lines.append(f"    Micro-F$_1$ (alle) & {n_gold:,}".replace(",", "\\,") + " & "
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


def tab_effizienz(G):
    lines = ["\\begin{tabular}{llrrrrrrr}", "    \\toprule",
             "    \\textbf{Modell} & \\textbf{Regime} & \\textbf{GPU} &",
             "    \\textbf{Training} & \\multicolumn{3}{c}{\\textbf{Latenz [ms]}} &",
             "    \\textbf{Durchsatz} & \\textbf{Speicher} \\\\", "    \\cmidrule(lr){5-7}",
             "    & & & \\textbf{[h]} & \\textbf{Mittel} & \\textbf{Median} & \\textbf{P95} &",
             "    \\textbf{[S\\\"atze/s]} & \\textbf{[GB]} \\\\", "    \\midrule"]
    for i, key in enumerate(TABLE_ORDER):
        g = G[key]
        a = g["agg"]
        if i in (2, 5):
            lines.append("    \\midrule")
        enc = key.startswith("deberta")
        if a["train_seconds"]["mean"] is None:
            train = "--"
        else:
            train = de(a["train_seconds"]["mean"] / 3600, 1)
            if a.get("train_lower_bound"):
                train = "$\\geq$\\," + train + "$^{a}$"
        nd = 1 if enc else 0
        lat = de(a["latency_mean"]["mean"], nd)
        med = de(a["latency_p50"]["mean"], nd) if a["latency_p50"]["mean"] is not None else "--$^{b}$"
        p95 = de(a["latency_p95"]["mean"], nd)
        thr = "--" if enc else de(a["throughput"]["mean"], 2)
        mem = de(a["vram_mb"]["mean"] / 1024, 1)
        lines.append(f"    {g['label']} & {g['regime']} & {g['gpu']} & {train} & {lat} & {med} & {p95} & {thr} & {mem} \\\\")
    lines += ["    \\bottomrule",
              "    \\multicolumn{9}{@{}l}{\\scriptsize $^{a}$ Nur der Abschnitt nach der",
              "      Wiederaufnahme protokolliert; Untergrenze",
              "      (Abschnitt~\\ref{sec:effizienzmetriken}).} \\\\",
              "    \\multicolumn{9}{@{}l}{\\scriptsize $^{b}$ F\\\"ur die Zero-Shot-L\\\"aufe wurde kein",
              "      Median protokolliert.} \\\\",
              "\\end{tabular}", ""]
    return "\n".join(lines)


def tab_formtreue(G):
    cols = ["qwen08-zs", "qwen4-zs", "qwen27-zs", "qwen08-qlora", "qwen4-qlora", "qwen27-qlora"]

    def val(key, field, pct=False):
        v = G[key]["agg"]["form"][field]["mean"]
        if pct:
            return de(v * 100, 2)
        return f"{round(v):,}".replace(",", "\\,")

    lines = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.4cm}rrrrrr}", "    \\toprule",
             "    & \\multicolumn{3}{c}{\\textbf{Zero-Shot}} &", "      \\multicolumn{3}{c}{\\textbf{QLoRA}} \\\\",
             "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "    \\textbf{Kennzahl} & \\textbf{0.8B} & \\textbf{4B} & \\textbf{27B} &",
             "      \\textbf{0.8B} & \\textbf{4B} & \\textbf{27B} \\\\", "    \\midrule"]
    rows = [("Parse-Fehlerrate [\\%]", lambda k: val(k, "failure_rate", pct=True)),
            ("\\quad davon ohne schlie\\ss{}ende Klammer", lambda k: val(k, "failed_no_close")),
            ("Parse direkt", lambda k: val(k, "parse_ok")),
            ("\\quad nach Entfernen eines Codeblocks", lambda k: val(k, "markdown")),
            ("\\quad durch Extraktion aus Text", lambda k: val(k, "regex")),
            ("Falsche Struktur (kein Array)", lambda k: val(k, "wrong_schema")),
            ("Nicht-Objekte", lambda k: val(k, "invalid_items")),
            ("Fehlende Felder", lambda k: val(k, "missing_fields")),
            ("Unzul\\\"assige Indizes", lambda k: val(k, "invalid_offsets")),
            ("Unbekannte Typen", lambda k: val(k, "unknown_types")),
            ("Text-Abweichungen", lambda k: val(k, "text_mismatches")),
            ("\\\"Uberlappungen", lambda k: val(k, "overlaps"))]
    for label, fn in rows:
        lines.append(f"    {label} & " + " & ".join(fn(k) for k in cols) + " \\\\")
    lines.append("    \\midrule")
    lines.append("    Leere Vorhersagen & " + " & ".join(val(k, "empty") for k in cols) + " \\\\")
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
    lines = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.2cm}rcccccc}", "    \\toprule",
             "    & & \\multicolumn{3}{c}{\\textbf{DeBERTa-v3-large}} &",
             "      \\multicolumn{3}{c}{\\textbf{Qwen3.5-27B (QLoRA)}} \\\\",
             "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "    \\textbf{Dom\\\"ane} & \\textbf{S\\\"atze} & \\textbf{P} & \\textbf{R} & \\textbf{F$_1$} &",
             "      \\textbf{P} & \\textbf{R} & \\textbf{F$_1$} \\\\", "    \\midrule"]
    counts = {"ai": 431, "literature": 416, "music": 465, "politics": 651, "science": 543}

    def trio(key, dom):
        r = agg[key][dom]
        return (f"{de(r['precision_mean'] * 100, 1)} & {de(r['recall_mean'] * 100, 1)} & "
                f"{pm(r['f1_mean'] * 100, r['f1_std'] * 100, 1)}")

    for dom, label in DOMAINS:
        lines.append(f"    {label} & {counts[dom]} & {trio('deberta-large', dom)} & {trio('qwen27-qlora', dom)} \\\\")
    lines.append("    \\midrule")
    lines.append(f"    Gepoolt & 2\\,506 & {trio('deberta-large', 'pooled_micro')} & {trio('qwen27-qlora', 'pooled_micro')} \\\\")
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
    G = {}
    for key, meta in TRAINED.items():
        runs = [analyse_multinerd_run(RES / "multinerd" / r, not key.startswith("deberta")) for r in meta["runs"]]
        G[key] = dict(meta, agg=aggregate_group(runs), runs=runs)
        print(f"  {key}: F1 {G[key]['agg']['f1']['mean'] * 100:.2f} +- {G[key]['agg']['f1']['std'] * 100:.2f}")
    for key, meta in ZEROSHOT.items():
        run = analyse_multinerd_run(RES / "multinerd" / meta["run"], True)
        G[key] = dict(meta, agg=aggregate_group([run]), runs=[run])
        print(f"  {key}: F1 {G[key]['agg']['f1']['mean'] * 100:.2f}")
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

    # Abgeleitete Vergleiche (Unterscheidbarkeit: Differenz > Summe der SDs)
    def compare(k1, k2):
        a1, a2 = G[k1]["agg"]["f1"], G[k2]["agg"]["f1"]
        diff = (a2["mean"] - a1["mean"]) * 100
        sd_sum = ((a1["std"] or 0) + (a2["std"] or 0)) * 100
        return dict(diff=diff, sd_sum=sd_sum, distinguishable=abs(diff) > sd_sum)

    comparisons = {f"{k1}->{k2}": compare(k1, k2) for k1, k2 in [
        ("deberta-base", "deberta-large"), ("qwen08-qlora", "qwen4-qlora"), ("qwen4-qlora", "qwen27-qlora"),
        ("qwen27-qlora", "deberta-large"), ("deberta-base", "qwen4-qlora"), ("qwen08-qlora", "deberta-base"),
        ("qwen27-zs", "qwen27-qlora"), ("qwen4-zs", "qwen4-qlora"), ("qwen08-zs", "qwen08-qlora"),
        ("qwen08-qlora", "deberta-large"), ("deberta-base", "qwen27-qlora")]}

    print("Tabellen und Abbildungsdaten schreiben ...")
    write(OUT / "tab-haupt.tex", tab_haupt(G))
    write(OUT / "tab-typen.tex", tab_typen(G))
    write(OUT / "tab-groesse.tex", tab_groesse(G))
    write(OUT / "tab-effizienz.tex", tab_effizienz(G))
    write(OUT / "tab-formtreue.tex", tab_formtreue(G))
    write(OUT / "tab-fehlertypen.tex", tab_fehlertypen(G))
    write(OUT / "tab-crossner.tex", tab_crossner(C))
    write(OUT / "tab-app-seeds.tex", tab_app_seeds(G))
    write(OUT / "tab-app-typen-prf.tex", tab_app_typen_prf(G))
    d_typen, d_groesse, d_tradeoff = fig_data(G)
    write(OUT / "fig-typen.dat", d_typen)
    write(OUT / "fig-groesse.dat", d_groesse)
    write(OUT / "fig-tradeoff.dat", d_tradeoff)

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
    )

    def default(o):
        if isinstance(o, Counter):
            return {f"{k[0]}->{k[1]}" if isinstance(k, tuple) else k: v for k, v in o.items()}
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(str(type(o)))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=default), encoding="utf-8")
    print(f"  geschrieben: thesis/generated/summary.json")
    print("Fertig.")


if __name__ == "__main__":
    main()
