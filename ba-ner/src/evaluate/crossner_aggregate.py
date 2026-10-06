"""Verified per-domain and pooled CrossNER tables, including paired baselines."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics

import yaml

from src.data.crossner_loader import COMMON_TYPES, DOMAINS, load_crossner_test
from src.evaluate.crossner import (
    GROUPS, SEEDS, baseline_metrics, completed_domain, ensure_safe_output,
    evaluation_contract, selected_runs, source_run,
)
from src.evaluate.metrics import compute_ner_metrics
from src.seed_study import atomic_write_bytes, atomic_write_yaml


def summarize(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def summary_row(model: str, domain: str, records: list[dict], expected: int) -> dict:
    row = {"model": model, "domain": domain, "successful_seeds": len(records), "expected_seeds": expected,
           "seeds": ",".join(str(r["seed"]) for r in records), "complete": len(records) == expected}
    for key in ("precision", "recall", "f1", "baseline_f1", "delta_f1"):
        row.update({f"{key}_{stat}": value for stat, value in summarize([r[key] for r in records]).items()})
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_write_bytes(path, b"")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def aggregate(source_root: Path, output_root: Path, cache_dir: Path, models=GROUPS, seeds=SEEDS, domains=DOMAINS,
              *, contract_factory=None, primary_models=GROUPS) -> dict:
    ensure_safe_output(output_root, source_root)
    contract_factory = contract_factory or evaluation_contract
    runs = selected_runs(models, seeds)
    datasets = {domain: load_crossner_test(domain, cache_dir) for domain in domains}
    raw_rows, class_rows, overall_rows, missing = [], [], [], []
    for run in runs:
        try:
            source = source_run(run, source_root)
            baseline = baseline_metrics(source)
        except (OSError, ValueError, KeyError) as exc:
            missing.extend({"model": run.group_key, "seed": run.seed, "domain": d, "reason": str(exc)} for d in domains)
            continue
        pooled_true, pooled_pred, domain_f1 = [], [], []
        for domain in domains:
            directory = output_root / source["config"]["experiment_name"] / domain
            try:
                contract = contract_factory(source, domain)
                if not completed_domain(directory, contract, datasets[domain]):
                    raise ValueError("Missing final metrics")
                metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
                if not metrics.get("complete") or metrics["sample_count"] != len(datasets[domain]):
                    raise ValueError("Smoke or incomplete evaluation is excluded")
                records = json.loads((directory / "test_predictions.json").read_text())
                true, pred = [r["gold_bio"] for r in records], [r["pred_bio"] for r in records]
                quality = compute_ner_metrics(true, pred)
                if any(abs(quality[k] - metrics[k]) > 1e-12 for k in quality):
                    raise ValueError("Saved metrics differ from verified predictions")
                if metrics["multinerd_baseline"] != baseline or abs(metrics["delta_f1"] - (quality["f1"] - baseline["f1"])) > 1e-12:
                    raise ValueError("Paired MultiNERD baseline mismatch")
                raw_rows.append({"model": run.group_key, "seed": run.seed, "domain": domain,
                                 "samples": len(records), **quality, "baseline_f1": baseline["f1"],
                                 "delta_f1": quality["f1"] - baseline["f1"]})
                for kind in COMMON_TYPES:
                    class_rows.append({"model": run.group_key, "seed": run.seed, "domain": domain,
                                       "entity_type": kind, **metrics["per_entity"][kind]})
                pooled_true.extend(true)
                pooled_pred.extend(pred)
                domain_f1.append(quality["f1"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append({"model": run.group_key, "seed": run.seed, "domain": domain, "reason": str(exc)})
        if len(domain_f1) == len(domains):
            pooled = compute_ner_metrics(pooled_true, pooled_pred)
            overall_rows.append({"model": run.group_key, "seed": run.seed, "domain": "pooled_micro",
                                 "samples": len(pooled_true), **pooled, "baseline_f1": baseline["f1"],
                                 "delta_f1": pooled["f1"] - baseline["f1"]})
            macro = statistics.fmean(domain_f1)
            overall_rows.append({"model": run.group_key, "seed": run.seed, "domain": "domain_mean",
                                 "samples": len(pooled_true), "precision": statistics.fmean(r["precision"] for r in raw_rows if r["model"] == run.group_key and r["seed"] == run.seed),
                                 "recall": statistics.fmean(r["recall"] for r in raw_rows if r["model"] == run.group_key and r["seed"] == run.seed),
                                 "f1": macro, "baseline_f1": baseline["f1"], "delta_f1": macro - baseline["f1"]})
    summaries = [summary_row(model, domain, [r for r in raw_rows + overall_rows if r["model"] == model and r["domain"] == domain], len(seeds))
                 for model in models for domain in (*domains, "pooled_micro", "domain_mean")]
    primary = set(models) == set(primary_models) and set(seeds) == set(SEEDS) and set(domains) == set(DOMAINS)
    report = {"complete": primary and not missing, "selection_complete": not missing,
              "successful_evaluations": len(raw_rows), "expected_evaluations": len(runs) * len(domains),
              "primary_expected_evaluations": len(selected_runs(primary_models, SEEDS)) * len(DOMAINS),
              "ddof": 1, "scored_types": list(COMMON_TYPES),
              "interpretation": "Transfer of PER/ORG/LOC between datasets with frozen models and prompts",
              "selection": {"models": list(models), "seeds": list(seeds), "domains": list(domains)},
              "summary": summaries, "missing_or_failed": missing}
    target = output_root / "aggregate"
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(target / "summary.yaml", report)
    atomic_write_yaml(target / "missing_or_failed_runs.yaml", missing)
    write_csv(target / "summary.csv", summaries)
    write_csv(target / "seed_metrics.csv", raw_rows + overall_rows)
    write_csv(target / "per_entity_metrics.csv", class_rows)

    def display(row, key):
        mean, sd = row[key + "_mean"], row[key + "_std"]
        if mean is None:
            return "—"
        return f"{mean * 100:.2f}" + (f" ± {sd * 100:.2f}" if sd is not None else " (SD n/a)")

    lines = ["# CrossNER domain transfer: PER / ORG / LOC", "",
             f"Status: {'COMPLETE' if report['complete'] else 'INCOMPLETE / SUBSET'}; {len(raw_rows)}/{len(runs) * len(domains)} evaluations.", "",
             "All values are percentages (mean ± sample SD across training seeds); ΔF1 is CrossNER minus MultiNERD in percentage points.", "",
             "| Model | Domain | Seeds | Precision | Recall | CrossNER F1 | MultiNERD F1 | ΔF1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        values = [display(row, k) for k in ("precision", "recall", "f1", "baseline_f1", "delta_f1")]
        lines.append(f"| {row['model']} | {row['domain']} | {row['successful_seeds']}/{row['expected_seeds']} | " + " | ".join(values) + " |")
    lines += ["", "pooled_micro concatenates all domain predictions per seed; domain_mean weights every selected domain equally.",
              "The MultiNERD baseline uses the same three classes. These scores describe transfer of PER/ORG/LOC, not the complete CrossNER taxonomy.", ""]
    if missing:
        lines += ["Missing or failed evaluations:", ""] + [f"- {m['model']}, seed {m['seed']}, {m['domain']}: {m['reason']}" for m in missing]
    atomic_write_bytes(target / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args(argv)
    report = aggregate(args.source_results_root, args.output_root, args.cache_dir, tuple(dict.fromkeys(args.models)),
                       tuple(dict.fromkeys(args.seeds)), tuple(dict.fromkeys(args.domains)))
    print(f"CROSSNER_AGGREGATE {report['successful_evaluations']}/{report['expected_evaluations']} complete={report['complete']}")
    return 0 if report["selection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
