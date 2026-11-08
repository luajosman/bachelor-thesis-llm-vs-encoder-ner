"""Control sample for the MultiNERD output-limit correction: re-generate untouched answers at a higher limit.

The merged 1,024-token result (``src.evaluate.multinerd_output_limit``) keeps every stored answer that was
parsed at 256 tokens and re-generates only the stored parse failures.  That rests on one premise: under
greedy decoding an answer that ended before 256 tokens comes out unchanged at a higher limit.  This module
tests the premise and measures inference cost on one GPU type.  For one canonical Qwen QLoRA run it
re-generates a seeded random sample of test sentences whose stored answer was *not* a parse failure, with
the same frozen adapter, pinned base model, prompt, parser and greedy settings at the higher limit, and
records per sentence whether the new answer equals the stored one, the generated token count, the stop
reason and the latency (batch size 1, cuda-synchronized, warm-up excluded).

Modes: ``--config`` (one GPU job per canonical run) and ``--aggregate`` (CPU summary of finished runs,
including a full-test latency estimate that combines the sample with the re-generated parse failures).

The module is additive: the canonical artifacts, the output-limit evaluation and the hashed scientific code
remain unchanged.  Source runs are read-only; only the separate output root is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import yaml

from src.config import load_experiment_config
from src.decoder.parse_output import entities_to_bio
from src.evaluate import multinerd_output_limit as mnol
from src.evaluate.crossner import PROJECT_ROOT, SEEDS, file_hash, source_run
from src.evaluate.crossner_aggregate import write_csv
from src.evaluate.crossner_output_limit import INVALID_ARTIFACT_ERRORS, _check_limit
from src.seed_provenance import canonical_hash
from src.seed_study import atomic_write_bytes, atomic_write_json, atomic_write_yaml

SAMPLE_SIZE = 1000
SAMPLE_SEED = 20261108
SELECTION_RULE = ("seeded random permutation of all test indices; the first N whose stored parse_status "
                  "is not failed; sorted")
CHECKPOINT_NAME = "control_checkpoint.jsonl"
JOB_SCRIPT = "scripts/cluster/job_multinerd_control_sample.sh"
CODE_PATHS = mnol.CODE_PATHS + ("src/evaluate/multinerd_output_limit.py",)
LATENCY_MEASUREMENT = "per sampled sentence, batch size 1, cuda-synchronized, warm-up excluded"
RECORD_KEYS = frozenset({"index", "stored_parse_status", "raw_output", "parse_status", "pred_entities",
                         "raw_output_equal_to_stored", "pred_entities_equal_to_stored", "stored_prefix_match",
                         "generated_token_count", "token_limit_reached", "stop_reason", "json_complete", "latency_ms"})


# ---------------------------------------------------------------------------
# Selection and contract
# ---------------------------------------------------------------------------

def control_indices(records, sample_size: int = SAMPLE_SIZE, sample_seed: int = SAMPLE_SEED) -> list:
    """Seeded random sample of sentences whose stored answer was parsed (the same permutation for every run)."""
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    order = list(range(len(records)))
    random.Random(sample_seed).shuffle(order)
    chosen = [index for index in order if records[index]["parse_status"] != "failed"][:sample_size]
    if len(chosen) < sample_size:
        raise ValueError(f"Only {len(chosen)} stored answers without parse failure for a sample of {sample_size}")
    return sorted(chosen)


def control_contract(source: dict, selection, *, max_new_tokens: int, sample_size: int, sample_seed: int) -> dict:
    contract = mnol.regeneration_contract(source, selection, max_new_tokens=max_new_tokens)
    contract["evaluation"] = f"multinerd_control_sample_{max_new_tokens}"
    contract["selection"] = {"rule": SELECTION_RULE, "count": len(contract["selection"]["indices"]),
                             "indices": contract["selection"]["indices"], "sample_size": sample_size,
                             "sample_seed": sample_seed}
    contract["code_sha256"] = {path: file_hash(PROJECT_ROOT / path) for path in CODE_PATHS}
    return contract


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def control_record(stored: dict, index: int, generated: dict) -> dict:
    return {
        "index": index, "stored_parse_status": stored["parse_status"],
        "raw_output": generated["raw_output"], "parse_status": generated["parse_status"],
        "pred_entities": generated["pred_entities"],
        "raw_output_equal_to_stored": generated["raw_output"] == stored["raw_output"],
        "pred_entities_equal_to_stored": generated["pred_entities"] == stored["pred_entities"],
        **{key: generated[key] for key in ("stored_prefix_match", "generated_token_count", "token_limit_reached",
                                           "stop_reason", "json_complete", "latency_ms")},
    }


def validate_control_record(record: dict, stored: dict, index: int, max_new_tokens: int) -> None:
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise ValueError("Control record has unexpected keys")
    if record["index"] != index or record["stored_parse_status"] != stored["parse_status"]:
        raise ValueError("Control record does not belong to the stored sentence")
    if (not isinstance(record["raw_output"], str) or record["parse_status"] not in mnol.PARSE_STATUSES
            or not isinstance(record["pred_entities"], list)):
        raise ValueError("Control record has invalid parse fields")
    if (record["raw_output_equal_to_stored"] is not (record["raw_output"] == stored["raw_output"])
            or record["pred_entities_equal_to_stored"] is not (record["pred_entities"] == stored["pred_entities"])):
        raise ValueError("Control record has inconsistent comparison flags")
    count = record["generated_token_count"]
    if type(count) is not int or not 0 < count <= max_new_tokens:
        raise ValueError("Control record has an invalid generated token count")
    if record["token_limit_reached"] is not (count == max_new_tokens) or record["stop_reason"] not in mnol.STOP_REASONS:
        raise ValueError("Control record has inconsistent stop diagnostics")
    if type(record["json_complete"]) is not bool or type(record["stored_prefix_match"]) is not bool:
        raise ValueError("Control record has invalid flags")
    if not mnol._is_number(record["latency_ms"]) or record["latency_ms"] < 0:
        raise ValueError("Control record has an invalid latency")


def read_control_checkpoint(path: Path, contract: dict, selection, stored) -> list:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or json.loads(lines[0]) != mnol.checkpoint_header(contract):
        raise ValueError(f"{path}: control contract mismatch")
    limit = contract["decoder"]["max_new_tokens"]
    records = []
    for i, line in enumerate(lines[1:]):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 2 and not line.endswith(b"\n"):
                break  # An interrupted final append is the only accepted corruption.
            raise ValueError(f"{path}: corrupt checkpoint record {i}")
        if i >= len(selection):
            raise ValueError(f"{path}: too many checkpoint records")
        validate_control_record(record, stored[selection[i]], selection[i], limit)
        records.append(record)
    return records


def control_summary(records, stored, selection, max_new_tokens: int, *, vram_peak_mb=None, total_params=None) -> dict:
    if [record["index"] for record in records] != list(selection):
        raise ValueError("Control records do not match the selection")
    before = [stored[index] for index in selection]
    after = [{"gold_bio": stored[r["index"]]["gold_bio"], "pred_bio": entities_to_bio(stored[r["index"]]["tokens"], r["pred_entities"])}
             for r in records]
    reasons = Counter(record["stop_reason"] for record in records)
    statuses = Counter(record["parse_status"] for record in records)
    tokens = mnol._distribution([record["generated_token_count"] for record in records])
    latencies = mnol._distribution([record["latency_ms"] for record in records])
    return {
        "selection_rule": SELECTION_RULE, "sample_count": len(records),
        "raw_output_equal_count": sum(r["raw_output_equal_to_stored"] for r in records),
        "pred_entities_equal_count": sum(r["pred_entities_equal_to_stored"] for r in records),
        "stored_prefix_match_count": sum(r["stored_prefix_match"] for r in records),
        "raw_output_mismatch_indices": [r["index"] for r in records if not r["raw_output_equal_to_stored"]],
        "subset_metrics": {"stored_256": mnol._quality(before), f"control_{max_new_tokens}": mnol._quality(after)},
        "stored_parse_status_counts": dict(Counter(record["parse_status"] for record in before)),
        "parse_status_counts_after": {status: statuses.get(status, 0) for status in mnol.PARSE_STATUSES},
        "limit_hit_count": sum(r["token_limit_reached"] for r in records),
        "stop_reason_counts": {reason: reasons.get(reason, 0) for reason in mnol.STOP_REASONS},
        "json_complete_count": sum(r["json_complete"] for r in records),
        "generated_tokens": tokens,
        "latency_ms": {"mean": latencies["mean"], "p50": latencies["p50"], "p95": latencies["p95"],
                       "total_seconds": latencies["total"] / 1000, "measurement": LATENCY_MEASUREMENT},
        "vram_peak_mb": vram_peak_mb, "total_params": total_params,
        "warmup": "first sampled sentence, unmeasured",
    }


def completed_control(directory: Path, contract: dict) -> bool:
    metrics_path, predictions_path = Path(directory) / "control_metrics.yaml", Path(directory) / "control_predictions.json"
    if not metrics_path.exists():
        return False
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("contract_hash") != canonical_hash(contract) or metrics.get("contract") != contract:
        raise ValueError(f"{directory}: existing control contract differs")
    if not predictions_path.exists() or metrics.get("predictions_sha256") != file_hash(predictions_path):
        raise ValueError(f"{directory}: completed control predictions missing or altered")
    return True


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_control(run, source_root, output_root, *, max_new_tokens: int = 1024, sample_size: int = SAMPLE_SIZE,
                sample_seed: int = SAMPLE_SEED, resume_failed: bool = False) -> dict:
    from src.run_metadata import collect_run_metadata

    _check_limit(max_new_tokens)
    if run.group_key not in mnol.GROUPS or run.model_family != "decoder":
        raise ValueError("Only the canonical Qwen QLoRA runs have a control sample")
    source_root, output_root = Path(source_root), Path(output_root)
    mnol.ensure_output_safe(output_root, source_root)
    if output_root.resolve().parent == source_root.resolve() and output_root.name.startswith("multinerd_output_limit"):
        raise ValueError("The control sample must not share the output-limit evaluation root")
    cfg = load_experiment_config(PROJECT_ROOT / run.config_path)
    experiment = cfg["experiment_name"]
    directory = output_root / experiment
    started_at, started = mnol._utc_now(), time.monotonic()
    done = []
    status = {"status": "RUNNING", "evaluation": f"multinerd_control_sample_{max_new_tokens}", "model": run.group_key,
              "experiment_name": experiment, "seed": run.seed, "max_new_tokens": max_new_tokens,
              "sample_size": sample_size, "sample_seed": sample_seed, "job_id": os.environ.get("SLURM_JOB_ID"),
              "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0), "started_at": started_at,
              "updated_at": started_at, "completed_count": 0, "raw_output_equal_count": 0, "error": None}

    def update_status(**changes) -> None:
        status.update(changes)
        status.update(updated_at=mnol._utc_now(), elapsed_seconds=time.monotonic() - started, completed_count=len(done),
                      raw_output_equal_count=sum(r["raw_output_equal_to_stored"] for r in done))
        atomic_write_json(directory / "status.json", status)

    with mnol.run_lock(directory, resume_failed=resume_failed):
        update_status()
        try:
            source = source_run(run, source_root)
            stored = mnol.load_source_predictions(source)
            mnol.verify_dataset_alignment(stored, mnol.load_test_split())
            mnol.baseline_metrics_256(source, stored)  # proves the source predictions equal the canonical metrics
            selection = control_indices(stored, sample_size, sample_seed)
            contract = control_contract(source, selection, max_new_tokens=max_new_tokens,
                                        sample_size=sample_size, sample_seed=sample_seed)
            if completed_control(directory, contract):
                update_status(status="COMPLETED")
                print(f"SKIP_COMPLETED {experiment} seed={run.seed}", flush=True)
                return yaml.safe_load((directory / "control_metrics.yaml").read_text(encoding="utf-8"))
            atomic_write_json(directory / "contract.json", contract)
            checkpoint = directory / CHECKPOINT_NAME
            records = read_control_checkpoint(checkpoint, contract, selection, stored)
            mnol.write_checkpoint_prefix(checkpoint, contract, records)
            done.extend(records)
            update_status()
            predictor = mnol.RegenerationPredictor(source, max_new_tokens=max_new_tokens)
            first = selection[min(len(records), len(selection) - 1)]
            predictor.predict(first, stored[first]["tokens"], stored[first]["raw_output"])  # unmeasured warm-up
            with checkpoint.open("a", encoding="utf-8") as handle:
                for position in range(len(records), len(selection)):
                    index = selection[position]
                    generated = predictor.predict(index, stored[index]["tokens"], stored[index]["raw_output"])
                    record = control_record(stored[index], index, generated)
                    validate_control_record(record, stored[index], index, max_new_tokens)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    done.append(record)
                    update_status()
                    if len(records) % 50 == 0 or len(records) == len(selection):
                        print(f"CTRL_PROGRESS {experiment} seed={run.seed} {len(records)}/{len(selection)}", flush=True)
            predictions_path = directory / "control_predictions.json"
            atomic_write_json(predictions_path, records)
            vram = predictor.torch.cuda.max_memory_allocated() / 1024**2
            summary = control_summary(records, stored, selection, max_new_tokens,
                                      vram_peak_mb=vram, total_params=predictor.total_params)
            run_metadata = collect_run_metadata(cfg)
            job_script = PROJECT_ROOT / JOB_SCRIPT
            metrics = {
                "evaluation": f"control_sample_{max_new_tokens}", "experiment_name": experiment,
                "model_name": cfg["model_name"], "model": run.group_key, "seed": run.seed,
                "max_new_tokens": max_new_tokens, "baseline_max_new_tokens": mnol.BASELINE_LIMIT,
                "derived_from": {"directory": str(Path(source["directory"]).resolve()),
                                 "predictions_sha256": source["identity"]["source_predictions_sha256"],
                                 "checkpoint_sha256": source["identity"]["checkpoint_sha256"]},
                "control": summary,
                "environment": mnol.environment_report(source, run_metadata),
                "contract_hash": canonical_hash(contract), "contract": contract,
                "predictions_sha256": file_hash(predictions_path),
                "code_hashes": {"src/evaluate/multinerd_control_sample.py": file_hash(Path(__file__)),
                                JOB_SCRIPT: file_hash(job_script) if job_script.is_file() else None,
                                **contract["code_sha256"]},
                "run_metadata": run_metadata, "job_id": os.environ.get("SLURM_JOB_ID"),
                "started_at": started_at, "finished_at": mnol._utc_now(), "elapsed_seconds": time.monotonic() - started,
            }
            atomic_write_yaml(directory / "control_metrics.yaml", metrics)
            checkpoint.unlink()
            update_status(status="COMPLETED")
            print(f"CTRL_RESULT {experiment} seed={run.seed} equal={summary['raw_output_equal_count']}/{len(records)} "
                  f"limit_hit={summary['limit_hit_count']} latency_mean_ms={summary['latency_ms']['mean']:.1f} "
                  f"gpu={metrics['environment']['gpu']}", flush=True)
            return metrics
        except BaseException as exc:
            update_status(status="FAILED", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Aggregate (CPU)
# ---------------------------------------------------------------------------

def full_test_latency(control: dict, output_limit_metrics: dict | None, test_size: int):
    """Mean latency over the whole test set: sample mean for untouched answers, measured times for re-generated ones."""
    if not output_limit_metrics:
        return None
    regeneration = output_limit_metrics.get("regeneration") or {}
    regenerated, total_seconds = regeneration.get("selected_count"), (regeneration.get("latency_ms") or {}).get("total_seconds")
    if type(regenerated) is not int or not mnol._is_number(total_seconds):
        return None
    return ((test_size - regenerated) * control["latency_ms"]["mean"] + total_seconds * 1000) / test_size


def aggregate(source_root, output_root, *, output_limit_root=None, max_new_tokens: int = 1024) -> dict:
    source_root, output_root = Path(source_root), Path(output_root)
    output_limit_root = Path(output_limit_root) if output_limit_root else source_root / f"multinerd_output_limit_{max_new_tokens}"
    rows, missing = [], []
    for run in mnol.selected_runs():
        try:
            source = source_run(run, source_root)
            experiment = source["config"]["experiment_name"]
            stored = mnol.load_source_predictions(source)
            metrics_path = output_root / experiment / "control_metrics.yaml"
            metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
            requested = metrics["contract"]["selection"]
            selection = control_indices(stored, requested["sample_size"], requested["sample_seed"])
            contract = control_contract(source, selection, max_new_tokens=max_new_tokens,
                                        sample_size=requested["sample_size"], sample_seed=requested["sample_seed"])
            if not completed_control(output_root / experiment, contract):
                raise ValueError("Missing final control metrics")
            records = json.loads((output_root / experiment / "control_predictions.json").read_text(encoding="utf-8"))
            if len(records) != len(selection):
                raise ValueError("Control predictions do not cover the selection")
            for index, record in zip(selection, records):
                validate_control_record(record, stored[index], index, max_new_tokens)
            control = control_summary(records, stored, selection, max_new_tokens,
                                      vram_peak_mb=metrics["control"]["vram_peak_mb"], total_params=metrics["control"]["total_params"])
            if not mnol._close(control, metrics["control"]):
                raise ValueError("Saved control summary differs from the verified records")
            limit_path = output_limit_root / experiment / "inference_metrics.yaml"
            limit_metrics = yaml.safe_load(limit_path.read_text(encoding="utf-8")) if limit_path.exists() else None
            stored_metrics = mnol.source_inference_metrics(source)
            n = control["sample_count"]
            rows.append({
                "model": run.group_key, "seed": run.seed, "experiment_name": experiment, "sample_count": n,
                "gpu": metrics["environment"]["gpu"], "source_gpu": metrics["environment"]["source_gpu"],
                "raw_output_equal": control["raw_output_equal_count"], "raw_output_equal_share": control["raw_output_equal_count"] / n,
                "pred_entities_equal": control["pred_entities_equal_count"],
                "subset_f1_stored_256": control["subset_metrics"]["stored_256"]["f1"],
                "subset_f1_control": control["subset_metrics"][f"control_{max_new_tokens}"]["f1"],
                "limit_hit_count": control["limit_hit_count"], "generated_tokens_mean": control["generated_tokens"]["mean"],
                "latency_ms_mean": control["latency_ms"]["mean"], "latency_ms_p50": control["latency_ms"]["p50"],
                "latency_ms_p95": control["latency_ms"]["p95"],
                "source_latency_ms_mean_256": stored_metrics.get("latency_ms_mean"),
                "full_test_latency_ms_estimate": full_test_latency(control, limit_metrics, len(stored)),
                "vram_peak_mb": control["vram_peak_mb"],
            })
        except INVALID_ARTIFACT_ERRORS as exc:
            missing.append({"model": run.group_key, "seed": run.seed, "reason": str(exc)})
    report = {"complete": not missing, "max_new_tokens": max_new_tokens, "selection_rule": SELECTION_RULE,
              "output_root": str(output_root.resolve()), "output_limit_root": str(output_limit_root.resolve()),
              "rows": rows, "missing_or_failed": missing}
    target = output_root / "aggregate"
    atomic_write_yaml(target / "control_summary.yaml", report)
    write_csv(target / "control_summary.csv", rows)
    lines = ["# MultiNERD control sample", "", f"Status: {'COMPLETE' if not missing else 'INCOMPLETE'}; {len(rows)}/9 runs.", "",
             "| Model | Seed | n | Equal answers | Subset F1 256 | Subset F1 ctrl | Limit hits | Gen. tokens | Latency ms mean "
             "| Full-test latency ms est. | Source latency ms (256) | GPU | VRAM MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        estimate = row["full_test_latency_ms_estimate"]
        lines.append(f"| {row['model']} | {row['seed']} | {row['sample_count']} | {row['raw_output_equal']} | "
                     f"{row['subset_f1_stored_256'] * 100:.2f} | {row['subset_f1_control'] * 100:.2f} | {row['limit_hit_count']} | "
                     f"{row['generated_tokens_mean']:.1f} | {row['latency_ms_mean']:.0f} | "
                     f"{'—' if estimate is None else f'{estimate:.0f}'} | {row['source_latency_ms_mean_256']:.0f} | "
                     f"{row['gpu']} | {row['vram_peak_mb']:.0f} |")
    if missing:
        lines += ["", "Missing or failed runs:", ""] + [f"- {m['model']}, seed {m['seed']}: {m['reason']}" for m in missing]
    atomic_write_bytes(target / "control_summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="canonical Qwen QLoRA config whose control sample is re-generated (GPU)")
    mode.add_argument("--aggregate", action="store_true", help="verify finished runs and write aggregate/control_summary.* (CPU)")
    parser.add_argument("--source-results-root", type=Path, required=True, help="results root holding multinerd/<experiment>")
    parser.add_argument("--output-root", type=Path, required=True, help="separate root for the control sample")
    parser.add_argument("--output-limit-root", type=Path, help="aggregate only: root of the output-limit evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="higher output limit (> 256; default 1024)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help=f"sampled sentences (default {SAMPLE_SIZE})")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED, help=f"permutation seed (default {SAMPLE_SEED})")
    parser.add_argument("--resume-failed", action="store_true", help="recover a stale lock of a dead job and resume the checkpoint")
    args = parser.parse_args(argv)
    try:
        _check_limit(args.max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.aggregate:
        report = aggregate(args.source_results_root, args.output_root, output_limit_root=args.output_limit_root,
                           max_new_tokens=args.max_new_tokens)
        print(f"CTRL_AGGREGATE {len(report['rows'])}/9 complete={report['complete']}", flush=True)
        return 0 if report["complete"] else 1
    try:
        run = mnol.run_for_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    run_control(run, args.source_results_root, args.output_root, max_new_tokens=args.max_new_tokens,
                sample_size=args.sample_size, sample_seed=args.sample_seed, resume_failed=args.resume_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
