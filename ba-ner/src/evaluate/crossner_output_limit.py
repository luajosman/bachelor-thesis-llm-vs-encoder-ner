"""Separate, inference-only CrossNER output-limit sensitivity evaluation.

The original evaluator and its scientific hashes remain unchanged. Baseline
artifacts are read and verified; only this experiment's output root is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import math
import os
from pathlib import Path
import statistics
import time

import yaml

from src.data.crossner_loader import CROSSNER_REVISION, DOMAINS, load_crossner_test, project_multinerd_bio
from src.data.dataset_loader import MULTINERD_INFO
from src.evaluate.crossner import (
    GROUPS, SEEDS, PROJECT_ROOT, FrozenPredictor, baseline_metrics, completed_domain,
    domain_metrics, ensure_safe_output, evaluation_contract, file_hash,
    read_checkpoint, run_lock, selected_runs, source_run, validate_record,
)
from src.evaluate.metrics import compute_ner_metrics
from src.seed_provenance import canonical_hash
from src.seed_study import atomic_write_bytes, atomic_write_json, atomic_write_yaml

QWEN_GROUP = "qwen35-27b-qlora-3ep"
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "configs" / "crossner_overlap_exclusions.yaml"
INVALID_ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError)


def _validated_memory(value) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Saved vram_peak_mb must be a finite nonnegative number or None")
    return float(value)


def _validate_saved_quality(metrics: dict) -> None:
    for field in ("precision", "recall", "f1", "delta_f1"):
        value = metrics.get(field)
        lower = -1 if field == "delta_f1" else 0
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= 1:
            raise ValueError(f"Saved {field} must be a finite number between {lower} and 1")


def _check_limit(max_new_tokens: int) -> None:
    if type(max_new_tokens) is not int or max_new_tokens <= 256:
        raise ValueError("The additional evaluation requires max_new_tokens > 256")


def output_limit_contract(source: dict, domain: str, smoke_limit: int | None = None,
                          *, max_new_tokens: int = 1024) -> dict:
    _check_limit(max_new_tokens)
    if source["identity"]["model_type"] != "decoder":
        raise ValueError("The output-limit evaluation supports Qwen decoder checkpoints only")
    contract = evaluation_contract(source, domain, smoke_limit)
    contract["version"] = 2
    contract["evaluation"] = f"output_limit_{max_new_tokens}"
    contract["decoder"]["max_new_tokens"] = max_new_tokens
    contract["generation_diagnostics"] = {
        "generated_token_count": "actual generated IDs, including terminal EOS",
        "stop_reason": "inferred from terminal EOS, otherwise limit, otherwise other",
        "json_complete": "raw output parses as one JSON array",
    }
    contract["code_sha256"]["src/evaluate/crossner_output_limit.py"] = file_hash(Path(__file__))
    return contract


def ensure_output_limit_safe(output_root: Path, source_root: Path, baseline_root: Path) -> None:
    ensure_safe_output(output_root, source_root)
    output = Path(output_root).resolve()
    for baseline in (Path(baseline_root).resolve(), (Path(source_root) / "crossner").resolve()):
        if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
            raise ValueError("Output-limit output must not overlap the original CrossNER baseline")


def load_exclusions(exclusions_path: Path = DEFAULT_EXCLUSIONS, cache_dir: Path | None = None) -> dict:
    path = Path(exclusions_path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("dataset") != "crossner" or config.get("revision") != CROSSNER_REVISION:
        raise ValueError("Exclusions do not identify the pinned CrossNER dataset")
    ids = config.get("excluded_sample_ids")
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Exclusions must contain unique sample IDs")
    if config.get("expected_excluded_samples") != len(ids):
        raise ValueError("Exclusion count differs from its fixed declaration")
    if cache_dir is not None:
        valid_ids = {r["sample_id"] for d in DOMAINS for r in load_crossner_test(d, cache_dir)}
        if set(ids) - valid_ids:
            raise ValueError("Exclusions contain unknown CrossNER sample IDs")
    return {**config, "sha256": file_hash(path), "excluded_sample_ids": sorted(ids)}


def _read_reference(source: dict, baseline_root: Path, domain: str, samples: list[dict]) -> dict:
    directory = Path(baseline_root) / source["config"]["experiment_name"] / domain
    contract = evaluation_contract(source, domain)
    if not completed_domain(directory, contract, samples):
        raise ValueError(f"{directory}: missing final reference metrics")
    metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
    if not metrics.get("complete") or metrics["sample_count"] != len(samples):
        raise ValueError(f"{directory}: smoke or incomplete reference")
    _validated_memory(metrics.get("vram_peak_mb"))
    _validate_saved_quality(metrics)
    records = json.loads((directory / "test_predictions.json").read_text())
    quality = compute_ner_metrics([r["gold_bio"] for r in records], [r["pred_bio"] for r in records])
    if any(abs(quality[k] - metrics[k]) > 1e-12 for k in quality):
        raise ValueError(f"{directory}: reference metrics differ from predictions")
    return {"records": records, "metrics": metrics, "source": source["identity"],
            "artifact": {"path": str((directory / "test_predictions.json").resolve()),
                         "predictions_path": str((directory / "test_predictions.json").resolve()),
                         "predictions_sha256": metrics["predictions_sha256"], "contract_hash": metrics["contract_hash"]}}


def validate_references(source_root: Path, baseline_root: Path, cache_dir: Path,
                        seeds=SEEDS, domains=DOMAINS) -> dict:
    datasets = {d: load_crossner_test(d, cache_dir) for d in domains}
    references = {}
    for run in selected_runs(GROUPS, seeds):
        source = source_run(run, source_root)
        evaluation = "encoder_baseline" if run.model_family == "encoder" else "baseline_256"
        for domain, samples in datasets.items():
            references.setdefault((run.seed, domain), {})[evaluation] = _read_reference(source, baseline_root, domain, samples)
    return references


def generation_diagnostics(new_token_ids: list[int], raw: str, eos_token_id: int | list[int] | None,
                           max_new_tokens: int) -> dict:
    count = len(new_token_ids)
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    eos = bool(new_token_ids) and new_token_ids[-1] in eos_ids
    try:
        complete = isinstance(json.loads(raw), list)
    except (ValueError, TypeError):
        complete = False
    return {
        "generated_token_count": count,
        "token_limit_reached": count == max_new_tokens,
        "stop_reason": "eos" if eos else "max_new_tokens" if count == max_new_tokens else "other",
        "json_complete": complete,
    }


class OutputLimitPredictor(FrozenPredictor):
    def __init__(self, source: dict, *, max_new_tokens: int = 1024):
        _check_limit(max_new_tokens)
        if source["identity"]["model_type"] != "decoder":
            raise ValueError("The output-limit evaluation supports Qwen decoder checkpoints only")
        super().__init__(source)
        self.max_new_tokens = max_new_tokens

    def predict(self, sample: dict) -> dict:
        from src.data.preprocess_decoder import build_system_prompt, format_numbered_tokens
        from src.decoder.generation import prepare_generation_inputs
        from src.decoder.parse_output import entities_to_bio, parse_llm_output_with_diagnostics

        torch, tokenizer, tokens = self.torch, self.tokenizer, sample["tokens"]
        messages = [{"role": "system", "content": build_system_prompt(MULTINERD_INFO.entity_types)},
                    {"role": "user", "content": format_numbered_tokens(tokens)}]
        inputs = prepare_generation_inputs(tokenizer, messages, self.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            result = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         temperature=None, top_p=None, eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        new_ids = result[0, inputs["input_ids"].shape[1]:].tolist()
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        entities, status, diagnostics = parse_llm_output_with_diagnostics(raw, tokens, frozenset(MULTINERD_INFO.entity_types))
        raw_bio = entities_to_bio(tokens, entities)
        return {
            **sample, "raw_pred_bio": raw_bio, "pred_bio": project_multinerd_bio(raw_bio),
            "latency_ms": latency, "raw_output": raw, "pred_entities": entities,
            "parse_status": status, "parse_diagnostics": diagnostics,
            **generation_diagnostics(new_ids, raw, tokenizer.eos_token_id, self.max_new_tokens),
        }


def validate_output_limit_record(record: dict, sample: dict, max_new_tokens: int) -> None:
    validate_record(record, sample)
    count = record.get("generated_token_count")
    if type(count) is not int or not 0 < count <= max_new_tokens:
        raise ValueError("Output-limit checkpoint has an invalid actual generated token count")
    if type(record.get("token_limit_reached")) is not bool or record["token_limit_reached"] != (count == max_new_tokens):
        raise ValueError("Output-limit checkpoint has inconsistent token-limit diagnostics")
    reason = record.get("stop_reason")
    if reason not in ("eos", "max_new_tokens", "other") or (reason == "max_new_tokens" and count != max_new_tokens):
        raise ValueError("Output-limit checkpoint has inconsistent stop diagnostics")
    if reason == "other" and count == max_new_tokens:
        raise ValueError("Output-limit checkpoint has inconsistent stop diagnostics")
    raw = record.get("raw_output")
    if not isinstance(raw, str):
        raise ValueError("Output-limit checkpoint is missing its raw output")
    expected_json = generation_diagnostics([0], raw, None, max_new_tokens)["json_complete"]
    if type(record.get("json_complete")) is not bool or record["json_complete"] != expected_json:
        raise ValueError("Output-limit checkpoint has inconsistent JSON diagnostics")
    if record.get("parse_status") not in ("ok", "markdown_stripped", "regex_fallback", "failed"):
        raise ValueError("Output-limit checkpoint has an invalid parse status")
    if record["parse_status"] == "failed" and any(tag != "O" for tag in record["raw_pred_bio"]):
        raise ValueError("Failed decoder parses must remain empty predictions")


def completed_output_limit_domain(directory: Path, contract: dict, samples: list[dict]) -> bool:
    if not completed_domain(directory, contract, samples):
        return False
    metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
    _validated_memory(metrics.get("vram_peak_mb"))
    _validate_saved_quality(metrics)
    records = json.loads((directory / "test_predictions.json").read_text())
    for record, sample in zip(records, samples):
        validate_output_limit_record(record, sample, contract["decoder"]["max_new_tokens"])
    return True


def _record_metrics(records: list[dict], evaluation: str, seed: int, max_new_tokens: int,
                    vram_peak_mb: float | None = None) -> dict:
    quality = compute_ner_metrics([r["gold_bio"] for r in records], [r["pred_bio"] for r in records]) if records else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    decoder = evaluation != "encoder_baseline"
    extended = evaluation.startswith("output_limit_")
    return {
        "model": QWEN_GROUP if decoder else GROUPS[0], "seed": seed,
        "max_new_tokens": max_new_tokens if extended else 256 if decoder else None,
        "sample_count": len(records), **quality,
        "parse_failure_rate": sum(r.get("parse_status") == "failed" for r in records) / len(records) if decoder and records else None,
        "output_limit_rate": sum(r["token_limit_reached"] for r in records) / len(records) if extended and records else None,
        "latency_ms": statistics.fmean(r["latency_ms"] for r in records) if records else None,
        "generated_tokens": statistics.fmean(r["generated_token_count"] for r in records) if extended and records else None,
        "json_complete_rate": sum(r["json_complete"] for r in records) / len(records) if extended and records else None,
        "eos_stop_rate": sum(r["stop_reason"] == "eos" for r in records) / len(records) if extended and records else None,
        "length_stop_rate": sum(r["stop_reason"] == "max_new_tokens" for r in records) / len(records) if extended and records else None,
        "other_stop_rate": sum(r["stop_reason"] == "other" for r in records) / len(records) if extended and records else None,
        "vram_peak_mb": _validated_memory(vram_peak_mb),
    }


def reference_metrics(records: list[dict], references: dict, excluded_ids: set[str], seed: int,
                      max_new_tokens: int, *, vram_peak_mb: float | None = None) -> dict:
    reference_by_id = {key: {r["sample_id"]: r for r in value["records"]} for key, value in references.items()}
    aligned = {"encoder_baseline": [], "baseline_256": [], f"output_limit_{max_new_tokens}": records}
    for record in records:
        for evaluation, by_id in reference_by_id.items():
            old = by_id.get(record["sample_id"])
            if old is None or any(old[k] != record[k] for k in ("tokens", "gold_bio", "native_bio")):
                raise ValueError("Paired CrossNER references have different sample IDs, tokens or gold")
            aligned[evaluation].append(old)
    result = {}
    for population in ("full", "deduplicated"):
        rows = {evaluation: rows if population == "full" else [r for r in rows if r["sample_id"] not in excluded_ids]
                for evaluation, rows in aligned.items()}
        metrics = {evaluation: _record_metrics(
            rows, evaluation, seed, max_new_tokens,
            references.get(evaluation, {}).get("metrics", {}).get("vram_peak_mb") if evaluation in references else vram_peak_mb,
        ) for evaluation, rows in rows.items()}
        treatment = metrics[f"output_limit_{max_new_tokens}"]
        treatment["delta_f1_vs_256"] = treatment["f1"] - metrics["baseline_256"]["f1"]
        treatment["delta_recall_vs_256"] = treatment["recall"] - metrics["baseline_256"]["recall"]
        result[population] = metrics
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_evaluation(run, source_root: Path, output_root: Path, cache_dir: Path, domains=DOMAINS,
                   *, max_new_tokens: int = 1024, baseline_root: Path | None = None,
                   exclusions_path: Path = DEFAULT_EXCLUSIONS, smoke_limit: int | None = None,
                   resume_failed: bool = False) -> None:
    from src.run_metadata import collect_run_metadata

    _check_limit(max_new_tokens)
    if run.group_key != QWEN_GROUP:
        raise ValueError("Only Qwen checkpoints can receive a higher output limit")
    if smoke_limit is not None and (type(smoke_limit) is not int or smoke_limit < 1):
        raise ValueError("smoke_limit must be positive")
    baseline_root = Path(baseline_root or Path(source_root) / "crossner")
    ensure_output_limit_safe(output_root, source_root, baseline_root)
    all_samples = {d: load_crossner_test(d, cache_dir) for d in domains}
    if smoke_limit is not None:
        all_samples = {d: rows[:smoke_limit] for d, rows in all_samples.items()}
    directory = Path(output_root) / load_experiment_name(run)
    predictor, measured_records, verified_records = None, [], []
    started = time.monotonic()
    status = {
        "status": "RUNNING", "model": QWEN_GROUP, "seed": run.seed, "max_new_tokens": max_new_tokens,
        "smoke_limit": smoke_limit, "job_id": os.environ.get("SLURM_JOB_ID"), "domains": list(domains),
        "started_at": _utc_now(), "completed_domains": [], "completed_samples": 0,
        "total_samples": sum(len(rows) for rows in all_samples.values()),
        "current_domain": None, "current_domain_completed": 0, "current_domain_total": 0,
    }

    def update_status(**changes):
        status.update(changes)
        elapsed = time.monotonic() - started
        measured = len(measured_records)
        remaining = status["total_samples"] - status["completed_samples"]
        status.update(updated_at=_utc_now(), elapsed_seconds=elapsed,
                      eta_seconds=elapsed / measured * remaining if measured and remaining else 0.0 if remaining == 0 else None,
                      parse_failure_count=sum(r.get("parse_status") == "failed" for r in verified_records),
                      output_limit_count=sum(r["token_limit_reached"] for r in verified_records))
        atomic_write_json(directory / "status.json", status)

    with run_lock(directory, resume_failed=resume_failed):
        update_status()
        try:
            source = source_run(run, source_root)
            exclusions = load_exclusions(exclusions_path, cache_dir)
            references = validate_references(source_root, baseline_root, cache_dir, (run.seed,), domains)
            baseline = baseline_metrics(source)
            atomic_write_yaml(directory / "multinerd_baseline.yaml", baseline)
            for domain, samples in all_samples.items():
                target = directory / domain
                target.mkdir(parents=True, exist_ok=True)
                contract = output_limit_contract(source, domain, smoke_limit, max_new_tokens=max_new_tokens)
                update_status(current_domain=domain, current_domain_completed=0, current_domain_total=len(samples))
                if completed_output_limit_domain(target, contract, samples):
                    verified_records.extend(json.loads((target / "test_predictions.json").read_text()))
                    status["completed_domains"].append(domain)
                    update_status(completed_samples=status["completed_samples"] + len(samples), current_domain_completed=len(samples))
                    print(f"SKIP_COMPLETED {QWEN_GROUP} seed={run.seed} domain={domain} max_new_tokens={max_new_tokens}", flush=True)
                    continue
                checkpoint = target / "inference_checkpoint.jsonl"
                records = read_checkpoint(checkpoint, contract, samples)
                for record, sample in zip(records, samples):
                    validate_output_limit_record(record, sample, max_new_tokens)
                verified_records.extend(records)
                atomic_write_json(target / "contract.json", contract)
                temporary = checkpoint.with_suffix(".tmp")
                with temporary.open("w") as handle:
                    handle.write(json.dumps({"contract": contract, "contract_hash": canonical_hash(contract)}) + "\n")
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, checkpoint)
                before_domain = status["completed_samples"]
                update_status(completed_samples=before_domain + len(records), current_domain_completed=len(records))
                if len(records) < len(samples) and predictor is None:
                    predictor = OutputLimitPredictor(source, max_new_tokens=max_new_tokens)
                    predictor.predict(samples[0])
                with checkpoint.open("a") as handle:
                    for sample in samples[len(records):]:
                        record = predictor.predict(sample)
                        validate_output_limit_record(record, sample, max_new_tokens)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(record)
                        measured_records.append(record)
                        verified_records.append(record)
                        status.update(completed_samples=before_domain + len(records), current_domain_completed=len(records))
                        if len(records) % 25 == 0 or len(records) == len(samples):
                            update_status(completed_samples=before_domain + len(records), current_domain_completed=len(records))
                            print(f"CROSSNER_PROGRESS {QWEN_GROUP} seed={run.seed} domain={domain} {len(records)}/{len(samples)} max_new_tokens={max_new_tokens}", flush=True)
                predictions = target / "test_predictions.json"
                atomic_write_json(predictions, records)
                metrics = domain_metrics(records, baseline, contract)
                vram = predictor.torch.cuda.max_memory_allocated() / 1024**2 if predictor is not None else None
                metrics.update(
                    max_new_tokens=max_new_tokens, predictions_sha256=file_hash(predictions),
                    run_metadata=collect_run_metadata(source["config"]),
                    generated_tokens_mean=statistics.fmean(r["generated_token_count"] for r in records),
                    output_limit_count=sum(r["token_limit_reached"] for r in records),
                    output_limit_rate=sum(r["token_limit_reached"] for r in records) / len(records),
                    stop_reason_counts={reason: sum(r["stop_reason"] == reason for r in records) for reason in ("eos", "max_new_tokens", "other")},
                    json_complete_rate=sum(r["json_complete"] for r in records) / len(records),
                    exclusions=exclusions,
                    reference_artifacts={evaluation: reference["artifact"] for evaluation, reference in references[(run.seed, domain)].items()},
                    reference_metrics=reference_metrics(records, references[(run.seed, domain)], set(exclusions["excluded_sample_ids"]), run.seed, max_new_tokens, vram_peak_mb=vram),
                )
                metrics["vram_peak_mb"] = vram
                atomic_write_yaml(target / "inference_metrics.yaml", metrics)
                checkpoint.unlink()
                status["completed_domains"].append(domain)
                update_status()
                print(f"CROSSNER_RESULT {domain} seed={run.seed} f1={metrics['f1']:.6f} max_new_tokens={max_new_tokens}", flush=True)
            update_status(status="COMPLETED")
        except BaseException as exc:
            update_status(status="FAILED", error=str(exc))
            raise


def load_experiment_name(run) -> str:
    from src.config import load_experiment_config
    return load_experiment_config(PROJECT_ROOT / run.config_path)["experiment_name"]


def _comparison_summary(seed_rows: list[dict], seeds, domains) -> list[dict]:
    summaries = []
    fields = ("precision", "recall", "f1", "parse_failure_rate", "output_limit_rate",
              "latency_ms", "generated_tokens", "json_complete_rate", "eos_stop_rate",
              "length_stop_rate", "other_stop_rate", "vram_peak_mb", "delta_f1_vs_256", "delta_recall_vs_256")
    evaluations = tuple(dict.fromkeys(r["evaluation"] for r in seed_rows))
    for population in ("full", "deduplicated"):
        for evaluation in evaluations:
            for domain in (*domains, "pooled_micro", "domain_mean"):
                rows = [r for r in seed_rows if r["population"] == population and r["evaluation"] == evaluation and r["domain"] == domain]
                summary = {
                    "model": QWEN_GROUP if evaluation != "encoder_baseline" else GROUPS[0],
                    "evaluation": evaluation, "population": population, "domain": domain,
                    "successful_seeds": len(rows), "expected_seeds": len(seeds), "complete": len(rows) == len(seeds),
                    "sample_count": rows[0]["sample_count"] if rows else None,
                }
                for field in fields:
                    values = [r[field] for r in rows if r.get(field) is not None]
                    summary[field + "_mean"] = statistics.fmean(values) if values else None
                    summary[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
                summaries.append(summary)
    return summaries


def aggregate_output_limit(source_root: Path, output_root: Path, cache_dir: Path, models=(QWEN_GROUP,),
                           seeds=SEEDS, domains=DOMAINS, *, max_new_tokens: int = 1024,
                           baseline_root: Path | None = None, exclusions_path: Path = DEFAULT_EXCLUSIONS) -> dict:
    from src.evaluate.crossner_aggregate import aggregate, write_csv

    _check_limit(max_new_tokens)
    if tuple(models) != (QWEN_GROUP,):
        raise ValueError("Only Qwen checkpoints can receive a higher output limit")
    baseline_root = Path(baseline_root or Path(source_root) / "crossner")
    ensure_output_limit_safe(output_root, source_root, baseline_root)
    exclusions = load_exclusions(exclusions_path, cache_dir)
    datasets = {d: load_crossner_test(d, cache_dir) for d in domains}
    try:
        primary = aggregate(source_root, output_root, cache_dir, models, seeds, domains,
                            contract_factory=partial(output_limit_contract, max_new_tokens=max_new_tokens),
                            primary_models=(QWEN_GROUP,))
    except INVALID_ARTIFACT_ERRORS as exc:
        primary = {"complete": False, "selection_complete": False,
                   "missing_or_failed": [{"model": QWEN_GROUP, "seed": seed, "domain": domain, "reason": str(exc)}
                                         for seed in seeds for domain in domains]}
        target = Path(output_root) / "aggregate"
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(target / "summary.yaml", primary)
    rows, missing = [], [dict(item) for item in primary["missing_or_failed"]]
    rejected_domains = {(item["seed"], item["domain"]) for item in missing}
    success = 0
    for run in selected_runs(models, seeds):
        try:
            source = source_run(run, source_root)
        except INVALID_ARTIFACT_ERRORS as exc:
            missing.extend({"model": QWEN_GROUP, "seed": run.seed, "domain": d, "reason": str(exc)}
                           for d in domains if (run.seed, d) not in rejected_domains)
            continue
        pooled = {evaluation: [] for evaluation in ("encoder_baseline", "baseline_256", f"output_limit_{max_new_tokens}")}
        memory = {evaluation: [] for evaluation in pooled}
        domain_rows = []
        for domain, samples in datasets.items():
            if (run.seed, domain) in rejected_domains:
                continue
            try:
                references = validate_references(source_root, baseline_root, cache_dir, (run.seed,), (domain,))[(run.seed, domain)]
                directory = Path(output_root) / source["config"]["experiment_name"] / domain
                contract = output_limit_contract(source, domain, max_new_tokens=max_new_tokens)
                if not completed_output_limit_domain(directory, contract, samples):
                    raise ValueError("Missing final output-limit metrics")
                saved_metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
                if not saved_metrics.get("complete"):
                    raise ValueError("Smoke or incomplete evaluation is excluded")
                records = json.loads((directory / "test_predictions.json").read_text())
                paired = reference_metrics(records, references, set(exclusions["excluded_sample_ids"]), run.seed, max_new_tokens, vram_peak_mb=saved_metrics.get("vram_peak_mb"))
                for population, evaluations in paired.items():
                    for evaluation, metrics in evaluations.items():
                        row = {"evaluation": evaluation, "population": population, "domain": domain, **metrics}
                        domain_rows.append(row)
                        rows.append(row)
                for evaluation in ("encoder_baseline", "baseline_256"):
                    pooled[evaluation].extend(references[evaluation]["records"])
                    if references[evaluation]["metrics"].get("vram_peak_mb") is not None:
                        memory[evaluation].append(references[evaluation]["metrics"]["vram_peak_mb"])
                pooled[f"output_limit_{max_new_tokens}"].extend(records)
                if saved_metrics.get("vram_peak_mb") is not None:
                    memory[f"output_limit_{max_new_tokens}"].append(saved_metrics["vram_peak_mb"])
                success += 1
            except INVALID_ARTIFACT_ERRORS as exc:
                missing.append({"model": QWEN_GROUP, "seed": run.seed, "domain": domain, "reason": str(exc)})
        if len(domain_rows) != len(domains) * 6:
            continue
        pooled_refs = {evaluation: {"records": pooled[evaluation], "metrics": {"vram_peak_mb": max(memory[evaluation]) if memory[evaluation] else None}} for evaluation in ("encoder_baseline", "baseline_256")}
        paired = reference_metrics(pooled[f"output_limit_{max_new_tokens}"], pooled_refs,
                                   set(exclusions["excluded_sample_ids"]), run.seed, max_new_tokens,
                                   vram_peak_mb=max(memory[f"output_limit_{max_new_tokens}"]) if memory[f"output_limit_{max_new_tokens}"] else None)
        for population, evaluations in paired.items():
            for evaluation, metrics in evaluations.items():
                rows.append({"evaluation": evaluation, "population": population, "domain": "pooled_micro", **metrics})
                domain_mean = dict(metrics)
                selected = [r for r in domain_rows if r["population"] == population and r["evaluation"] == evaluation]
                for key in ("precision", "recall", "f1", "parse_failure_rate", "output_limit_rate",
                            "latency_ms", "generated_tokens", "json_complete_rate", "eos_stop_rate",
                            "length_stop_rate", "other_stop_rate", "delta_f1_vs_256", "delta_recall_vs_256"):
                    if key in metrics:
                        values = [r.get(key) for r in selected]
                        domain_mean[key] = statistics.fmean(values) if all(v is not None for v in values) else None
                rows.append({"evaluation": evaluation, "population": population, "domain": "domain_mean", **domain_mean})
    summary = _comparison_summary(rows, seeds, domains)
    complete = primary["complete"] and not missing
    report = {
        "complete": complete, "selection_complete": primary["selection_complete"] and not missing,
        "evaluation": f"output_limit_{max_new_tokens}", "max_new_tokens": max_new_tokens,
        "successful_evaluations": success, "expected_evaluations": len(seeds) * len(domains),
        "primary_expected_evaluations": len(SEEDS) * len(DOMAINS), "ddof": 1,
        "selection": {"models": list(models), "seeds": list(seeds), "domains": list(domains)},
        "baseline_output_root": str(baseline_root.resolve()), "exclusions": exclusions,
        "source_results_root": str(Path(source_root).resolve()), "output_root": str(Path(output_root).resolve()),
        "interpretation": "Post-hoc output-limit sensitivity analysis; fixed checkpoints, prompts, parser and PER/ORG/LOC scoring",
        "summary": summary, "seed_metrics": rows, "missing_or_failed": missing,
    }
    target = Path(output_root) / "aggregate"
    atomic_write_yaml(target / "comparison.yaml", report)
    write_csv(target / "comparison.csv", summary)
    lines = ["# CrossNER output-limit sensitivity", "",
             f"Status: {'COMPLETE' if complete else 'INCOMPLETE / SUBSET'}; {success}/{len(seeds) * len(domains)} paired evaluations.",
             f"Qwen output limit: {max_new_tokens}; reference: 256. No new training or DeBERTa inference.", "",
             "Values are percentages (mean ± sample SD across training seeds); ΔF1 and ΔRecall compare Qwen with its 256-token reference in percentage points.",
             "Historical output-token counts and stop causes were not recorded; their limit rates remain unavailable.", "",
             "| Population | Evaluation | Domain | Seeds | Precision | Recall | F1 | ΔF1 vs 256 | Parse errors | Limit reached | JSON complete | Length stop | Latency ms | VRAM MiB |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def display(row, field, scale=100):
        value, deviation = row[field + "_mean"], row[field + "_std"]
        return "—" if value is None else f"{value * scale:.2f}" + (f" ± {deviation * scale:.2f}" if deviation is not None else " (SD n/a)")
    for row in summary:
        values = [display(row, field) for field in ("precision", "recall", "f1", "delta_f1_vs_256", "parse_failure_rate", "output_limit_rate", "json_complete_rate", "length_stop_rate")]
        lines.append(f"| {row['population']} | {row['evaluation']} | {row['domain']} | {row['successful_seeds']}/{row['expected_seeds']} | " + " | ".join(values) + f" | {display(row, 'latency_ms', 1)} | {display(row, 'vram_peak_mb', 1)} |")
    lines += ["", f"The deduplicated population excludes the fixed {len(exclusions['excluded_sample_ids'])} source-training overlap IDs from every arm.",
              "Failed parses remain included as empty predictions. pooled_micro pools domains within each seed; domain_mean weights domains equally.", ""]
    if missing:
        lines.extend(f"- Seed {m['seed']}, {m['domain']}: {m['reason']}" for m in missing)
    atomic_write_bytes(target / "comparison.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-root", type=Path)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--models", nargs="+", choices=(QWEN_GROUP,), default=[QWEN_GROUP])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--resume-failed", action="store_true")
    args = parser.parse_args(argv)
    try:
        _check_limit(args.max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke_limit is not None and args.smoke_limit < 1:
        parser.error("--smoke-limit must be positive")
    if args.aggregate and (args.smoke_limit is not None or args.resume_failed):
        parser.error("Aggregate mode does not accept smoke or resume flags")
    domains = tuple(dict.fromkeys(args.domains))
    shared = {"max_new_tokens": args.max_new_tokens, "baseline_root": args.baseline_output_root, "exclusions_path": args.exclusions}
    if args.aggregate:
        report = aggregate_output_limit(args.source_results_root, args.output_root, args.cache_dir,
                                        tuple(dict.fromkeys(args.models)), tuple(dict.fromkeys(args.seeds)), domains, **shared)
        print(f"CROSSNER_OUTPUT_LIMIT_AGGREGATE {report['successful_evaluations']}/{report['expected_evaluations']} complete={report['complete']}")
        return 0 if report["selection_complete"] else 1
    runs = [r for r in selected_runs((QWEN_GROUP,)) if (PROJECT_ROOT / r.config_path).resolve() == Path(args.config).resolve()]
    if len(runs) != 1:
        parser.error("--config must identify a canonical Qwen checkpoint")
    run_evaluation(runs[0], args.source_results_root, args.output_root, args.cache_dir, domains,
                   smoke_limit=args.smoke_limit, resume_failed=args.resume_failed, **shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
