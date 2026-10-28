"""Inference-only MultiNERD output-limit re-generation of stored parse failures.

The canonical Qwen3.5 QLoRA test predictions were generated with
``max_new_tokens=256``.  Every stored parse failure in those runs is a JSON
array that was cut off at that limit.  This module re-generates exactly those
sentences with the same frozen adapter, pinned base-model revision, prompt,
parser and greedy settings at a higher limit (1,024 tokens by default), keeps
every other stored answer untouched, and scores the merged 16,454-sentence
test set again.

Modes: ``--config`` (one GPU job per canonical run), ``--preflight`` (CPU
checks of sources, selections, locks and cached base models) and
``--aggregate`` (CPU; verified seed means and sample standard deviations).

The module is purely additive: the canonical artifacts, the hashed scientific
code and the CrossNER evaluators remain unchanged.  Source runs are read-only;
only the separate output root is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time

import yaml

from src.config import load_experiment_config
from src.data.dataset_loader import MULTINERD_ENGLISH_SPLIT_SIZES, MULTINERD_INFO
from src.data.preprocess_decoder import build_system_prompt, extract_entities_from_bio, format_numbered_tokens
from src.decoder.generation import prepare_generation_inputs
from src.decoder.parse_output import entities_to_bio, evaluate_llm_predictions, parse_llm_output_with_diagnostics
from src.evaluate.crossner import PROJECT_ROOT, SEEDS, ensure_safe_output, file_hash, source_run
from src.evaluate.crossner_aggregate import write_csv
from src.evaluate.crossner_output_limit import (
    INVALID_ARTIFACT_ERRORS, OutputLimitPredictor, _check_limit, generation_diagnostics,
)
from src.evaluate.metrics import compute_ner_metrics, compute_per_entity_metrics
from src.seed_provenance import DATASET_REVISION, canonical_hash
from src.seed_study import atomic_write_bytes, atomic_write_json, atomic_write_yaml, iter_run_descriptors

GROUPS = ("qwen35-27b-qlora-3ep", "qwen35-4b-qlora", "qwen35-08b-qlora")
BASELINE_LIMIT = 256
EXPECTED_SAMPLES = MULTINERD_ENGLISH_SPLIT_SIZES["test"]
ENTITY_TYPES = tuple(MULTINERD_INFO.entity_types)
RECORD_KEYS = frozenset({"gold_bio", "gold_entities", "parse_diagnostics", "parse_status", "pred_bio",
                         "pred_entities", "raw_output", "tokens"})
REGENERATED_KEYS = ("raw_output", "parse_status", "parse_diagnostics", "pred_entities", "pred_bio")
GENERATION_KEYS = ("latency_ms", "stored_prefix_match", "generated_token_count", "token_limit_reached",
                   "stop_reason", "json_complete")
REGENERATED_RECORD_KEYS = frozenset(RECORD_KEYS | set(GENERATION_KEYS) | {"regenerated", "index", "baseline_256"})
PARSE_STATUSES = ("ok", "markdown_stripped", "regex_fallback", "failed")
STOP_REASONS = ("eos", "max_new_tokens", "other")
STORED_METRIC_KEYS = ("precision", "recall", "f1", "parse_failure_rate", "parse_ok", "parse_markdown_stripped",
                      "parse_regex_fallback", "parse_failed", "parse_invalid_items", "parse_missing_fields",
                      "parse_unknown_types", "parse_wrong_schema", "parse_invalid_offsets", "parse_text_mismatches",
                      "parse_overlaps")
SELECTION_RULE = "parse_status == failed"
LOCK_NAME = ".multinerd_output_limit.lock"
CHECKPOINT_NAME = "inference_checkpoint.jsonl"
JOB_SCRIPT = "scripts/cluster/job_multinerd_output_limit.sh"
CODE_PATHS = ("src/data/dataset_loader.py", "src/data/preprocess_decoder.py", "src/decoder/generation.py",
              "src/decoder/parse_output.py", "src/evaluate/metrics.py", "src/evaluate/crossner.py",
              "src/evaluate/crossner_output_limit.py")
LATENCY_MEASUREMENT = "per re-generated sentence, batch size 1, cuda-synchronized, warm-up excluded"
SUMMARY_FIELDS = ("precision", "recall", "f1", "macro_f1", "parse_failed", "parse_failure_rate", "regenerated_count",
                  "still_failed_count", "limit_hit_count", "stored_prefix_mismatch_count", "generated_tokens_mean",
                  "regenerated_latency_ms_mean", "vram_peak_mb", "delta_f1_vs_256", "delta_recall_vs_256",
                  "delta_precision_vs_256", "delta_macro_f1_vs_256")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _close(reference, candidate, tolerance: float = 1e-12) -> bool:
    """Structural equality with an absolute tolerance for floating-point leaves."""
    if isinstance(reference, dict):
        return (isinstance(candidate, dict) and set(reference) == set(candidate)
                and all(_close(reference[key], candidate[key], tolerance) for key in reference))
    if isinstance(reference, list):
        return (isinstance(candidate, list) and len(reference) == len(candidate)
                and all(_close(a, b, tolerance) for a, b in zip(reference, candidate)))
    if type(reference) in (int, float) and type(candidate) in (int, float):
        return math.isfinite(reference) and math.isfinite(candidate) and abs(reference - candidate) <= tolerance
    return reference == candidate


def _validated_memory(value):
    if value is None:
        return None
    if not _is_number(value) or value < 0:
        raise ValueError("Saved vram_peak_mb must be a finite nonnegative number or None")
    return float(value)


def _check_smoke_limit(smoke_limit) -> None:
    if smoke_limit is not None and (type(smoke_limit) is not int or smoke_limit < 1):
        raise ValueError("smoke_limit must be a positive integer")


def _load_status(directory: Path) -> dict:
    path = Path(directory) / "status.json"
    if not path.exists():
        return {}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return status if isinstance(status, dict) else {}


# ---------------------------------------------------------------------------
# Run selection and output safety
# ---------------------------------------------------------------------------

def selected_runs(models=GROUPS, seeds=SEEDS) -> list:
    """The nine canonical Qwen QLoRA runs (0.8B, 4B, 27B x seeds 42/123/456); no encoders, no historical runs."""
    if not models or not seeds or set(models) - set(GROUPS) or set(seeds) - set(SEEDS):
        raise ValueError("Select supported Qwen QLoRA groups and seeds 42, 123, 456")
    return [run for run in iter_run_descriptors(include_historical=False)
            if run.group_key in models and run.seed in seeds and run.model_family == "decoder"]


def run_for_config(config_path):
    runs = [r for r in selected_runs() if (PROJECT_ROOT / r.config_path).resolve() == Path(config_path).resolve()]
    if len(runs) != 1:
        raise ValueError("--config must identify one of the nine canonical Qwen QLoRA runs")
    return runs[0]


def ensure_output_safe(output_root, source_root, *, smoke_limit=None) -> None:
    output, source = Path(output_root).resolve(), Path(source_root).resolve()
    if output == source:
        raise ValueError("Output root must not be the source results root")
    if output.is_relative_to(source) and output.relative_to(source).parts[0].startswith("crossner"):
        raise ValueError("Output root must not overlap the CrossNER evaluations")
    if smoke_limit is not None and output.parent == source and re.fullmatch(r"multinerd_output_limit_\d+", output.name):
        raise ValueError("Smoke output must have a separate path")
    ensure_safe_output(Path(output_root), Path(source_root))  # protects <source_root>/multinerd


# ---------------------------------------------------------------------------
# Stored source predictions and the pinned dataset
# ---------------------------------------------------------------------------

def validate_stored_record(record, index: int) -> None:
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise ValueError(f"stored prediction {index}: unexpected record keys")
    tokens = record["tokens"]
    if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"stored prediction {index}: tokens must be a list of strings")
    if (record["parse_status"] not in PARSE_STATUSES or not isinstance(record["raw_output"], str)
            or not isinstance(record["parse_diagnostics"], dict)
            or not isinstance(record["gold_entities"], list) or not isinstance(record["pred_entities"], list)):
        raise ValueError(f"stored prediction {index}: invalid parse fields")
    if entities_to_bio(tokens, record["gold_entities"]) != record["gold_bio"]:
        raise ValueError(f"stored prediction {index}: gold entities and gold BIO disagree")
    if entities_to_bio(tokens, record["pred_entities"]) != record["pred_bio"]:
        raise ValueError(f"stored prediction {index}: predicted entities and predicted BIO disagree")
    if record["parse_status"] == "failed" and record["pred_entities"]:
        raise ValueError(f"stored prediction {index}: failed parses must be empty predictions")


def load_source_predictions(source: dict) -> list:
    path = Path(source["predictions"])
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"{path}: expected {EXPECTED_SAMPLES} stored MultiNERD test predictions")
    for index, record in enumerate(records):
        validate_stored_record(record, index)
    return records


def failed_indices(records, *, smoke_limit=None) -> list:
    """Indices of stored parse failures in storage order (optionally only the first ``smoke_limit``).

    This is the only selection rule.  Selecting every sentence whose 256-token
    answer merely reached the limit (a future ``--all-limit-hits``) would need the
    generated token counts, which the canonical runs did not store.
    """
    _check_smoke_limit(smoke_limit)
    indices = [index for index, record in enumerate(records) if record["parse_status"] == "failed"]
    return indices if smoke_limit is None else indices[:smoke_limit]


def load_test_split() -> list:
    """Pinned MultiNERD English test rows with gold entities derived exactly as in the original inference."""
    from src.data.dataset_loader import load_ner_dataset

    raw, info = load_ner_dataset()
    rows = []
    for sample in raw["test"]:
        tokens, tags = list(sample["tokens"]), list(sample["ner_tags"])
        gold = extract_entities_from_bio(tokens, tags, info.id2label)
        rows.append({"tokens": tokens, "gold_entities": gold, "gold_bio": entities_to_bio(tokens, gold)})
    return rows


def verify_dataset_alignment(records, split_rows) -> dict:
    if len(records) != len(split_rows):
        raise ValueError(f"dataset/prediction mismatch: {len(split_rows)} dataset rows for {len(records)} predictions")
    for index, (record, row) in enumerate(zip(records, split_rows)):
        if any(record[key] != row[key] for key in ("tokens", "gold_entities", "gold_bio")):
            raise ValueError(f"dataset/prediction mismatch at index {index}")
    return {"dataset_revision": DATASET_REVISION, "verified_samples": len(records)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_entity_metrics(true, pred) -> dict:
    report = compute_per_entity_metrics(true, pred) if true else {}
    return {kind: report.get(kind, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}) for kind in ENTITY_TYPES}


def macro_f1(per_entity: dict) -> float:
    return statistics.fmean(per_entity[kind]["f1"] for kind in ENTITY_TYPES)


def llm_metrics(records) -> dict:
    """The exact metric path of the original decoder inference plus 15-type per-entity and macro F1."""
    quality = evaluate_llm_predictions(
        [r["tokens"] for r in records], [r["gold_entities"] for r in records],
        [r["pred_entities"] for r in records], [r["parse_status"] for r in records],
        [r["parse_diagnostics"] for r in records])
    per_entity = per_entity_metrics([r["gold_bio"] for r in records], [r["pred_bio"] for r in records])
    return {"test_f1": quality["f1"], "test_precision": quality["precision"], "test_recall": quality["recall"],
            **quality, "per_entity": per_entity, "macro_f1": macro_f1(per_entity)}


def source_inference_metrics(source: dict) -> dict:
    metrics = yaml.safe_load((Path(source["directory"]) / "inference_metrics.yaml").read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"{source['directory']}: invalid inference_metrics.yaml")
    return metrics


def baseline_metrics_256(source: dict, records) -> dict:
    """Recompute the stored 256-token result and prove that it equals the canonical metrics file."""
    recomputed = llm_metrics(records)
    stored = source_inference_metrics(source)
    for key in STORED_METRIC_KEYS:
        value = stored.get(key)
        if not _is_number(value) or abs(value - recomputed[key]) > 1e-12:
            raise ValueError(f"{source['directory']}: stored {key}={value!r} differs from the recomputed {recomputed[key]!r}")
    if stored.get("max_new_tokens") != BASELINE_LIMIT:
        raise ValueError(f"{source['directory']}: baseline output limit is not {BASELINE_LIMIT}")
    run_metadata = stored.get("run_metadata") or {}
    return {
        **recomputed,
        "stored_f1": stored["f1"], "stored_precision": stored["precision"], "stored_recall": stored["recall"],
        "recomputed_matches_stored": True, "max_new_tokens": BASELINE_LIMIT,
        "latency_ms_mean": stored.get("latency_ms_mean"), "latency_ms_p50": stored.get("latency_ms_p50"),
        "latency_ms_p95": stored.get("latency_ms_p95"), "latency_measurement": stored.get("latency_measurement"),
        "generation_samples_per_second": stored.get("generation_samples_per_second"),
        "vram_peak_mb": stored.get("vram_peak_mb"), "total_params": stored.get("total_params"),
        "gpu": ((run_metadata.get("cuda") or {}).get("devices") or [None])[0],
        "git_commit": (run_metadata.get("git") or {}).get("commit"),
    }


def delta_block(current: dict, baseline: dict) -> dict:
    return {
        "f1": current["f1"] - baseline["f1"], "precision": current["precision"] - baseline["precision"],
        "recall": current["recall"] - baseline["recall"], "macro_f1": current["macro_f1"] - baseline["macro_f1"],
        "parse_failed": current["parse_failed"] - baseline["parse_failed"],
        "per_entity_f1": {kind: current["per_entity"][kind]["f1"] - baseline["per_entity"][kind]["f1"] for kind in ENTITY_TYPES},
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def regeneration_contract(source: dict, selection, *, max_new_tokens: int = 1024, smoke_limit=None) -> dict:
    _check_limit(max_new_tokens)
    _check_smoke_limit(smoke_limit)
    if source["identity"]["model_type"] != "decoder":
        raise ValueError("The MultiNERD output-limit evaluation supports Qwen decoder checkpoints only")
    selection = [int(index) for index in selection]
    return {
        "version": 1,
        "evaluation": f"multinerd_output_limit_{max_new_tokens}",
        "source": source["identity"],
        "dataset": "multinerd", "language": "en", "split": "test",
        "dataset_revision": DATASET_REVISION, "expected_samples": EXPECTED_SAMPLES,
        "selection": {"rule": SELECTION_RULE, "count": len(selection), "indices": selection, "smoke_limit": smoke_limit},
        "decoder": {"prompt_types": list(ENTITY_TYPES), "thinking_enabled": False, "do_sample": False,
                    "max_new_tokens": max_new_tokens, "baseline_max_new_tokens": BASELINE_LIMIT, "batch_size": 1,
                    "precision": "qlora_4bit", "attention": "sdpa"},
        "metric": "strict_entity_level_iob2",
        "code_sha256": {path: file_hash(PROJECT_ROOT / path) for path in CODE_PATHS},
    }


def checkpoint_header(contract: dict) -> dict:
    return {"contract": contract, "contract_hash": canonical_hash(contract)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class RegenerationPredictor(OutputLimitPredictor):
    """Frozen QLoRA checkpoint, original prompt and parser, higher output limit, 15-type BIO output."""

    def __init__(self, source: dict, *, max_new_tokens: int = 1024):
        super().__init__(source, max_new_tokens=max_new_tokens)
        from src.evaluate.efficiency import count_parameters
        self.total_params = count_parameters(self.model)[0]

    def predict(self, index: int, tokens: list, stored_raw_output: str) -> dict:
        torch, tokenizer = self.torch, self.tokenizer
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
        entities, status, diagnostics = parse_llm_output_with_diagnostics(raw, tokens, frozenset(ENTITY_TYPES))
        prefix = tokenizer.decode(new_ids[:BASELINE_LIMIT], skip_special_tokens=True)
        return {
            "index": index, "raw_output": raw, "pred_entities": entities, "parse_status": status,
            "parse_diagnostics": diagnostics, "pred_bio": entities_to_bio(tokens, entities),
            "latency_ms": latency, "stored_prefix_match": prefix == stored_raw_output,
            **generation_diagnostics(new_ids, raw, tokenizer.eos_token_id, self.max_new_tokens),
        }


def build_regenerated_record(stored: dict, index: int, generated: dict) -> dict:
    return {
        "tokens": stored["tokens"], "gold_entities": stored["gold_entities"], "gold_bio": stored["gold_bio"],
        **{key: generated[key] for key in REGENERATED_KEYS + GENERATION_KEYS},
        "regenerated": True, "index": index,
        "baseline_256": {key: stored[key] for key in REGENERATED_KEYS},
    }


def validate_regenerated_record(record: dict, stored: dict, max_new_tokens: int) -> None:
    if not isinstance(record, dict) or set(record) != REGENERATED_RECORD_KEYS:
        raise ValueError("Regenerated record has unexpected keys")
    if type(record["index"]) is not int or record["regenerated"] is not True:
        raise ValueError("Regenerated record is missing its integer index or regenerated flag")
    if any(record[key] != stored[key] for key in ("tokens", "gold_entities", "gold_bio")):
        raise ValueError("Regenerated record does not belong to the stored sentence")
    if record["baseline_256"] != {key: stored[key] for key in REGENERATED_KEYS}:
        raise ValueError("Regenerated record carries a different 256-token baseline answer")
    raw = record["raw_output"]
    if (not isinstance(raw, str) or record["parse_status"] not in PARSE_STATUSES
            or not isinstance(record["parse_diagnostics"], dict) or not isinstance(record["pred_entities"], list)):
        raise ValueError("Regenerated record has invalid parse fields")
    if record["pred_bio"] != entities_to_bio(stored["tokens"], record["pred_entities"]):
        raise ValueError("Regenerated entities and BIO tags disagree")
    if record["parse_status"] == "failed" and (record["pred_entities"] or any(tag != "O" for tag in record["pred_bio"])):
        raise ValueError("Failed decoder parses must remain empty predictions")
    count = record["generated_token_count"]
    if type(count) is not int or not 0 < count <= max_new_tokens:
        raise ValueError("Regenerated record has an invalid generated token count")
    if type(record["token_limit_reached"]) is not bool or record["token_limit_reached"] != (count == max_new_tokens):
        raise ValueError("Regenerated record has inconsistent token-limit diagnostics")
    reason = record["stop_reason"]
    if (reason not in STOP_REASONS or (reason == "max_new_tokens" and count != max_new_tokens)
            or (reason == "other" and count == max_new_tokens)):
        raise ValueError("Regenerated record has inconsistent stop diagnostics")
    expected_json = generation_diagnostics([0], raw, None, max_new_tokens)["json_complete"]
    if type(record["json_complete"]) is not bool or record["json_complete"] != expected_json:
        raise ValueError("Regenerated record has inconsistent JSON diagnostics")
    if not _is_number(record["latency_ms"]) or record["latency_ms"] < 0:
        raise ValueError("Regenerated record has an invalid latency")
    if type(record["stored_prefix_match"]) is not bool:
        raise ValueError("Regenerated record is missing the stored-prefix comparison")


# ---------------------------------------------------------------------------
# Checkpoint, merge, summary
# ---------------------------------------------------------------------------

def read_regeneration_checkpoint(path: Path, contract: dict, selection, stored) -> list:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or json.loads(lines[0]) != checkpoint_header(contract):
        raise ValueError(f"{path}: regeneration contract mismatch")
    records = []
    limit = contract["decoder"]["max_new_tokens"]
    for i, line in enumerate(lines[1:]):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 2 and not line.endswith(b"\n"):
                break  # An interrupted final append is the only accepted corruption.
            raise ValueError(f"{path}: corrupt checkpoint record {i}")
        if i >= len(selection):
            raise ValueError(f"{path}: too many checkpoint records")
        if not isinstance(record, dict) or record.get("index") != selection[i]:
            raise ValueError(f"{path}: checkpoint record {i} is out of selection order")
        validate_regenerated_record(record, stored[selection[i]], limit)
        records.append(record)
    return records


def write_checkpoint_prefix(path: Path, contract: dict, records) -> None:
    """Rewrite the verified prefix atomically, discarding an interrupted append."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint_header(contract)) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge_predictions(stored, regenerated) -> list:
    """Stored records marked ``regenerated: False``; selected sentences replaced by their regenerated records.

    Nested lists stay shared with ``stored``; nothing in this module mutates them.
    """
    merged = [{**record, "regenerated": False} for record in stored]
    seen = set()
    for record in regenerated:
        index = record.get("index")
        if type(index) is not int or not 0 <= index < len(stored) or index in seen:
            raise ValueError(f"Regenerated index {index!r} is duplicated or out of range")
        if any(record.get(key) != stored[index][key] for key in ("tokens", "gold_entities", "gold_bio")):
            raise ValueError(f"Regenerated record {index} does not belong to the stored sentence")
        seen.add(index)
        merged[index] = {**record, "regenerated": True}
    return merged


def _quality(records) -> dict:
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return compute_ner_metrics([r["gold_bio"] for r in records], [r["pred_bio"] for r in records])


def _distribution(values) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "p50": None, "p95": None, "total": 0}
    import numpy as np
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "total": sum(values)}


def regeneration_summary(regenerated, stored, selection, max_new_tokens: int, *, vram_peak_mb=None, total_params=None) -> dict:
    selection = list(selection)
    if [record["index"] for record in regenerated] != selection:
        raise ValueError("Regenerated records do not match the selection")
    before = [stored[index] for index in selection]
    statuses = Counter(record["parse_status"] for record in regenerated)
    reasons = Counter(record["stop_reason"] for record in regenerated)
    gold_total = sum(len(record["gold_entities"]) for record in stored)
    gold_selected = sum(len(record["gold_entities"]) for record in before)
    tokens = _distribution([record["generated_token_count"] for record in regenerated])
    latencies = _distribution([record["latency_ms"] for record in regenerated])
    return {
        "selection_rule": SELECTION_RULE, "selected_count": len(selection), "selected_indices": selection,
        "truncated_without_closing_bracket": sum(not record["raw_output"].rstrip().endswith("]") for record in before),
        "gold_entities_in_selection": gold_selected,
        "gold_entity_share": gold_selected / gold_total if gold_total else None,
        "subset_metrics": {"baseline_256": _quality(before), f"output_limit_{max_new_tokens}": _quality(regenerated)},
        "parse_status_counts_after": {status: statuses.get(status, 0) for status in PARSE_STATUSES},
        "still_failed_count": statuses.get("failed", 0),
        "limit_hit_count": sum(record["token_limit_reached"] for record in regenerated),
        "stop_reason_counts": {reason: reasons.get(reason, 0) for reason in STOP_REASONS},
        "json_complete_count": sum(record["json_complete"] for record in regenerated),
        "generated_tokens": tokens,
        "latency_ms": {"mean": latencies["mean"], "p50": latencies["p50"], "p95": latencies["p95"],
                       "total_seconds": latencies["total"] / 1000, "measurement": LATENCY_MEASUREMENT},
        "stored_prefix_match_count": sum(record["stored_prefix_match"] for record in regenerated),
        "stored_prefix_mismatch_indices": [record["index"] for record in regenerated if not record["stored_prefix_match"]],
        "vram_peak_mb": vram_peak_mb, "total_params": total_params,
        "warmup": "first remaining selected sentence, unmeasured",
    }


def environment_report(source: dict, run_metadata: dict) -> dict:
    """Compare the current packages and GPU with the ones recorded by the canonical source run."""
    source_metadata = source_inference_metrics(source).get("run_metadata") or {}
    source_packages = {name: (details or {}).get("version") for name, details in (source_metadata.get("packages") or {}).items()}
    packages = {name: (details or {}).get("version") for name, details in (run_metadata.get("packages") or {}).items()}
    differences = {name: {"source": source_packages.get(name), "current": packages.get(name)}
                   for name in sorted(set(source_packages) | set(packages)) if source_packages.get(name) != packages.get(name)}
    gpu = ((run_metadata.get("cuda") or {}).get("devices") or [None])[0]
    source_gpu = ((source_metadata.get("cuda") or {}).get("devices") or [None])[0]
    return {"matches_source_packages": not differences, "package_differences": differences,
            "gpu": gpu, "source_gpu": source_gpu, "matches_source_gpu": gpu == source_gpu}


def control_sample(predictor, stored, control_limit: int) -> list:
    """Re-generate the first stored ``ok`` sentences: a direct test that untouched answers are limit-independent."""
    indices = [index for index, record in enumerate(stored) if record["parse_status"] == "ok"][:control_limit]
    rows = []
    for index in indices:
        record = stored[index]
        generated = predictor.predict(index, record["tokens"], record["raw_output"])
        rows.append({
            "index": index, "tokens": record["tokens"], "stored_raw_output": record["raw_output"],
            "raw_output": generated["raw_output"],
            "raw_output_equal_to_stored": generated["raw_output"] == record["raw_output"],
            "pred_entities_equal_to_stored": generated["pred_entities"] == record["pred_entities"],
            "stored_prefix_match": generated["stored_prefix_match"], "parse_status": generated["parse_status"],
            "generated_token_count": generated["generated_token_count"], "stop_reason": generated["stop_reason"],
            "latency_ms": generated["latency_ms"],
        })
    return rows


def build_metrics(run, source: dict, contract: dict, merged, baseline: dict, summary: dict, *, predictions_path,
                  environment: dict, run_metadata: dict, started_at: str, started_monotonic: float,
                  dataset_alignment=None, control=None) -> dict:
    cfg, identity = source["config"], source["identity"]
    limit = contract["decoder"]["max_new_tokens"]
    smoke_limit = contract["selection"]["smoke_limit"]
    current = llm_metrics(merged)
    job_script = PROJECT_ROOT / JOB_SCRIPT
    return {
        "evaluation": f"output_limit_{limit}",
        "experiment_name": cfg["experiment_name"], "model_name": cfg["model_name"], "model_type": "decoder",
        "regime": run.regime, "dataset": cfg["dataset"], "dataset_language": cfg["dataset_language"],
        "seed": int(cfg["seed"]), "variant": run.variant, "canonical": False,
        "derived_from": {"experiment_name": identity["experiment_name"],
                         "directory": str(Path(source["directory"]).resolve()),
                         "predictions_sha256": identity["source_predictions_sha256"],
                         "checkpoint": identity["checkpoint"], "checkpoint_sha256": identity["checkpoint_sha256"]},
        "max_new_tokens": limit, "baseline_max_new_tokens": BASELINE_LIMIT, "sample_count": len(merged),
        "complete": smoke_limit is None and len(merged) == contract["expected_samples"], "smoke_limit": smoke_limit,
        **current,
        "baseline_256": baseline,
        "delta_vs_256": delta_block(current, baseline),
        "regeneration": summary,
        "dataset_alignment": dataset_alignment,
        "control_sample": control,
        "environment": environment,
        "vram_peak_mb": summary["vram_peak_mb"], "total_params": summary["total_params"],
        "contract_hash": canonical_hash(contract), "contract": contract,
        "predictions_sha256": file_hash(Path(predictions_path)),
        "code_hashes": {"src/evaluate/multinerd_output_limit.py": file_hash(Path(__file__)),
                        JOB_SCRIPT: file_hash(job_script) if job_script.is_file() else None,
                        **contract["code_sha256"]},
        "run_metadata": run_metadata,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started_at, "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    }


def completed_experiment(directory: Path, contract: dict) -> bool:
    directory = Path(directory)
    metrics_path, predictions_path = directory / "inference_metrics.yaml", directory / "test_predictions.json"
    if not metrics_path.exists():
        return False
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("contract_hash") != canonical_hash(contract) or metrics.get("contract") != contract:
        raise ValueError(f"{directory}: existing evaluation contract differs")
    if not predictions_path.exists() or metrics.get("predictions_sha256") != file_hash(predictions_path):
        raise ValueError(f"{directory}: completed predictions missing or altered")
    records = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != contract["expected_samples"]:
        raise ValueError(f"{directory}: incomplete saved predictions")
    regenerated = [r.get("index") for r in records if isinstance(r, dict) and r.get("regenerated") is True]
    if regenerated != contract["selection"]["indices"]:
        raise ValueError(f"{directory}: regenerated indices differ from the contract selection")
    return True


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

@contextmanager
def run_lock(directory: Path, *, resume_failed: bool = False):
    """Exclusive per-experiment lock; a stale lock is recoverable with ``resume_failed`` or after a SLURM requeue."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_NAME
    payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID")}
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    if path.exists():
        owner = json.loads(path.read_text(encoding="utf-8"))
        same_requeued_job = bool(payload["job_id"]) and owner.get("job_id") == payload["job_id"] and restart_count > 0
        if same_requeued_job:
            pass  # The previous attempt of this very SLURM job is dead by definition; squeue would report it alive.
        elif not resume_failed:
            raise RuntimeError(f"MultiNERD output-limit run is locked: {path}")
        elif owner.get("job_id"):
            check = subprocess.run(["squeue", "-h", "-j", str(owner["job_id"]), "-o", "%i"], capture_output=True, text=True, timeout=20)
            if check.returncode or check.stdout.strip():
                raise RuntimeError(f"Cannot recover a live or unverified lock: {owner}")
        elif owner.get("hostname") == socket.gethostname():
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"MultiNERD output-limit process is still alive: {owner}")
        else:
            raise RuntimeError(f"Cannot verify remote non-SLURM lock: {owner}")
        path.unlink()
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"MultiNERD output-limit run is locked: {path}") from exc
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) == payload:
            path.unlink()


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_evaluation(run, source_root, output_root, *, max_new_tokens: int = 1024, smoke_limit=None,
                   resume_failed: bool = False, control_limit: int = 0) -> dict:
    from src.run_metadata import collect_run_metadata

    _check_limit(max_new_tokens)
    _check_smoke_limit(smoke_limit)
    if run.group_key not in GROUPS or run.model_family != "decoder":
        raise ValueError("Only the canonical Qwen QLoRA runs can receive a higher output limit")
    if type(control_limit) is not int or control_limit < 0:
        raise ValueError("control_limit must be a nonnegative integer")
    if control_limit and smoke_limit is None:
        raise ValueError("Control sentences are only re-generated in smoke runs")
    source_root, output_root = Path(source_root), Path(output_root)
    ensure_output_safe(output_root, source_root, smoke_limit=smoke_limit)
    cfg = load_experiment_config(PROJECT_ROOT / run.config_path)
    experiment = cfg["experiment_name"]
    directory = output_root / experiment
    started_at, started = _utc_now(), time.monotonic()
    verified, measured = [], []
    status = {
        "status": "RUNNING", "evaluation": f"multinerd_output_limit_{max_new_tokens}", "model": run.group_key,
        "experiment_name": experiment, "seed": run.seed, "max_new_tokens": max_new_tokens, "smoke_limit": smoke_limit,
        "job_id": os.environ.get("SLURM_JOB_ID"), "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0),
        "started_at": started_at, "updated_at": started_at, "elapsed_seconds": 0.0, "selected_count": None,
        "completed_count": 0, "eta_seconds": None, "baseline_f1": None, "still_failed_count": 0,
        "limit_hit_count": 0, "stored_prefix_match_count": 0, "error": None,
    }

    def update_status(**changes) -> None:
        status.update(changes)
        remaining = (status["selected_count"] or 0) - len(verified)
        if measured and remaining > 0:
            eta = statistics.fmean(measured) / 1000 * remaining
        else:
            eta = 0.0 if status["selected_count"] is not None and remaining == 0 else None
        status.update(updated_at=_utc_now(), elapsed_seconds=time.monotonic() - started, completed_count=len(verified),
                      eta_seconds=eta, still_failed_count=sum(r["parse_status"] == "failed" for r in verified),
                      limit_hit_count=sum(r["token_limit_reached"] for r in verified),
                      stored_prefix_match_count=sum(r["stored_prefix_match"] for r in verified))
        atomic_write_json(directory / "status.json", status)

    predictor = None
    with run_lock(directory, resume_failed=resume_failed):
        update_status()
        try:
            source = source_run(run, source_root)
            stored = load_source_predictions(source)
            selection = failed_indices(stored, smoke_limit=smoke_limit)
            update_status(selected_count=len(selection))
            alignment = verify_dataset_alignment(stored, load_test_split())
            baseline = baseline_metrics_256(source, stored)
            update_status(baseline_f1=baseline["f1"])
            contract = regeneration_contract(source, selection, max_new_tokens=max_new_tokens, smoke_limit=smoke_limit)
            if completed_experiment(directory, contract):
                metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text(encoding="utf-8"))
                saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
                verified.extend(r for r in saved if r.get("regenerated") is True)
                update_status(status="COMPLETED")
                print(f"SKIP_COMPLETED {experiment} seed={run.seed} max_new_tokens={max_new_tokens}", flush=True)
                return metrics
            atomic_write_json(directory / "contract.json", contract)
            checkpoint = directory / CHECKPOINT_NAME
            records = read_regeneration_checkpoint(checkpoint, contract, selection, stored)
            write_checkpoint_prefix(checkpoint, contract, records)
            verified.extend(records)
            update_status()
            if len(records) < len(selection) or control_limit:
                predictor = RegenerationPredictor(source, max_new_tokens=max_new_tokens)
                first = stored[selection[len(records)]] if len(records) < len(selection) else stored[0]
                predictor.predict(selection[len(records)] if len(records) < len(selection) else 0,
                                  first["tokens"], first["raw_output"])  # Same unmeasured warm-up as the original inference.
            with checkpoint.open("a", encoding="utf-8") as handle:
                for position in range(len(records), len(selection)):
                    index = selection[position]
                    generated = predictor.predict(index, stored[index]["tokens"], stored[index]["raw_output"])
                    record = build_regenerated_record(stored[index], index, generated)
                    validate_regenerated_record(record, stored[index], max_new_tokens)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    verified.append(record)
                    measured.append(record["latency_ms"])
                    update_status()
                    print(f"MNOL_PROGRESS {experiment} seed={run.seed} {len(records)}/{len(selection)} max_new_tokens={max_new_tokens}", flush=True)
            control = None
            if control_limit:
                rows = control_sample(predictor, stored, control_limit)
                atomic_write_json(directory / "control_predictions.json", rows)
                control = {"count": len(rows), "indices": [row["index"] for row in rows],
                           "raw_output_equal_count": sum(row["raw_output_equal_to_stored"] for row in rows),
                           "pred_entities_equal_count": sum(row["pred_entities_equal_to_stored"] for row in rows)}
            merged = merge_predictions(stored, records)
            predictions_path = directory / "test_predictions.json"
            atomic_write_json(predictions_path, merged)
            vram = predictor.torch.cuda.max_memory_allocated() / 1024**2 if predictor is not None else None
            total_params = predictor.total_params if predictor is not None else None
            summary = regeneration_summary(records, stored, selection, max_new_tokens, vram_peak_mb=vram, total_params=total_params)
            run_metadata = collect_run_metadata(cfg)
            metrics = build_metrics(run, source, contract, merged, baseline, summary, predictions_path=predictions_path,
                                    environment=environment_report(source, run_metadata), run_metadata=run_metadata,
                                    started_at=started_at, started_monotonic=started, dataset_alignment=alignment, control=control)
            atomic_write_yaml(directory / "inference_metrics.yaml", metrics)
            checkpoint.unlink()
            update_status(status="COMPLETED")
            print(f"MNOL_RESULT {experiment} seed={run.seed} f1_256={baseline['f1']:.6f} "
                  f"f1_{max_new_tokens}={metrics['f1']:.6f} delta={metrics['delta_vs_256']['f1']:.6f} "
                  f"still_failed={summary['still_failed_count']} limit_hit={summary['limit_hit_count']} "
                  f"prefix_match={summary['stored_prefix_match_count']}/{summary['selected_count']}", flush=True)
            return metrics
        except BaseException as exc:
            update_status(status="FAILED", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Preflight (CPU)
# ---------------------------------------------------------------------------

def base_model_snapshot(run, source: dict, source_root) -> str:
    """Require the pinned base-model snapshot in the local Hub cache; never download."""
    from huggingface_hub import snapshot_download

    default_home = str(Path(source_root).resolve().parent / "hf-cache")
    cache = os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", default_home)) / "hub"))
    snapshot = Path(snapshot_download(run.model_name, revision=source["identity"]["model_revision"], cache_dir=cache,
                                      allow_patterns=["config.json", "generation_config.json", "model*.safetensors",
                                                      "model.safetensors.index.json", "*.py"],
                                      local_files_only=True))
    index = snapshot / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    if not (snapshot / "config.json").is_file() or any(not (snapshot / shard).is_file() for shard in shards):
        raise ValueError(f"{snapshot}: missing base-model weight shards")
    return str(snapshot)


def preflight(runs, source_root, output_root, *, max_new_tokens: int = 1024, smoke_limit=None, verify_dataset: bool = False) -> dict:
    _check_limit(max_new_tokens)
    source_root, output_root = Path(source_root), Path(output_root)
    ensure_output_safe(output_root, source_root, smoke_limit=smoke_limit)
    split_rows = load_test_split() if verify_dataset else None
    report = {"evaluation": f"multinerd_output_limit_{max_new_tokens}", "max_new_tokens": max_new_tokens,
              "smoke_limit": smoke_limit, "dataset_verified": verify_dataset, "runs": [], "selected": {}}
    for run in runs:
        source = source_run(run, source_root)
        experiment = source["config"]["experiment_name"]
        stored = load_source_predictions(source)
        selection = failed_indices(stored, smoke_limit=smoke_limit)
        baseline = baseline_metrics_256(source, stored)
        if split_rows is not None:
            verify_dataset_alignment(stored, split_rows)
        snapshot = base_model_snapshot(run, source, source_root)
        directory = output_root / experiment
        status = _load_status(directory)
        if (directory / LOCK_NAME).exists() or status.get("status") == "RUNNING":
            raise RuntimeError(f"{experiment}: an evaluation lock or RUNNING status exists in {directory}; "
                               "confirm that the job is dead before resubmitting with --resume-failed")
        contract = regeneration_contract(source, selection, max_new_tokens=max_new_tokens, smoke_limit=smoke_limit)
        completed = completed_experiment(directory, contract)
        entry = {"model": run.group_key, "seed": run.seed, "experiment_name": experiment,
                 "failed_count": len(failed_indices(stored)), "selected_count": len(selection),
                 "baseline_f1": baseline["f1"], "base_model_snapshot": snapshot, "completed": completed,
                 "status": status.get("status")}
        report["runs"].append(entry)
        report["selected"][experiment] = len(selection)
        print(f"MNOL_PREFLIGHT {experiment} seed={run.seed} failed={entry['failed_count']} selected={len(selection)} "
              f"baseline_f1={baseline['f1']:.6f} completed={completed}", flush=True)
    print(f"MNOL_PREFLIGHT_OK runs={len(runs)} selected={json.dumps(report['selected'])}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Aggregate (CPU)
# ---------------------------------------------------------------------------

def verify_merged_artifact(merged, stored, selection, metrics: dict, baseline: dict, max_new_tokens: int) -> None:
    """Re-verify a completed experiment from its artifacts before it enters the comparison."""
    if not isinstance(merged, list) or len(merged) != len(stored):
        raise ValueError("Merged predictions do not cover the stored test set")
    selected, regenerated = set(selection), []
    for index, (record, original) in enumerate(zip(merged, stored)):
        if index in selected:
            if not isinstance(record, dict) or record.get("regenerated") is not True or record.get("index") != index:
                raise ValueError(f"Selected sentence {index} was not regenerated")
            validate_regenerated_record(record, original, max_new_tokens)
            regenerated.append(record)
        elif (not isinstance(record, dict) or record.get("regenerated") is not False
              or {key: value for key, value in record.items() if key != "regenerated"} != original):
            raise ValueError(f"Untouched sentence {index} differs from the stored prediction")
    recomputed = llm_metrics(merged)
    if not _close(recomputed, {key: metrics.get(key) for key in recomputed}):
        raise ValueError("Saved metrics differ from the verified merged predictions")
    if not _close(baseline, metrics.get("baseline_256")):
        raise ValueError("Saved baseline_256 differs from the recomputed source baseline")
    if not _close(delta_block(recomputed, baseline), metrics.get("delta_vs_256")):
        raise ValueError("Saved delta_vs_256 differs from the recomputed difference")
    summary = metrics.get("regeneration")
    if not isinstance(summary, dict):
        raise ValueError("Saved regeneration summary is missing")
    expected = regeneration_summary(regenerated, stored, selection, max_new_tokens,
                                    vram_peak_mb=summary.get("vram_peak_mb"), total_params=summary.get("total_params"))
    if not _close(expected, summary):
        raise ValueError("Saved regeneration summary differs from the verified records")
    _validated_memory(metrics.get("vram_peak_mb"))


def seed_metric_rows(run, experiment: str, metrics: dict, baseline: dict, evaluation: str) -> list:
    base = {"model": run.group_key, "seed": run.seed, "experiment_name": experiment, "sample_count": metrics["sample_count"]}
    summary, delta = metrics["regeneration"], metrics["delta_vs_256"]
    return [
        {**base, "evaluation": "baseline_256", "max_new_tokens": BASELINE_LIMIT,
         "precision": baseline["precision"], "recall": baseline["recall"], "f1": baseline["f1"], "macro_f1": baseline["macro_f1"],
         "parse_failed": baseline["parse_failed"], "parse_failure_rate": baseline["parse_failure_rate"],
         "regenerated_count": None, "still_failed_count": None, "limit_hit_count": None, "stored_prefix_mismatch_count": None,
         "generated_tokens_mean": None, "regenerated_latency_ms_mean": None, "vram_peak_mb": baseline["vram_peak_mb"],
         "delta_f1_vs_256": None, "delta_recall_vs_256": None, "delta_precision_vs_256": None, "delta_macro_f1_vs_256": None,
         "per_entity_f1": {kind: baseline["per_entity"][kind]["f1"] for kind in ENTITY_TYPES}},
        {**base, "evaluation": evaluation, "max_new_tokens": metrics["max_new_tokens"],
         "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "macro_f1": metrics["macro_f1"],
         "parse_failed": metrics["parse_failed"], "parse_failure_rate": metrics["parse_failure_rate"],
         "regenerated_count": summary["selected_count"], "still_failed_count": summary["still_failed_count"],
         "limit_hit_count": summary["limit_hit_count"],
         "stored_prefix_mismatch_count": summary["selected_count"] - summary["stored_prefix_match_count"],
         "generated_tokens_mean": summary["generated_tokens"]["mean"], "regenerated_latency_ms_mean": summary["latency_ms"]["mean"],
         "vram_peak_mb": metrics["vram_peak_mb"], "delta_f1_vs_256": delta["f1"], "delta_recall_vs_256": delta["recall"],
         "delta_precision_vs_256": delta["precision"], "delta_macro_f1_vs_256": delta["macro_f1"],
         "per_entity_f1": {kind: metrics["per_entity"][kind]["f1"] for kind in ENTITY_TYPES}},
    ]


def summarize(values) -> dict:
    values = [value for value in values if value is not None]
    return {"mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def summary_rows(seed_rows, models, seeds, evaluation: str, max_new_tokens: int) -> list:
    rows = []
    for model in models:
        for kind in ("baseline_256", evaluation):
            selected = [r for r in seed_rows if r["model"] == model and r["evaluation"] == kind]
            row = {"model": model, "evaluation": kind,
                   "max_new_tokens": BASELINE_LIMIT if kind == "baseline_256" else max_new_tokens,
                   "successful_seeds": len(selected), "expected_seeds": len(seeds),
                   "seeds": ",".join(str(r["seed"]) for r in selected), "complete": len(selected) == len(seeds),
                   "sample_count": selected[0]["sample_count"] if selected else None}
            for field in SUMMARY_FIELDS:
                stats = summarize([r[field] for r in selected])
                row[field + "_mean"], row[field + "_std"] = stats["mean"], stats["std"]
            rows.append(row)
    return rows


def per_entity_summary_rows(seed_rows, models, seeds, evaluation: str) -> list:
    rows = []
    for model in models:
        baseline_by_seed = {r["seed"]: r for r in seed_rows if r["model"] == model and r["evaluation"] == "baseline_256"}
        for kind in ("baseline_256", evaluation):
            selected = [r for r in seed_rows if r["model"] == model and r["evaluation"] == kind]
            for entity in ENTITY_TYPES:
                f1 = summarize([r["per_entity_f1"][entity] for r in selected])
                if kind == "baseline_256":
                    delta = {"mean": None, "std": None}
                else:
                    delta = summarize([r["per_entity_f1"][entity] - baseline_by_seed[r["seed"]]["per_entity_f1"][entity]
                                       for r in selected if r["seed"] in baseline_by_seed])
                rows.append({"model": model, "evaluation": kind, "entity_type": entity, "successful_seeds": len(selected),
                             "expected_seeds": len(seeds), "f1_mean": f1["mean"], "f1_std": f1["std"],
                             "delta_f1_vs_256_mean": delta["mean"], "delta_f1_vs_256_std": delta["std"]})
    return rows


def _display(row: dict, field: str, scale: float = 100) -> str:
    value, deviation = row[field + "_mean"], row[field + "_std"]
    if value is None:
        return "—"
    return f"{value * scale:.2f}" + (f" ± {deviation * scale:.2f}" if deviation is not None else " (SD n/a)")


def aggregate(source_root, output_root, models=GROUPS, seeds=SEEDS, *, max_new_tokens: int = 1024) -> dict:
    _check_limit(max_new_tokens)
    source_root, output_root = Path(source_root), Path(output_root)
    ensure_output_safe(output_root, source_root)
    runs = selected_runs(models, seeds)
    evaluation = f"output_limit_{max_new_tokens}"
    seed_rows, missing, successful = [], [], 0
    for run in runs:
        try:
            source = source_run(run, source_root)
            experiment = source["config"]["experiment_name"]
            stored = load_source_predictions(source)
            selection = failed_indices(stored)
            baseline = baseline_metrics_256(source, stored)
            directory = output_root / experiment
            contract = regeneration_contract(source, selection, max_new_tokens=max_new_tokens)
            if not completed_experiment(directory, contract):
                raise ValueError("Missing final metrics")
            metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text(encoding="utf-8"))
            if not metrics.get("complete") or metrics.get("smoke_limit") is not None or metrics.get("sample_count") != len(stored):
                raise ValueError("Smoke or incomplete evaluation is excluded")
            merged = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
            verify_merged_artifact(merged, stored, selection, metrics, baseline, max_new_tokens)
            seed_rows.extend(seed_metric_rows(run, experiment, metrics, baseline, evaluation))
            successful += 1
        except INVALID_ARTIFACT_ERRORS as exc:
            missing.append({"model": run.group_key, "seed": run.seed, "reason": str(exc)})
    summary = summary_rows(seed_rows, models, seeds, evaluation, max_new_tokens)
    per_entity = per_entity_summary_rows(seed_rows, models, seeds, evaluation)
    primary = set(models) == set(GROUPS) and set(seeds) == set(SEEDS)
    report = {
        "complete": primary and not missing, "selection_complete": not missing,
        "evaluation": evaluation, "max_new_tokens": max_new_tokens, "baseline_max_new_tokens": BASELINE_LIMIT,
        "expected_runs": len(runs), "successful_runs": successful, "primary_expected_runs": len(GROUPS) * len(SEEDS),
        "ddof": 1, "selection": {"models": list(models), "seeds": list(seeds)}, "selection_rule": SELECTION_RULE,
        "source_results_root": str(source_root.resolve()), "output_root": str(output_root.resolve()),
        "interpretation": "Post-hoc output-limit sensitivity on the MultiNERD test set: stored 256-token parse failures "
                          "re-generated at a higher limit with fixed checkpoints, prompt, parser and 15-type scoring; "
                          "untouched sentences keep their stored answers",
        "summary": summary, "per_entity_summary": per_entity, "seed_metrics": seed_rows, "missing_or_failed": missing,
    }
    target = output_root / "aggregate"
    atomic_write_yaml(target / "comparison.yaml", report)
    write_csv(target / "comparison.csv", summary)
    write_csv(target / "seed_metrics.csv", [{k: v for k, v in row.items() if k != "per_entity_f1"} for row in seed_rows])
    write_csv(target / "per_entity_summary.csv", per_entity)
    lines = ["# MultiNERD output-limit re-generation", "",
             f"Status: {'COMPLETE' if report['complete'] else 'INCOMPLETE / SUBSET'}; {successful}/{len(runs)} verified runs.",
             f"Qwen output limit: {max_new_tokens} for stored parse failures; reference: {BASELINE_LIMIT}. No new training.", "",
             "Values are percentages, mean ± sample SD over seeds 42/123/456 (ddof=1); Δ in percentage points vs the stored "
             "256-token predictions; untouched sentences keep their stored answers. Counts, token means, latencies and VRAM are "
             "shown unscaled; the re-generation latency covers only the re-generated sentences.", "",
             "| Model | Limit | Seeds | Precision | Recall | Micro-F1 | Macro-F1 | ΔF1 vs 256 | ΔRecall vs 256 | Parse failed "
             "| Re-generated | Still at limit | Gen. tokens | Latency re-gen ms | VRAM MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        values = [_display(row, field) for field in ("precision", "recall", "f1", "macro_f1", "delta_f1_vs_256", "delta_recall_vs_256")]
        counts = [_display(row, field, 1) for field in ("parse_failed", "regenerated_count", "limit_hit_count",
                                                        "generated_tokens_mean", "regenerated_latency_ms_mean", "vram_peak_mb")]
        lines.append(f"| {row['model']} | {row['max_new_tokens']} | {row['successful_seeds']}/{row['expected_seeds']} | "
                     + " | ".join(values + counts) + " |")
    lines.append("")
    if missing:
        lines += ["Missing or failed runs:", ""] + [f"- {m['model']}, seed {m['seed']}: {m['reason']}" for m in missing]
    atomic_write_bytes(target / "comparison.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="canonical Qwen QLoRA config whose stored parse failures are re-generated (GPU)")
    mode.add_argument("--aggregate", action="store_true", help="verify completed runs and write aggregate/comparison.* (CPU)")
    mode.add_argument("--preflight", action="store_true", help="verify sources, selections, locks and cached base models (CPU)")
    parser.add_argument("--source-results-root", type=Path, required=True, help="results root holding multinerd/<experiment>")
    parser.add_argument("--output-root", type=Path, required=True, help="separate root for this evaluation's outputs")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="higher output limit (> 256; default 1024)")
    parser.add_argument("--models", nargs="+", choices=GROUPS, default=list(GROUPS), help="aggregate/preflight selection")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS), help="aggregate/preflight selection")
    parser.add_argument("--smoke-limit", type=int, help="re-generate only the first N selected sentences (separate output root)")
    parser.add_argument("--resume-failed", action="store_true", help="recover a stale lock of a dead job and resume the checkpoint")
    parser.add_argument("--control-limit", type=int, default=0, help="smoke only: also re-generate the first K stored ok sentences")
    parser.add_argument("--verify-dataset", action="store_true", help="preflight only: align stored predictions with the pinned dataset")
    args = parser.parse_args(argv)
    try:
        _check_limit(args.max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke_limit is not None and args.smoke_limit < 1:
        parser.error("--smoke-limit must be positive")
    if args.control_limit < 0:
        parser.error("--control-limit must be nonnegative")
    if args.control_limit and args.smoke_limit is None:
        parser.error("--control-limit requires --smoke-limit")
    if args.aggregate and (args.smoke_limit is not None or args.resume_failed or args.control_limit):
        parser.error("Aggregate mode does not accept smoke, control or resume flags")
    if args.verify_dataset and not args.preflight:
        parser.error("--verify-dataset applies to --preflight only (runs always verify the dataset)")
    models, seeds = tuple(dict.fromkeys(args.models)), tuple(dict.fromkeys(args.seeds))
    if args.aggregate:
        report = aggregate(args.source_results_root, args.output_root, models, seeds, max_new_tokens=args.max_new_tokens)
        print(f"MNOL_AGGREGATE {report['successful_runs']}/{report['expected_runs']} complete={report['complete']}", flush=True)
        return 0 if report["selection_complete"] else 1
    if args.preflight:
        try:
            preflight(selected_runs(models, seeds), args.source_results_root, args.output_root,
                      max_new_tokens=args.max_new_tokens, smoke_limit=args.smoke_limit, verify_dataset=args.verify_dataset)
        except (RuntimeError, *INVALID_ARTIFACT_ERRORS) as exc:
            print(f"MNOL_PREFLIGHT_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    try:
        run = run_for_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    run_evaluation(run, args.source_results_root, args.output_root, max_new_tokens=args.max_new_tokens,
                   smoke_limit=args.smoke_limit, resume_failed=args.resume_failed, control_limit=args.control_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
