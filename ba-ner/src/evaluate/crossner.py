"""Inference-only domain transfer evaluation of frozen MultiNERD checkpoints.

Source runs are read-only. CrossNER has its own contract, locks and outputs,
independent of the managed training/inference phases of the primary study.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import yaml

from src.config import load_experiment_config
from src.data.crossner_loader import (
    COMMON_TYPES, CROSSNER_REVISION, DOMAINS, TEST_SHA256, TEST_SIZES,
    TYPE_MAPPING, load_crossner_test, project_multinerd_bio,
)
from src.data.dataset_loader import MULTINERD_INFO, MULTINERD_LABEL_LIST
from src.evaluate.metrics import compute_ner_metrics, compute_per_entity_metrics
from src.seed_provenance import MODEL_REVISIONS, canonical_hash, scientific_contract
from src.seed_study import (
    atomic_write_json, atomic_write_yaml, full_run_config_hash,
    iter_run_descriptors, scientific_config_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("deberta-v3-large", "qwen35-27b-qlora-3ep")
SEEDS = (42, 123, 456)
_FILE_HASHES: dict[tuple, str] = {}


def file_hash(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _FILE_HASHES[key] = digest.hexdigest()
    return _FILE_HASHES[key]


def selected_runs(models=GROUPS, seeds=SEEDS):
    if not models or not seeds or set(models) - set(GROUPS) or set(seeds) - set(SEEDS):
        raise ValueError("Select supported model groups and seeds 42, 123, 456")
    return [run for run in iter_run_descriptors(include_historical=False)
            if run.group_key in models and run.seed in seeds]


def source_run(run, source_root: Path) -> dict:
    """Verify checkpoint ownership against the unmodified canonical contract."""
    cfg = load_experiment_config(PROJECT_ROOT / run.config_path)
    directory = source_root / Path(run.output_dir).relative_to("results")
    checkpoint = directory / ("best_model" if run.model_family == "encoder" else "best_lora_adapter")
    metadata = yaml.safe_load((directory / "run_metadata.yaml").read_text())
    metrics = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
    status = json.loads((directory / "status.json").read_text())
    contract = yaml.safe_load((directory / "scientific_contract.yaml").read_text())
    resolved = yaml.safe_load((directory / "config_resolved.yaml").read_text())
    if resolved != cfg or contract != scientific_contract(cfg):
        raise ValueError(f"{directory}: source configuration/scientific contract mismatch")
    scientific_hash, full_hash = scientific_config_hash(cfg), full_run_config_hash(cfg)
    expected = {"seed": run.seed, "canonical": True,
                "scientific_config_hash": scientific_hash, "full_run_config_hash": full_hash}
    for name, document in (("metadata", metadata), ("metrics", metrics), ("status", status)):
        if any(document.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{directory}: {name} does not identify this canonical run")
    if status.get("status") != "COMPLETED" or metadata.get("model_name") != run.model_name:
        raise ValueError(f"{directory}: source run is incomplete or has a different model")
    selection = contract["evaluation"]["checkpoint_selection"]
    if metrics.get("checkpoint_selection") != selection:
        raise ValueError(f"{directory}: incorrect validation checkpoint selection")
    for document in (metadata, metrics):
        if Path(document.get("checkpoint_used", "")).resolve() != checkpoint.resolve():
            raise ValueError(f"{directory}: checkpoint ownership mismatch")
    if metadata.get("model_revision") != MODEL_REVISIONS[run.model_name]:
        raise ValueError(f"{directory}: base-model revision mismatch")
    if run.model_family == "encoder":
        model_config = json.loads((checkpoint / "config.json").read_text())
        if {int(k): v for k, v in model_config["id2label"].items()} != MULTINERD_INFO.id2label:
            raise ValueError(f"{checkpoint}: incorrect MultiNERD label schema")
        required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    else:
        adapter = json.loads((checkpoint / "adapter_config.json").read_text())
        if adapter.get("base_model_name_or_path") != run.model_name or not adapter.get("inference_mode"):
            raise ValueError(f"{checkpoint}: adapter/base-model mismatch")
        if not cfg.get("use_qlora") or cfg.get("attn_impl", "sdpa") != "sdpa":
            raise ValueError("CrossNER requires the frozen QLoRA/SDPA setup")
        settings = {"max_new_tokens": 256, "inference_batch_size": 1, "precision_mode": "qlora_4bit", "attn_impl": "sdpa", "thinking_enabled": False}
        if any(metrics.get(key) != value for key, value in settings.items()):
            raise ValueError(f"{directory}: frozen decoder inference settings differ")
        required = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    checkpoint_hashes = {name: file_hash(checkpoint / name) for name in required}
    predictions = directory / "test_predictions.json"
    identity = {
        "experiment_name": cfg["experiment_name"], "model_name": run.model_name,
        "model_type": run.model_family, "seed": run.seed, "variant": run.variant,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_selection": selection, "model_revision": MODEL_REVISIONS[run.model_name],
        "source_scientific_config_hash": scientific_hash, "source_full_run_config_hash": full_hash,
        "source_predictions_sha256": file_hash(predictions),
    }
    return {"config": cfg, "directory": directory, "checkpoint": checkpoint,
            "predictions": predictions, "identity": identity}


def evaluation_contract(source: dict, domain: str, smoke_limit: int | None = None) -> dict:
    code_paths = ("src/data/crossner_loader.py", "src/evaluate/crossner.py",
                  "src/evaluate/metrics.py", "src/data/preprocess_decoder.py",
                  "src/decoder/generation.py", "src/decoder/parse_output.py")
    return {
        "version": 1, "source": source["identity"], "dataset": "crossner", "language": "en",
        "revision": CROSSNER_REVISION, "domain": domain, "split": "test",
        "data_sha256": TEST_SHA256[domain], "expected_samples": TEST_SIZES[domain],
        "scored_types": list(COMMON_TYPES), "mapping": TYPE_MAPPING,
        "unsupported_gold": "O; shared-type predictions count as false positives",
        "metric": "strict_entity_level_iob2", "smoke_limit": smoke_limit,
        "encoder_max_length": source["config"].get("max_length", 256),
        "decoder": {"prompt_types": MULTINERD_INFO.entity_types, "thinking_enabled": False,
                    "do_sample": False, "max_new_tokens": 256, "batch_size": 1,
                    "precision": "qlora_4bit", "attention": "sdpa"},
        "code_sha256": {p: file_hash(PROJECT_ROOT / p) for p in code_paths},
    }


def ensure_safe_output(output: Path, source_root: Path) -> None:
    output, source = output.resolve(), (source_root / "multinerd").resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("CrossNER output must not overlap MultiNERD source runs")


def baseline_metrics(source: dict) -> dict:
    records = json.loads(source["predictions"].read_text())
    if len(records) != 16_454:
        raise ValueError("MultiNERD baseline is not the complete English test split")
    gold_key, pred_key = ("gold", "pred") if source["identity"]["model_type"] == "encoder" else ("gold_bio", "pred_bio")
    true, pred = [], []
    for sample in records:
        if len(sample["tokens"]) != len(sample[gold_key]) or len(sample[gold_key]) != len(sample[pred_key]):
            raise ValueError("MultiNERD baseline has inconsistent word coverage")
        true.append(project_multinerd_bio(sample[gold_key]))
        pred.append(project_multinerd_bio(sample[pred_key]))
    return {**compute_ner_metrics(true, pred), "per_entity": per_entity_metrics(true, pred),
            "sample_count": len(records), "scored_types": list(COMMON_TYPES),
            "source_predictions_sha256": source["identity"]["source_predictions_sha256"]}


def per_entity_metrics(true, pred) -> dict:
    report = compute_per_entity_metrics(true, pred)
    return {kind: report.get(kind, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}) for kind in COMMON_TYPES}


def validate_record(record: dict, sample: dict) -> None:
    if not isinstance(record, dict) or any(record.get(k) != sample[k] for k in ("sample_id", "tokens", "native_bio", "gold_bio")):
        raise ValueError("Inference checkpoint sample identity mismatch")
    if not isinstance(record.get("raw_pred_bio"), list) or len(record["raw_pred_bio"]) != len(sample["tokens"]):
        raise ValueError("Inference checkpoint has incomplete word coverage")
    if record.get("pred_bio") != project_multinerd_bio(record["raw_pred_bio"]):
        raise ValueError("Inference checkpoint prediction projection mismatch")
    latency = record.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not 0 <= latency < float("inf"):
        raise ValueError("Inference checkpoint has invalid latency")


def read_checkpoint(path: Path, contract: dict, samples: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or json.loads(lines[0]) != {"contract": contract, "contract_hash": canonical_hash(contract)}:
        raise ValueError(f"{path}: inference contract mismatch")
    records = []
    for i, line in enumerate(lines[1:]):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 2 and not line.endswith(b"\n"):
                break  # An interrupted final append is the only accepted corruption.
            raise ValueError(f"{path}: corrupt checkpoint record {i}")
        if i >= len(samples):
            raise ValueError(f"{path}: too many checkpoint records")
        validate_record(record, samples[i])
        records.append(record)
    return records


def domain_metrics(records: list[dict], baseline: dict, contract: dict) -> dict:
    true, pred = [r["gold_bio"] for r in records], [r["pred_bio"] for r in records]
    quality = compute_ner_metrics(true, pred)
    latencies = sorted(r["latency_ms"] for r in records)
    import numpy as np
    diagnostics = Counter()
    statuses = Counter()
    for record in records:
        diagnostics.update(record.get("parse_diagnostics", {}))
        if "parse_status" in record:
            statuses[record["parse_status"]] += 1
    return {
        "contract_hash": canonical_hash(contract), "contract": contract,
        "sample_count": len(records), "complete": contract["smoke_limit"] is None and len(records) == contract["expected_samples"],
        **quality, "per_entity": per_entity_metrics(true, pred),
        "multinerd_baseline": baseline, "delta_f1": quality["f1"] - baseline["f1"],
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "parse_status_counts": dict(statuses), "parser_diagnostics": dict(diagnostics),
        "parse_failure_rate": statuses["failed"] / len(records) if statuses else None,
    }


@contextmanager
def run_lock(directory: Path, *, resume_failed: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".crossner.lock"
    payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID")}
    if path.exists() and resume_failed:
        owner = json.loads(path.read_text())
        if owner.get("job_id"):
            check = subprocess.run(["squeue", "-h", "-j", str(owner["job_id"]), "-o", "%i"], capture_output=True, text=True, timeout=20)
            if check.returncode or check.stdout.strip():
                raise RuntimeError(f"Cannot recover a live or unverified lock: {owner}")
        elif owner.get("hostname") == socket.gethostname():
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"CrossNER process is still alive: {owner}")
        else:
            raise RuntimeError(f"Cannot verify remote non-SLURM lock: {owner}")
        path.unlink()
    try:
        with path.open("x") as handle:
            json.dump(payload, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"CrossNER run is locked: {path}") from exc
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()) == payload:
            path.unlink()


class FrozenPredictor:
    """One model load for all domains; match the original inference settings."""

    def __init__(self, source: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig, set_seed
        self.torch, self.source = torch, source
        cfg = source["config"]
        set_seed(int(cfg["seed"]))
        if not torch.cuda.is_available():
            raise RuntimeError("Run CrossNER inference on a CUDA compute node")
        self.encoder = cfg["model_type"] == "encoder"
        self.tokenizer = AutoTokenizer.from_pretrained(str(source["checkpoint"]), use_fast=True, local_files_only=True,
                                                       **({"add_prefix_space": True} if self.encoder else {"trust_remote_code": True}))
        if self.encoder:
            self.device = torch.device("cuda")
            self.model = AutoModelForTokenClassification.from_pretrained(str(source["checkpoint"]), local_files_only=True).to(self.device)
        else:
            from peft import PeftModel
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], revision=MODEL_REVISIONS[cfg["model_name"]],
                                                      quantization_config=quantization, attn_implementation="sdpa",
                                                      device_map="auto", trust_remote_code=True, local_files_only=True)
            self.model = PeftModel.from_pretrained(base, str(source["checkpoint"]), is_trainable=False)
            self.device = next(self.model.parameters()).device
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        self.model.eval()
        torch.cuda.reset_peak_memory_stats()

    def predict(self, sample: dict) -> dict:
        torch, tokenizer = self.torch, self.tokenizer
        tokens = sample["tokens"]
        if self.encoder:
            inputs = tokenizer(tokens, is_split_into_words=True, truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > self.source["config"].get("max_length", 256):
                raise ValueError(f"{sample['sample_id']}: exceeds frozen encoder input length")
            word_ids = inputs.word_ids()
            positions = first_subword_positions(word_ids, len(tokens))
            inputs = inputs.to(self.device)
        else:
            from src.data.preprocess_decoder import build_system_prompt, format_numbered_tokens
            from src.decoder.generation import prepare_generation_inputs
            messages = [{"role": "system", "content": build_system_prompt(MULTINERD_INFO.entity_types)},
                        {"role": "user", "content": format_numbered_tokens(tokens)}]
            inputs = prepare_generation_inputs(tokenizer, messages, self.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            if self.encoder:
                result = self.model(**inputs).logits[0]
            else:
                result = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None,
                                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        extra = {}
        if self.encoder:
            result = result.argmax(-1).cpu().tolist()
            raw_bio = [MULTINERD_INFO.id2label[int(result[pos])] for pos in positions]
        else:
            from src.decoder.parse_output import entities_to_bio, parse_llm_output_with_diagnostics
            raw = tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            entities, status, diagnostics = parse_llm_output_with_diagnostics(raw, tokens, frozenset(MULTINERD_INFO.entity_types))
            raw_bio = entities_to_bio(tokens, entities)
            extra = {"raw_output": raw, "pred_entities": entities, "parse_status": status, "parse_diagnostics": diagnostics}
        return {**sample, "raw_pred_bio": raw_bio, "pred_bio": project_multinerd_bio(raw_bio), "latency_ms": latency, **extra}


def first_subword_positions(word_ids: list, count: int) -> list[int]:
    positions = {}
    for position, word in enumerate(word_ids):
        if word is not None:
            positions.setdefault(word, position)
    if sorted(positions) != list(range(count)):
        raise ValueError("Encoder tokenization does not cover every original word")
    return [positions[i] for i in range(count)]


def completed_domain(directory: Path, contract: dict, samples: list[dict]) -> bool:
    metrics_path = directory / "inference_metrics.yaml"
    predictions_path = directory / "test_predictions.json"
    if not metrics_path.exists():
        return False
    metrics = yaml.safe_load(metrics_path.read_text())
    if metrics.get("contract_hash") != canonical_hash(contract) or metrics.get("contract") != contract:
        raise ValueError(f"{directory}: existing evaluation contract differs")
    if not predictions_path.exists() or metrics.get("predictions_sha256") != file_hash(predictions_path):
        raise ValueError(f"{directory}: completed predictions missing or altered")
    records = json.loads(predictions_path.read_text())
    if len(records) != len(samples):
        raise ValueError(f"{directory}: incomplete saved predictions")
    for record, sample in zip(records, samples):
        validate_record(record, sample)
    return True


def run_evaluation(run, source_root: Path, output_root: Path, cache_dir: Path, domains=DOMAINS,
                   *, smoke_limit: int | None = None, resume_failed: bool = False) -> None:
    from src.run_metadata import collect_run_metadata
    ensure_safe_output(output_root, source_root)
    source = source_run(run, source_root)
    directory = output_root / source["config"]["experiment_name"]
    all_samples = {domain: load_crossner_test(domain, cache_dir) for domain in domains}
    if smoke_limit is not None:
        all_samples = {domain: rows[:smoke_limit] for domain, rows in all_samples.items()}
    predictor = None
    with run_lock(directory, resume_failed=resume_failed):
        baseline = baseline_metrics(source)
        atomic_write_yaml(directory / "multinerd_baseline.yaml", baseline)
        atomic_write_json(directory / "status.json", {"status": "RUNNING", "domains": list(domains), "job_id": os.environ.get("SLURM_JOB_ID")})
        try:
            for domain, samples in all_samples.items():
                target = directory / domain
                target.mkdir(parents=True, exist_ok=True)
                contract = evaluation_contract(source, domain, smoke_limit)
                if completed_domain(target, contract, samples):
                    print(f"SKIP_COMPLETED {run.group_key} seed={run.seed} domain={domain}", flush=True)
                    continue
                checkpoint = target / "inference_checkpoint.jsonl"
                records = read_checkpoint(checkpoint, contract, samples)
                # Rewrite only the verified prefix to discard an interrupted append.
                atomic_write_json(target / "contract.json", contract)
                temporary = checkpoint.with_suffix(".tmp")
                with temporary.open("w") as handle:
                    handle.write(json.dumps({"contract": contract, "contract_hash": canonical_hash(contract)}) + "\n")
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, checkpoint)
                if len(records) < len(samples) and predictor is None:
                    predictor = FrozenPredictor(source)
                    predictor.predict(samples[0])  # Same unmeasured warmup as the original inference.
                with checkpoint.open("a") as handle:
                    for sample in samples[len(records):]:
                        record = predictor.predict(sample)
                        validate_record(record, sample)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(record)
                        if len(records) % 25 == 0 or len(records) == len(samples):
                            print(f"CROSSNER_PROGRESS {run.group_key} seed={run.seed} domain={domain} {len(records)}/{len(samples)}", flush=True)
                predictions = target / "test_predictions.json"
                atomic_write_json(predictions, records)
                metrics = domain_metrics(records, baseline, contract)
                metrics["predictions_sha256"] = file_hash(predictions)
                metrics["run_metadata"] = collect_run_metadata(source["config"])
                if predictor is not None:
                    metrics["vram_peak_mb"] = predictor.torch.cuda.max_memory_allocated() / 1024**2
                atomic_write_yaml(target / "inference_metrics.yaml", metrics)
                checkpoint.unlink()
                print(f"CROSSNER_RESULT {domain} seed={run.seed} f1={metrics['f1']:.6f} delta={metrics['delta_f1']:.6f}", flush=True)
            atomic_write_json(directory / "status.json", {"status": "COMPLETED", "domains": list(domains), "smoke": smoke_limit is not None})
        except BaseException as exc:
            atomic_write_json(directory / "status.json", {"status": "FAILED", "domains": list(domains), "error": str(exc), "job_id": os.environ.get("SLURM_JOB_ID")})
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--resume-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_limit is not None and args.smoke_limit < 1:
        parser.error("--smoke-limit must be positive")
    runs = [r for r in selected_runs() if (PROJECT_ROOT / r.config_path).resolve() == Path(args.config).resolve()]
    if len(runs) != 1:
        parser.error("--config must identify one of the six selected canonical runs")
    run_evaluation(runs[0], args.source_results_root, args.output_root, args.cache_dir, tuple(dict.fromkeys(args.domains)),
                   smoke_limit=args.smoke_limit, resume_failed=args.resume_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
