#!/usr/bin/env python3
"""Prepare and submit the six inference-only CrossNER checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crossner_loader import DOMAINS, load_crossner_test
from src.evaluate.crossner import (
    GROUPS, SEEDS, completed_domain, ensure_safe_output, evaluation_contract,
    file_hash, first_subword_positions, selected_runs, source_run,
)
from src.evaluate.crossner_aggregate import aggregate
from src.seed_provenance import canonical_hash
from src.seed_study import atomic_write_json


def output_limit_module():
    """Keep the original evaluation path independent of the extension."""
    from src.evaluate import crossner_output_limit
    return crossner_output_limit


def is_output_limit(args) -> bool:
    return getattr(args, "max_new_tokens", 256) > 256


def output_limit_arguments(args) -> list[str]:
    if not is_output_limit(args):
        return []
    return ["--max-new-tokens", str(args.max_new_tokens),
            "--baseline-output-root", str(args.baseline_output_root),
            "--exclusions", str(args.exclusions)]


def output_limit_identity(args) -> dict:
    return {"max_new_tokens": args.max_new_tokens,
            "source_results_root": str(args.source_results_root),
            "baseline_output_root": str(args.baseline_output_root),
            "output_root": str(args.output_root),
            "exclusions": str(args.exclusions),
            "exclusions_sha256": file_hash(args.exclusions),
            "smoke_limit": args.smoke_limit}


def verify_output_limit_ledger(ledger, args) -> dict:
    identity = output_limit_identity(args)
    if ledger.get("identity") != identity:
        raise ValueError("Existing output-limit submission identity differs; use a separate output root")
    for entry in ledger.get("jobs", []) + ledger.get("aggregations", []):
        if entry.get("identity") != identity:
            raise ValueError("Existing output-limit job identity differs; refusing to reuse the submission ledger")
    return identity


def resources_for(run) -> dict:
    resources = dict(run.resources["inference"])
    # The reference partition was retired. H100 now permits at most one day.
    if resources["partition"] == "H100-Trails":
        resources.update(partition="H100", time="1-00:00:00")
    return resources


def command_for(run, args, *, commit: str = "<committed HEAD>") -> list[str]:
    resources = resources_for(run)
    name = "crossner-" + ("deb-large" if run.model_family == "encoder" else "qwen27b") + f"-s{run.seed}"
    if is_output_limit(args):
        name += f"-nt{args.max_new_tokens}"
    if args.smoke_limit:
        name += "-smoke"
    exports = {
        "BA_NER_EXPECTED_GIT_COMMIT": commit,
        "BA_NER_VENV": os.environ.get("BA_NER_VENV", str(PROJECT_ROOT / ".venv")),
        "HF_HOME": os.environ.get("HF_HOME", str(args.source_results_root.parent / "hf-cache")),
        "BA_NER_RESULTS_ROOT": str(args.source_results_root),
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }
    if any("," in value for value in exports.values()):
        raise ValueError("SLURM export paths cannot contain commas")
    command = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={resources['partition']}",
               f"--gres={resources['gres']}", f"--mem={resources['memory']}", f"--time={resources['time']}",
               "--cpus-per-task=4", f"--chdir={PROJECT_ROOT}",
               f"--output={args.output_root}/logs/{name}-%j.out", f"--error={args.output_root}/logs/{name}-%j.err",
               "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()),
               "scripts/cluster/job_crossner.sh", *(["--output-limit"] if is_output_limit(args) else []), "--config", run.config_path,
               "--source-results-root", str(args.source_results_root), "--output-root", str(args.output_root),
               "--cache-dir", str(args.cache_dir), "--domains", *args.domains,
               *output_limit_arguments(args)]
    if args.smoke_limit:
        command += ["--smoke-limit", str(args.smoke_limit)]
    if args.resume_failed:
        command.append("--resume-failed")
    return command


def preflight(runs, args) -> dict:
    """Read frozen weights and tokenize all target words; do not load models."""
    from transformers import AutoTokenizer
    ensure_safe_output(args.output_root, args.source_results_root)
    datasets = {domain: load_crossner_test(domain, args.cache_dir, download=True) for domain in args.domains}
    report = {"sources": [], "domains": {d: len(rows) for d, rows in datasets.items()}}
    for run in runs:
        source = source_run(run, args.source_results_root)
        report["sources"].append(source["identity"])
        if run.model_family == "encoder":
            tokenizer = AutoTokenizer.from_pretrained(str(source["checkpoint"]), add_prefix_space=True, use_fast=True, local_files_only=True)
            for domain, rows in datasets.items():
                inputs = tokenizer([r["tokens"] for r in rows], is_split_into_words=True, truncation=False)
                for i, row in enumerate(rows):
                    if len(inputs["input_ids"][i]) > source["config"].get("max_length", 256):
                        raise ValueError(f"{row['sample_id']}: exceeds frozen encoder input length")
                    first_subword_positions(inputs.word_ids(batch_index=i), len(row["tokens"]))
        else:
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(run.model_name, revision=source["identity"]["model_revision"],
                                             cache_dir=os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(args.source_results_root.parent / "hf-cache"))) / "hub")),
                                             allow_patterns=["config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json", "*.py"],
                                             local_files_only=True))
            index = json.loads((snapshot / "model.safetensors.index.json").read_text())
            if not (snapshot / "config.json").is_file() or any(not (snapshot / shard).is_file() for shard in set(index["weight_map"].values())):
                raise ValueError(f"{snapshot}: missing base-model weight shards")
    report["inference_jobs"] = len(runs)
    report["evaluations"] = len(runs) * len(args.domains)
    if is_output_limit(args):
        extension = output_limit_module()
        references = extension.validate_references(args.source_results_root, args.baseline_output_root,
                                                   args.cache_dir, args.seeds, args.domains)
        exclusions = extension.load_exclusions(args.exclusions, args.cache_dir)
        report["reference_evaluations"] = sum(len(pair) for pair in references.values())
        report["max_new_tokens"] = args.max_new_tokens
        report["exclusions"] = str(args.exclusions)
        report["excluded_samples"] = len(exclusions["excluded_sample_ids"])
        report["deduplicated_samples"] = sum(len(load_crossner_test(d, args.cache_dir)) for d in DOMAINS) - report["excluded_samples"]
        print(f"PREFLIGHT_OUTPUT_LIMIT max_new_tokens={args.max_new_tokens} references={report['reference_evaluations']} "
              f"excluded_samples={report['excluded_samples']} deduplicated_samples={report['deduplicated_samples']}", flush=True)
    print(f"PREFLIGHT_OK jobs={len(runs)} evaluations={report['evaluations']} source_checkpoints=verified", flush=True)
    return report


def frozen_commit() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=PROJECT_ROOT, text=True)
    if status.strip():
        raise RuntimeError("Commit the tested CrossNER implementation and keep the repository clean before submission")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def _active_job(job_id: str) -> bool:
    result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%i"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"Cannot verify SLURM job {job_id}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def submit(runs, args) -> dict:
    preflight(runs, args)
    commit = frozen_commit()
    ensure_safe_output(args.output_root, args.source_results_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    submissions = args.output_root / "submissions.json"
    ledger = json.loads(submissions.read_text()) if submissions.exists() else {"jobs": []}
    extension = output_limit_module() if is_output_limit(args) else None
    identity = None
    if extension is not None:
        if not submissions.exists():
            ledger["identity"] = output_limit_identity(args)
        identity = verify_output_limit_ledger(ledger, args)
    elif "identity" in ledger:
        raise ValueError("Existing output-limit submission identity differs from the original 256-token evaluation")
    jobs = []
    for run in runs:
        source = source_run(run, args.source_results_root)
        experiment = source["config"]["experiment_name"]
        directory = args.output_root / experiment
        samples = {domain: load_crossner_test(domain, args.cache_dir) for domain in args.domains}
        if args.smoke_limit:
            samples = {d: rows[:args.smoke_limit] for d, rows in samples.items()}
        if extension is None:
            contracts = {d: evaluation_contract(source, d, args.smoke_limit) for d in samples}
            completed = completed_domain
        else:
            contracts = {d: extension.output_limit_contract(source, d, args.smoke_limit,
                                                            max_new_tokens=args.max_new_tokens) for d in samples}
            completed = extension.completed_output_limit_domain
        previous = [j for j in ledger["jobs"] if j.get("experiment") == experiment]
        if extension is not None:
            contract_hashes = {d: canonical_hash(contract) for d, contract in contracts.items()}
            for entry in previous:
                if (entry.get("domains") != args.domains or entry.get("contract_hashes") != contract_hashes):
                    raise ValueError(f"{experiment}: existing output-limit selection or contract differs")
        if all(completed(directory / d, contracts[d], rows) for d, rows in samples.items()):
            print(f"SKIP_COMPLETED {experiment}")
            continue
        live = [j for j in previous if _active_job(str(j["job_id"]))]
        if live:
            print(f"SKIP_RUNNING {experiment} job={live[-1]['job_id']}")
            jobs.append(live[-1]["job_id"])
            continue
        status_file = directory / "status.json"
        if (directory / ".crossner.lock").exists() or (status_file.exists() and json.loads(status_file.read_text()).get("status") in ("FAILED", "RUNNING")):
            if not args.resume_failed:
                raise RuntimeError(f"{experiment}: use --resume-failed for this interrupted evaluation")
        command = command_for(run, args, commit=commit)
        job_id = subprocess.check_output(command, cwd=PROJECT_ROOT, text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected SLURM submission response: {job_id}")
        entry = {"job_id": job_id, "experiment": experiment, "model": run.group_key, "seed": run.seed,
                 "domains": args.domains, "git_commit": commit, "command": command, "smoke_limit": args.smoke_limit}
        if extension is not None:
            entry.update(identity=identity, max_new_tokens=args.max_new_tokens, contract_hashes=contract_hashes)
        ledger["jobs"].append(entry)
        atomic_write_json(submissions, ledger)
        jobs.append(job_id)
        print(f"SUBMITTED {experiment} job={job_id}", flush=True)
    if jobs and args.smoke_limit is None:
        exports = "ALL,BA_NER_EXPECTED_GIT_COMMIT=" + commit + ",BA_NER_VENV=" + os.environ.get("BA_NER_VENV", str(PROJECT_ROOT / ".venv"))
        aggregate_name = "crossner-aggregate" + (f"-nt{args.max_new_tokens}" if extension is not None else "")
        command = ["sbatch", "--parsable", f"--job-name={aggregate_name}", "--partition=batch", "--mem=8G", "--time=00:30:00",
                   "--cpus-per-task=2", f"--chdir={PROJECT_ROOT}", "--dependency=afterany:" + ":".join(jobs),
                   f"--output={args.output_root}/logs/aggregate-%j.out", f"--error={args.output_root}/logs/aggregate-%j.err",
                   "--export=" + exports, "scripts/cluster/job_crossner.sh", *(["--output-limit"] if extension is not None else []), "--aggregate",
                   "--source-results-root", str(args.source_results_root), "--output-root", str(args.output_root),
                   "--cache-dir", str(args.cache_dir), "--models", *args.model, "--seeds", *map(str, args.seeds), "--domains", *args.domains,
                   *output_limit_arguments(args)]
        aggregate_id = subprocess.check_output(command, cwd=PROJECT_ROOT, text=True).strip().split(";")[0]
        aggregate_entry = {"job_id": aggregate_id, "dependencies": jobs, "git_commit": commit}
        if extension is not None:
            aggregate_entry.update(identity=identity, max_new_tokens=args.max_new_tokens)
        ledger.setdefault("aggregations", []).append(aggregate_entry)
        atomic_write_json(submissions, ledger)
        print(f"SUBMITTED_AGGREGATE job={aggregate_id}", flush=True)
    return ledger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "dry-run", "submit", "aggregate-only", "status"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--model", nargs="+", choices=GROUPS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    default_source = Path(os.environ.get("BA_NER_RESULTS_ROOT", str(Path(os.environ["BA_NER_SCRATCH"]) / "results") if os.environ.get("BA_NER_SCRATCH") else str(PROJECT_ROOT / "results")))
    parser.add_argument("--source-results-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".hf_cache" / "crossner")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--baseline-output-root", type=Path)
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args(argv)
    if args.smoke_limit is not None and args.smoke_limit < 1:
        parser.error("--smoke-limit must be positive")
    if args.max_new_tokens < 256:
        parser.error("--max-new-tokens must be at least 256")
    extension = output_limit_module() if is_output_limit(args) else None
    if args.model is None:
        args.model = [extension.QWEN_GROUP] if extension is not None else list(GROUPS)
    if extension is not None and set(args.model) != {extension.QWEN_GROUP}:
        parser.error("Higher output limits support only the frozen Qwen 27B group")
    args.source_results_root = args.source_results_root.resolve()
    args.baseline_output_root = (args.baseline_output_root or args.source_results_root / "crossner").resolve()
    if extension is not None:
        args.exclusions = (args.exclusions or extension.DEFAULT_EXCLUSIONS).resolve()
    default_output = args.source_results_root / (f"crossner_output_limit_{args.max_new_tokens}" if extension is not None else "crossner") / ("smoke" if args.smoke_limit else "")
    args.output_root = (args.output_root or default_output).resolve()
    args.cache_dir = args.cache_dir.resolve()
    for field in ("model", "seeds", "domains"):
        setattr(args, field, list(dict.fromkeys(getattr(args, field))))
    if args.smoke_limit is not None and args.output_root == (args.source_results_root / "crossner").resolve():
        parser.error("Smoke output must have a separate path")
    ensure_safe_output(args.output_root, args.source_results_root)
    if extension is not None:
        extension.ensure_output_limit_safe(args.output_root, args.source_results_root, args.baseline_output_root)
        extension.ensure_output_limit_safe(args.output_root, args.source_results_root, args.source_results_root / "crossner")
        if args.smoke_limit is not None and args.output_root == (args.source_results_root / f"crossner_output_limit_{args.max_new_tokens}").resolve():
            parser.error("Smoke output must have a separate path")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = selected_runs(args.model, args.seeds)
    if args.dry_run:
        for run in runs:
            print(shlex.join(command_for(run, args)))
        print(f"INFERENCE_ONLY jobs={len(runs)} evaluations={len(runs) * len(args.domains)} training_jobs=0")
    elif args.preflight:
        preflight(runs, args)
    elif args.submit:
        submit(runs, args)
    elif args.aggregate_only:
        if is_output_limit(args):
            report = output_limit_module().aggregate_output_limit(
                args.source_results_root, args.output_root, args.cache_dir, args.model, args.seeds, args.domains,
                max_new_tokens=args.max_new_tokens, baseline_root=args.baseline_output_root, exclusions_path=args.exclusions)
            print(json.dumps({key: report[key] for key in (
                "complete", "selection_complete", "successful_evaluations", "expected_evaluations", "max_new_tokens") if key in report}))
        else:
            report = aggregate(args.source_results_root, args.output_root, args.cache_dir, args.model, args.seeds, args.domains)
            print(json.dumps({k: v for k, v in report.items() if k not in ("summary", "missing_or_failed")}))
        return 0 if report["selection_complete"] else 1
    else:
        path = args.output_root / "submissions.json"
        ledger = json.loads(path.read_text()) if path.exists() else {"jobs": []}
        if is_output_limit(args) and path.exists():
            verify_output_limit_ledger(ledger, args)
        for job in ledger["jobs"]:
            print(f"{job['job_id']} {job['experiment']} {'ACTIVE' if _active_job(str(job['job_id'])) else 'NOT_IN_QUEUE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
