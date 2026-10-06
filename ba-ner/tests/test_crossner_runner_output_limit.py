import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.evaluate import crossner
from src.seed_provenance import canonical_hash
from src.data.crossner_loader import TEST_SIZES


def runner():
    spec = importlib.util.spec_from_file_location(
        "run_crossner_output_limit_test", crossner.PROJECT_ROOT / "scripts/run_crossner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def high_args(module, tmp_path, *extra):
    exclusions = tmp_path / "exclusions.json"
    if not exclusions.exists():
        exclusions.write_text("fixture")
    return module.parse_args([
        "--submit", "--max-new-tokens", "1024", "--source-results-root", str(tmp_path / "source"),
        "--exclusions", str(exclusions), *extra])


def test_original_commands_do_not_import_extension(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, "output_limit_module", lambda: pytest.fail("Unexpected extension import"))
    args = module.parse_args(["--dry-run", "--source-results-root", str(tmp_path / "source")])
    assert args.max_new_tokens == 256
    assert args.model == list(crossner.GROUPS)
    assert args.output_root == tmp_path / "source/crossner"
    runs = crossner.selected_runs(args.model, args.seeds)
    assert len(runs) == 6
    for run in runs:
        command = module.command_for(run, args)
        worker = command.index("scripts/cluster/job_crossner.sh")
        assert command[worker + 1] == "--config"
        assert "--max-new-tokens" not in command
        assert "--baseline-output-root" not in command
        assert "--exclusions" not in command
        assert "--output-limit" not in command


def test_high_limit_defaults_and_dispatch(tmp_path):
    module = runner()
    args = high_args(module, tmp_path)
    assert args.model == ["qwen35-27b-qlora-3ep"]
    assert args.seeds == [42, 123, 456]
    assert args.domains == list(crossner.DOMAINS)
    assert args.output_root == tmp_path / "source/crossner_output_limit_1024"
    assert args.baseline_output_root == tmp_path / "source/crossner"
    runs = crossner.selected_runs(args.model, args.seeds)
    assert len(runs) == 3
    assert len(runs) * len(args.domains) == 15
    for run in runs:
        command = module.command_for(run, args)
        worker = command.index("scripts/cluster/job_crossner.sh")
        assert command[worker + 1:worker + 3] == ["--output-limit", "--config"]
        assert f"--job-name=crossner-qwen27b-s{run.seed}-nt1024" in command
        assert "--partition=H100" in command
        assert "--time=1-00:00:00" in command
        assert command[command.index("--max-new-tokens") + 1] == "1024"
        assert command[command.index("--baseline-output-root") + 1] == str(args.baseline_output_root)
        assert command[command.index("--exclusions") + 1] == str(args.exclusions)


@pytest.mark.parametrize("limit", ["0", "-1", "255", "12.5"])
def test_invalid_limit_rejected(tmp_path, limit):
    with pytest.raises(SystemExit):
        runner().parse_args(["--dry-run", "--max-new-tokens", limit,
                             "--source-results-root", str(tmp_path / "source")])


def test_encoder_and_mixed_selection_rejected(tmp_path):
    module = runner()
    for models in [["deberta-v3-large"], list(crossner.GROUPS)]:
        with pytest.raises(SystemExit):
            high_args(module, tmp_path, "--model", *models)


@pytest.mark.parametrize("target", ["crossner", "crossner/new", "multinerd/new", "."])
def test_high_output_cannot_overlap_originals(tmp_path, target):
    module = runner()
    with pytest.raises(ValueError, match="overlap"):
        high_args(module, tmp_path, "--output-root", str(tmp_path / "source" / target))


def test_high_smoke_is_isolated(tmp_path):
    module = runner()
    args = high_args(module, tmp_path, "--seeds", "42", "--domains", "music", "politics", "--smoke-limit", "25")
    assert args.output_root == tmp_path / "source/crossner_output_limit_1024/smoke"
    assert len(crossner.selected_runs(args.model, args.seeds)) == 1
    command = module.command_for(crossner.selected_runs(args.model, args.seeds)[0], args)
    assert "--job-name=crossner-qwen27b-s42-nt1024-smoke" in command
    with pytest.raises(SystemExit, match="2"):
        high_args(module, tmp_path, "--smoke-limit", "25", "--output-root",
                  str(tmp_path / "source/crossner_output_limit_1024"))


def prepare_submission(module, tmp_path, monkeypatch, *extra):
    args = high_args(module, tmp_path, *extra)
    extension = module.output_limit_module()
    calls = []
    monkeypatch.setattr(module, "preflight", lambda *a: {})
    monkeypatch.setattr(module, "frozen_commit", lambda: "test-commit")
    monkeypatch.setattr(module, "source_run", lambda run, root: {
        "config": {"experiment_name": f"qwen-s{run.seed}"}, "identity": {"seed": run.seed}})
    monkeypatch.setattr(module, "load_crossner_test", lambda domain, cache: [{"sample_id": domain}])
    monkeypatch.setattr(extension, "output_limit_contract", lambda source, domain, smoke_limit=None, *, max_new_tokens: {
        "seed": source["identity"]["seed"], "domain": domain, "limit": max_new_tokens, "smoke": smoke_limit})
    monkeypatch.setattr(extension, "completed_output_limit_domain", lambda *a: False)
    monkeypatch.setattr(module, "_active_job", lambda job: False)

    def submit_command(command, **kwargs):
        assert command[0] == "sbatch"
        calls.append(command)
        return str(1000 + len(calls))

    monkeypatch.setattr(module.subprocess, "check_output", submit_command)
    return args, extension, calls


def test_submit_schedules_three_jobs_and_dependent_aggregate(tmp_path, monkeypatch):
    module = runner()
    args, extension, calls = prepare_submission(module, tmp_path, monkeypatch)
    ledger = module.submit(crossner.selected_runs(args.model, args.seeds), args)
    assert len(calls) == 4
    assert len(ledger["jobs"]) == 3
    assert ledger["identity"]["max_new_tokens"] == 1024
    for entry in ledger["jobs"]:
        assert entry["identity"] == ledger["identity"]
        assert entry["max_new_tokens"] == 1024
        assert set(entry["contract_hashes"]) == set(args.domains)
    aggregate = calls[-1]
    assert "--dependency=afterany:1001:1002:1003" in aggregate
    assert "--job-name=crossner-aggregate-nt1024" in aggregate
    worker = aggregate.index("scripts/cluster/job_crossner.sh")
    assert aggregate[worker + 1:worker + 3] == ["--output-limit", "--aggregate"]
    assert aggregate[aggregate.index("--max-new-tokens") + 1] == "1024"
    assert aggregate[aggregate.index("--models") + 1] == "qwen35-27b-qlora-3ep"
    assert ledger["aggregations"][0]["identity"] == ledger["identity"]
    assert json.loads((args.output_root / "submissions.json").read_text()) == ledger


def test_high_preflight_verifies_references_and_exclusion_counts(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    module = runner()
    args = high_args(module, tmp_path)
    extension = module.output_limit_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
    (snapshot / "weights.safetensors").write_text("fixture")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))
    monkeypatch.setattr(module, "source_run", lambda run, root: {"identity": {"model_revision": "fixed", "seed": run.seed}})
    monkeypatch.setattr(module, "load_crossner_test", lambda domain, cache, **kwargs: [{}] * TEST_SIZES[domain])
    calls = []

    def references(*arguments):
        calls.append(("references", arguments))
        return {(seed, domain): {"baseline_256": {}, "encoder_baseline": {}}
                for seed in args.seeds for domain in args.domains}

    def exclusions(*arguments):
        calls.append(("exclusions", arguments))
        return {"excluded_sample_ids": [str(i) for i in range(21)]}

    monkeypatch.setattr(extension, "validate_references", references)
    monkeypatch.setattr(extension, "load_exclusions", exclusions)
    report = module.preflight(crossner.selected_runs(args.model, args.seeds), args)
    assert calls == [
        ("references", (args.source_results_root, args.baseline_output_root, args.cache_dir, args.seeds, args.domains)),
        ("exclusions", (args.exclusions, args.cache_dir)),
    ]
    assert report["inference_jobs"] == 3
    assert report["evaluations"] == 15
    assert report["reference_evaluations"] == 30
    assert report["excluded_samples"] == 21
    assert report["deduplicated_samples"] == 2485
    assert "excluded_samples=21 deduplicated_samples=2485" in capsys.readouterr().out


def test_resume_and_running_skip_stay_in_same_contract(tmp_path, monkeypatch):
    module = runner()
    args, extension, calls = prepare_submission(module, tmp_path, monkeypatch, "--seeds", "42", "--smoke-limit", "25", "--resume-failed")
    runs = crossner.selected_runs(args.model, args.seeds)
    module.submit(runs, args)
    assert "--resume-failed" in calls[0]
    monkeypatch.setattr(module, "_active_job", lambda job: True)
    module.submit(runs, args)
    assert len(calls) == 1
    monkeypatch.setattr(extension, "completed_output_limit_domain", lambda *a: True)
    module.submit(runs, args)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["limit", "original_limit", "exclusions", "job_identity", "contract"])
def test_mismatch_rejected_before_skipping_completed_or_live_job(tmp_path, monkeypatch, change):
    module = runner()
    args, extension, calls = prepare_submission(module, tmp_path, monkeypatch, "--seeds", "42", "--smoke-limit", "25")
    runs = crossner.selected_runs(args.model, args.seeds)
    module.submit(runs, args)
    ledger_path = args.output_root / "submissions.json"
    if change == "limit":
        args.max_new_tokens = 2048
    elif change == "original_limit":
        args.max_new_tokens = 256
    elif change == "exclusions":
        args.exclusions.write_text("changed fixture")
    else:
        ledger = json.loads(ledger_path.read_text())
        if change == "job_identity":
            ledger["jobs"][0]["identity"]["max_new_tokens"] = 2048
        else:
            ledger["jobs"][0]["contract_hashes"]["ai"] = canonical_hash({"different": True})
        ledger_path.write_text(json.dumps(ledger))
    monkeypatch.setattr(module, "_active_job", lambda job: pytest.fail("Must validate before querying live jobs"))
    monkeypatch.setattr(extension, "completed_output_limit_domain", lambda *a: pytest.fail("Must validate before skipping completed jobs"))
    with pytest.raises(ValueError, match="differs"):
        module.submit(runs, args)
    assert len(calls) == 1


def test_high_status_reads_high_root(tmp_path, monkeypatch, capsys):
    module = runner()
    args = high_args(module, tmp_path, "--seeds", "42")
    args.output_root.mkdir(parents=True)
    identity = module.output_limit_identity(args)
    (args.output_root / "submissions.json").write_text(json.dumps({
        "identity": identity, "jobs": [{"identity": identity, "job_id": "42", "experiment": "qwen"}]}))
    monkeypatch.setattr(module, "_active_job", lambda job: True)
    assert module.main(["--status", "--max-new-tokens", "1024", "--source-results-root", str(args.source_results_root),
                        "--exclusions", str(args.exclusions)]) == 0
    assert "42 qwen ACTIVE" in capsys.readouterr().out


def test_high_aggregate_routes_comparison_and_prints_counts(tmp_path, monkeypatch, capsys):
    module = runner()
    args = high_args(module, tmp_path)
    calls = []

    def aggregate(*positional, **kwargs):
        calls.append((positional, kwargs))
        return {"complete": True, "selection_complete": True, "successful_evaluations": 15,
                "expected_evaluations": 15, "max_new_tokens": 1024, "records": ["large report omitted"]}

    monkeypatch.setattr(module.output_limit_module(), "aggregate_output_limit", aggregate)
    assert module.main(["--aggregate-only", "--max-new-tokens", "1024", "--source-results-root", str(args.source_results_root),
                        "--exclusions", str(args.exclusions)]) == 0
    positional, kwargs = calls[0]
    assert positional[3:] == (args.model, args.seeds, args.domains)
    assert kwargs == {"max_new_tokens": 1024, "baseline_root": args.baseline_output_root, "exclusions_path": args.exclusions}
    output = json.loads(capsys.readouterr().out)
    assert output["successful_evaluations"] == 15
    assert "records" not in output


@pytest.mark.parametrize("arguments,module_name", [
    (["--config", "fixture.yaml"], "src.evaluate.crossner"),
    (["--aggregate"], "src.evaluate.crossner_aggregate"),
    (["--output-limit", "--config", "fixture.yaml"], "src.evaluate.crossner_output_limit"),
    (["--output-limit", "--aggregate"], "src.evaluate.crossner_output_limit"),
])
def test_job_wrapper_dispatches_original_and_added_modes(tmp_path, arguments, module_name):
    scripts = tmp_path / "scripts/cluster"
    scripts.mkdir(parents=True)
    (scripts / "common.sh").write_text(
        "activate_ba_ner_env() { :; }\nprint_runtime_info() { :; }\n"
        "python() { printf '%s\\n' \"$@\"; }\n")
    wrapper = crossner.PROJECT_ROOT / "scripts/cluster/job_crossner.sh"
    result = subprocess.run(["bash", str(wrapper), *arguments], capture_output=True, text=True,
                            env={**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path)}, check=True)
    assert result.stdout.splitlines()[:2] == ["-m", module_name]
    assert "--output-limit" not in result.stdout.splitlines()
    if arguments[0] == "--output-limit":
        assert result.stdout.splitlines()[2:] == arguments[1:]
