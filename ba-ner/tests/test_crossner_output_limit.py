import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from src.data import crossner_loader as loader
from src.data.dataset_loader import MULTINERD_INFO
from src.evaluate import crossner as original
from src.evaluate import crossner_aggregate as aggregation
from src.evaluate import crossner_output_limit as extension
from src.seed_provenance import scientific_contract
from src.seed_study import full_run_config_hash, scientific_config_hash


def _sample_rows(domain):
    return loader.parse_crossner("Ada\tB-person\nvisited\tO\nParis\tB-location\n\nBob\tB-person\n", domain)


def _record(sample, *, failed=False, extended=False):
    raw_bio = ["O"] * len(sample["tokens"]) if failed else sample["gold_bio"]
    record = {**sample, "raw_pred_bio": raw_bio, "pred_bio": loader.project_multinerd_bio(raw_bio),
              "latency_ms": 10.0}
    if failed or extended:
        record.update(raw_output="[" if failed else "[]", pred_entities=[],
                      parse_status="failed" if failed else "ok", parse_diagnostics={})
    if extended:
        record.update(generated_token_count=1024 if failed else 7,
                      token_limit_reached=failed, stop_reason="max_new_tokens" if failed else "eos",
                      json_complete=not failed)
    return record


def _create_source(run, root):
    cfg = original.load_experiment_config(original.PROJECT_ROOT / run.config_path)
    directory = root / Path(run.output_dir).relative_to("results")
    checkpoint = directory / ("best_model" if run.model_family == "encoder" else "best_lora_adapter")
    checkpoint.mkdir(parents=True)
    if run.model_family == "encoder":
        (checkpoint / "config.json").write_text(json.dumps({"id2label": MULTINERD_INFO.id2label}))
        files = ("model.safetensors", "tokenizer.json", "tokenizer_config.json")
    else:
        (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": run.model_name, "inference_mode": True}))
        files = ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for filename in files:
        (checkpoint / filename).write_text("fixture")
    identity = {
        "seed": run.seed, "canonical": True, "model_name": run.model_name,
        "scientific_config_hash": scientific_config_hash(cfg), "full_run_config_hash": full_run_config_hash(cfg),
        "checkpoint_used": str(checkpoint), "model_revision": original.MODEL_REVISIONS[run.model_name],
        "checkpoint_selection": scientific_contract(cfg)["evaluation"]["checkpoint_selection"],
    }
    metrics = dict(identity)
    if run.model_family == "decoder":
        metrics.update(max_new_tokens=256, inference_batch_size=1, precision_mode="qlora_4bit",
                       attn_impl="sdpa", thinking_enabled=False)
    for filename, content in (("run_metadata.yaml", identity), ("inference_metrics.yaml", metrics),
                              ("scientific_contract.yaml", scientific_contract(cfg)), ("config_resolved.yaml", cfg)):
        (directory / filename).write_text(yaml.safe_dump(content))
    (directory / "status.json").write_text(json.dumps({**identity, "status": "COMPLETED"}))
    (directory / "test_predictions.json").write_text("[]")
    return original.source_run(run, root)


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "source"
    baseline = root / "crossner"
    output = root / "crossner_output_limit_1024"
    datasets = {d: _sample_rows(d) for d in loader.DOMAINS}
    baseline_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    for module in (original, extension, aggregation):
        monkeypatch.setattr(module, "load_crossner_test", lambda domain, *a, **k: datasets[domain])
        monkeypatch.setattr(module, "baseline_metrics", lambda source: baseline_metrics)
    for domain in loader.DOMAINS:
        monkeypatch.setitem(original.TEST_SIZES, domain, 2)
    monkeypatch.setattr("src.run_metadata.collect_run_metadata", lambda *a: {"fixture": True})
    sources = {}
    for run in original.selected_runs():
        source = _create_source(run, root)
        sources[(run.group_key, run.seed)] = source
        for domain, samples in datasets.items():
            directory = baseline / source["config"]["experiment_name"] / domain
            directory.mkdir(parents=True)
            records = [_record(sample, failed=run.model_family == "decoder" and i == 0) for i, sample in enumerate(samples)]
            contract = original.evaluation_contract(source, domain)
            original.atomic_write_json(directory / "test_predictions.json", records)
            metrics = original.domain_metrics(records, baseline_metrics, contract)
            metrics["predictions_sha256"] = original.file_hash(directory / "test_predictions.json")
            original.atomic_write_yaml(directory / "inference_metrics.yaml", metrics)
    exclusions = tmp_path / "exclusions.yaml"
    exclusions.write_text(yaml.safe_dump({"version": 1, "dataset": "crossner", "revision": loader.CROSSNER_REVISION,
                                        "expected_excluded_samples": 1, "excluded_sample_ids": ["music:test:0"]}))
    return root, baseline, output, exclusions, sources, datasets


def _fake_predictor(monkeypatch, calls, *, fail_after=None, latencies=None):
    class Predictor:
        def __init__(self, source, *, max_new_tokens):
            assert max_new_tokens == 1024
            self.torch = type("Torch", (), {"cuda": type("Cuda", (), {"max_memory_allocated": staticmethod(lambda: 0)})})

        def predict(self, sample):
            calls.append(sample["sample_id"])
            if fail_after is not None and len(calls) > fail_after:
                raise RuntimeError("interrupted generation")
            record = _record(sample, extended=True)
            record["latency_ms"] = (latencies or {}).get(sample["sample_id"].split(":")[0], 10.0)
            return record
    monkeypatch.setattr(extension, "OutputLimitPredictor", Predictor)


def test_default_mask_matches_fixed_report():
    mask = extension.load_exclusions()
    assert mask["expected_excluded_samples"] == 21
    assert len(mask["excluded_sample_ids"]) == 21
    assert {domain: sum(i.startswith(domain + ":") for i in mask["excluded_sample_ids"]) for domain in loader.DOMAINS} == {
        "ai": 2, "literature": 0, "music": 4, "politics": 3, "science": 12}
    report_path = original.PROJECT_ROOT / mask["source_report"]
    if report_path.exists():
        source = json.loads(report_path.read_text())
        assert mask["excluded_sample_ids"] == sorted(source["matched_source_english_row_indices"])
        assert mask["source_report_sha256"] == original.file_hash(report_path)


def test_original_contract_and_source_settings_remain_frozen(study):
    root, baseline, output, exclusions, sources, datasets = study
    source = sources[(extension.QWEN_GROUP, 42)]
    before = original.evaluation_contract(source, "music")
    contract = extension.output_limit_contract(source, "music")
    assert original.evaluation_contract(source, "music") == before
    assert before["decoder"]["max_new_tokens"] == 256
    assert contract["decoder"]["max_new_tokens"] == 1024
    assert contract["code_sha256"]["src/evaluate/crossner.py"] == before["code_sha256"]["src/evaluate/crossner.py"]
    assert "src/evaluate/crossner_output_limit.py" in contract["code_sha256"]
    assert source["config"].get("max_new_tokens", 256) == 256
    assert yaml.safe_load((source["directory"] / "inference_metrics.yaml").read_text())["max_new_tokens"] == 256
    original.completed_domain(baseline / source["config"]["experiment_name"] / "music", before, datasets["music"])
    with pytest.raises(ValueError, match="decoder"):
        extension.output_limit_contract(sources[(original.GROUPS[0], 42)], "music")


@pytest.mark.parametrize("limit", [256, 0, -1, 1024.0, True])
def test_invalid_extension_limits_rejected(limit):
    with pytest.raises(ValueError, match="max_new_tokens"):
        extension.output_limit_contract({}, "music", max_new_tokens=limit)


@pytest.mark.parametrize("relative", ["crossner", "crossner/subdir", "."])
def test_new_output_cannot_overlap_baseline(tmp_path, relative):
    with pytest.raises(ValueError, match="overlap"):
        extension.ensure_output_limit_safe(tmp_path / relative, tmp_path, tmp_path / "crossner")


def test_actual_generation_diagnostics_and_eos_at_cap():
    diag = extension.generation_diagnostics([1] * 1023 + [9], "[]", 9, 1024)
    assert diag == {"generated_token_count": 1024, "token_limit_reached": True, "stop_reason": "eos", "json_complete": True}
    diag = extension.generation_diagnostics([1] * 1024, '[{"type":', 9, 1024)
    assert diag["stop_reason"] == "max_new_tokens"
    assert not diag["json_complete"]
    sample = _sample_rows("music")[0]
    record = _record(sample, extended=True)
    extension.validate_output_limit_record(record, sample, 1024)
    for key, bad in (("generated_token_count", 1025), ("token_limit_reached", True),
                     ("stop_reason", "max_new_tokens"), ("json_complete", False)):
        with pytest.raises(ValueError):
            extension.validate_output_limit_record({**record, key: bad}, sample, 1024)
    extension.validate_output_limit_record(_record(sample, extended=True, failed=True), sample, 1024)


def test_predictor_uses_actual_ids_and_keeps_generation_and_parser_settings(monkeypatch):
    import src.decoder.generation as generation

    calls = []
    inputs = {"input_ids": type("Input", (), {"shape": (1, 5)})()}
    new_ids = [1] * 299 + [9]
    class Result:
        def __getitem__(self, index):
            assert index == (0, slice(5, None))
            return type("Tokens", (), {"tolist": lambda self: new_ids})()
    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Result()
    class Tokenizer:
        eos_token_id = pad_token_id = 9
        def decode(self, ids, *, skip_special_tokens):
            assert ids == new_ids and skip_special_tokens
            return "["
    monkeypatch.setattr(generation, "prepare_generation_inputs", lambda *a: inputs)
    predictor = object.__new__(extension.OutputLimitPredictor)
    predictor.torch = type("Torch", (), {"cuda": type("Cuda", (), {"synchronize": staticmethod(lambda: None)}),
                                         "no_grad": staticmethod(nullcontext)})()
    predictor.tokenizer, predictor.model, predictor.device = Tokenizer(), Model(), "cpu"
    predictor.max_new_tokens = 1024
    sample = _sample_rows("music")[0]
    record = predictor.predict(sample)
    assert calls[0]["max_new_tokens"] == 1024
    assert calls[0]["do_sample"] is False
    assert calls[0]["temperature"] is calls[0]["top_p"] is None
    assert record["generated_token_count"] == 300
    assert record["stop_reason"] == "eos"
    assert record["parse_status"] == "failed"
    assert record["pred_bio"] == ["O"] * len(sample["tokens"])


def test_resume_recovers_prefix_without_recomputing_and_keeps_sources_read_only(study, monkeypatch):
    root, baseline, output, exclusions, sources, datasets = study
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    run = original.selected_runs((extension.QWEN_GROUP,), (42,))[0]
    calls = []
    _fake_predictor(monkeypatch, calls, fail_after=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    status = json.loads((output / sources[(extension.QWEN_GROUP, 42)]["config"]["experiment_name"] / "status.json").read_text())
    assert status["status"] == "FAILED"
    calls.clear()
    _fake_predictor(monkeypatch, calls)
    extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions, resume_failed=True)
    assert calls == ["music:test:0", "music:test:1"]  # warmup and only the unfinished sample
    source = sources[(extension.QWEN_GROUP, 42)]
    domain = output / source["config"]["experiment_name"] / "music"
    metrics = yaml.safe_load((domain / "inference_metrics.yaml").read_text())
    artifact = metrics["reference_artifacts"]["baseline_256"]
    assert Path(artifact["path"]).is_relative_to(baseline)
    assert original.file_hash(Path(artifact["path"])) == artifact["predictions_sha256"]
    assert metrics["reference_metrics"]["full"]["output_limit_1024"]["delta_f1_vs_256"] > 0
    assert metrics["reference_metrics"]["deduplicated"]["output_limit_1024"]["sample_count"] == 1
    assert metrics["delta_f1"] == 0  # still CrossNER minus MultiNERD
    status = json.loads((domain.parent / "status.json").read_text())
    assert status["completed_samples"] == status["total_samples"] == 2
    assert status["status"] == "COMPLETED" and status["seed"] == 42
    extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    assert calls == ["music:test:0", "music:test:1"]
    for path, contents in before.items():
        assert Path(path).read_bytes() == contents
    with pytest.raises(ValueError, match="contract"):
        extension.run_evaluation(run, root, output, root, ("music",), max_new_tokens=2048, exclusions_path=exclusions)


def test_full_three_arm_comparison_has_15_evaluations_and_fixed_dedup(study, monkeypatch):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [])
    for run in original.selected_runs((extension.QWEN_GROUP,)):
        extension.run_evaluation(run, root, output, root, exclusions_path=exclusions)
    report = extension.aggregate_output_limit(root, output, root, exclusions_path=exclusions)
    assert report["complete"] and report["selection_complete"]
    assert report["successful_evaluations"] == report["expected_evaluations"] == 15
    summary = report["summary"]
    extended = next(r for r in summary if r["population"] == "full" and r["domain"] == "pooled_micro" and r["evaluation"] == "output_limit_1024")
    assert extended["successful_seeds"] == 3
    assert extended["sample_count"] == 10
    assert extended["f1_mean"] == 1
    assert extended["delta_f1_vs_256_mean"] > 0
    filtered = next(r for r in summary if r["population"] == "deduplicated" and r["domain"] == "pooled_micro" and r["evaluation"] == "output_limit_1024")
    assert filtered["sample_count"] == 9
    baseline_row = next(r for r in summary if r["evaluation"] == "baseline_256")
    assert baseline_row["output_limit_rate_mean"] is None
    assert baseline_row["generated_tokens_mean"] is None
    assert (output / "aggregate/comparison.csv").is_file()
    primary = yaml.safe_load((output / "aggregate/summary.yaml").read_text())
    assert primary["complete"] and primary["primary_expected_evaluations"] == 15
    # A valid new cohort cannot hide a damaged old reference.
    old = baseline / sources[(original.GROUPS[0], 42)]["config"]["experiment_name"] / "music" / "test_predictions.json"
    old.write_text("[]")
    report = extension.aggregate_output_limit(root, output, root, exclusions_path=exclusions)
    assert not report["complete"] and not report["selection_complete"]
    assert report["missing_or_failed"]


def test_pairing_rejects_changed_sample_identity(study):
    root, baseline, output, exclusions, sources, datasets = study
    refs = extension.validate_references(root, baseline, root, (42,), ("music",))[(42, "music")]
    records = [_record(r, extended=True) for r in datasets["music"]]
    records[0] = {**records[0], "tokens": ["different"]}
    with pytest.raises(ValueError, match="Paired"):
        extension.reference_metrics(records, refs, set(), 42, 1024)


def test_corrupt_baseline_metadata_fails_worker_with_status_and_comparison_incomplete(study, monkeypatch):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [])
    run = original.selected_runs((extension.QWEN_GROUP,), (42,))[0]
    extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    old = baseline / sources[(original.GROUPS[0], 42)]["config"]["experiment_name"] / "music" / "inference_metrics.yaml"
    old.write_text("[]")
    report = extension.aggregate_output_limit(root, output, root, seeds=(42,), domains=("music",), exclusions_path=exclusions)
    assert not report["selection_complete"] and report["missing_or_failed"]
    with pytest.raises(AttributeError):
        extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    status_path = output / sources[(extension.QWEN_GROUP, 42)]["config"]["experiment_name"] / "status.json"
    assert json.loads(status_path.read_text())["status"] == "FAILED"


def test_original_root_protected_even_when_custom_baseline_is_given(tmp_path):
    with pytest.raises(ValueError, match="CrossNER"):
        extension.ensure_output_limit_safe(tmp_path / "crossner" / "new", tmp_path, tmp_path / "custom_reference")


@pytest.mark.parametrize("bad_memory", ["large", {"allocated": 1}, float("nan"), float("inf"), -1, True])
@pytest.mark.parametrize("evaluation", ["encoder_baseline", "output_limit_1024"])
def test_corrupt_memory_is_reported_as_missing_instead_of_crashing(study, monkeypatch, bad_memory, evaluation):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [])
    run = original.selected_runs((extension.QWEN_GROUP,), (42,))[0]
    extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    group = original.GROUPS[0] if evaluation == "encoder_baseline" else extension.QWEN_GROUP
    metrics_path = (baseline if evaluation == "encoder_baseline" else output) / sources[(group, 42)]["config"]["experiment_name"] / "music/inference_metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text())
    metrics["vram_peak_mb"] = bad_memory
    original.atomic_write_yaml(metrics_path, metrics)
    report = extension.aggregate_output_limit(root, output, root, seeds=(42,), domains=("music",), exclusions_path=exclusions)
    assert not report["complete"] and not report["selection_complete"]
    assert report["successful_evaluations"] == 0
    assert report["summary"] == []
    assert any("vram_peak_mb" in entry["reason"] for entry in report["missing_or_failed"])
    assert yaml.safe_load((output / "aggregate/comparison.yaml").read_text())["missing_or_failed"]


@pytest.mark.parametrize("metric", ["f1", "delta_f1"])
def test_rejected_treatment_metrics_are_excluded_from_paired_cohort(study, monkeypatch, metric):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [])
    for run in original.selected_runs((extension.QWEN_GROUP,), (42,)):
        extension.run_evaluation(run, root, output, root, ("music", "politics"), exclusions_path=exclusions)
    metrics_path = output / sources[(extension.QWEN_GROUP, 42)]["config"]["experiment_name"] / "music/inference_metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text())
    metrics[metric] -= 0.2
    original.atomic_write_yaml(metrics_path, metrics)
    report = extension.aggregate_output_limit(root, output, root, seeds=(42,), domains=("music", "politics"), exclusions_path=exclusions)
    assert not report["selection_complete"]
    assert report["successful_evaluations"] == 1
    assert report["expected_evaluations"] == 2
    assert {(entry["seed"], entry["domain"]) for entry in report["missing_or_failed"]} == {(42, "music")}
    assert {row["domain"] for row in report["seed_metrics"]} == {"politics"}
    assert not any(row["successful_seeds"] for row in report["summary"] if row["domain"] == "music")


def test_domain_mean_diagnostics_weight_domains_equally(study, monkeypatch):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [], latencies={"music": 10.0, "politics": 20.0})
    run = original.selected_runs((extension.QWEN_GROUP,), (42,))[0]
    extension.run_evaluation(run, root, output, root, ("music", "politics"), exclusions_path=exclusions)
    report = extension.aggregate_output_limit(root, output, root, seeds=(42,), domains=("music", "politics"), exclusions_path=exclusions)
    rows = {row["domain"]: row for row in report["seed_metrics"]
            if row["population"] == "deduplicated" and row["evaluation"] == "output_limit_1024"}
    assert rows["pooled_micro"]["latency_ms"] == pytest.approx(50 / 3)
    assert rows["domain_mean"]["latency_ms"] == 15
    assert rows["domain_mean"]["vram_peak_mb"] == 0
    old_mean = next(row for row in report["seed_metrics"] if row["population"] == "deduplicated"
                    and row["evaluation"] == "baseline_256" and row["domain"] == "domain_mean")
    assert old_mean["generated_tokens"] is None
    assert old_mean["output_limit_rate"] is None


@pytest.mark.parametrize("metric", ["f1", "delta_f1"])
@pytest.mark.parametrize("evaluation", ["encoder_baseline", "output_limit_1024"])
def test_nan_quality_cannot_bypass_comparison_validation(study, monkeypatch, metric, evaluation):
    root, baseline, output, exclusions, sources, datasets = study
    _fake_predictor(monkeypatch, [])
    run = original.selected_runs((extension.QWEN_GROUP,), (42,))[0]
    extension.run_evaluation(run, root, output, root, ("music",), exclusions_path=exclusions)
    group = original.GROUPS[0] if evaluation == "encoder_baseline" else extension.QWEN_GROUP
    metrics_path = (baseline if evaluation == "encoder_baseline" else output) / sources[(group, 42)]["config"]["experiment_name"] / "music/inference_metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text())
    metrics[metric] = float("nan")
    original.atomic_write_yaml(metrics_path, metrics)
    report = extension.aggregate_output_limit(root, output, root, seeds=(42,), domains=("music",), exclusions_path=exclusions)
    assert not report["selection_complete"]
    assert report["successful_evaluations"] == 0
    assert any(metric in entry["reason"] for entry in report["missing_or_failed"])
