import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.data import crossner_loader as loader
from src.data.dataset_loader import MULTINERD_INFO
from src.evaluate import crossner as evaluation
from src.evaluate import crossner_aggregate as aggregation
from src.seed_provenance import scientific_contract
from src.seed_study import full_run_config_hash, scientific_config_hash


def _runner():
    spec = importlib.util.spec_from_file_location("run_crossner", evaluation.PROJECT_ROOT / "scripts/run_crossner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    run = evaluation.selected_runs(seeds=(42,), models=(evaluation.GROUPS[0],))[0]
    cfg = evaluation.load_experiment_config(evaluation.PROJECT_ROOT / run.config_path)
    root = tmp_path / "source"
    directory = root / Path(run.output_dir).relative_to("results")
    checkpoint = directory / "best_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"id2label": MULTINERD_INFO.id2label}))
    for filename in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (checkpoint / filename).write_text("fixture")
    identity = {"seed": run.seed, "canonical": True, "model_name": run.model_name,
                "scientific_config_hash": scientific_config_hash(cfg), "full_run_config_hash": full_run_config_hash(cfg),
                "checkpoint_used": str(checkpoint), "model_revision": evaluation.MODEL_REVISIONS[run.model_name],
                "checkpoint_selection": "highest_validation_f1"}
    for filename, content in (("run_metadata.yaml", identity), ("inference_metrics.yaml", identity),
                              ("scientific_contract.yaml", scientific_contract(cfg)), ("config_resolved.yaml", cfg)):
        (directory / filename).write_text(yaml.safe_dump(content))
    (directory / "status.json").write_text(json.dumps({**identity, "status": "COMPLETED"}))
    sample = {"tokens": ["Ada", "Paris"], "gold": ["B-PER", "B-LOC"], "pred": ["B-PER", "B-LOC"]}
    (directory / "test_predictions.json").write_text(json.dumps([sample] * 16_454))
    return run, root, directory


def _samples():
    return loader.parse_crossner("Ada\tB-person\nvisited\tO\nParis\tB-location\n\nDune\tB-book\n", "literature")


def _record(sample, prediction=None):
    raw = prediction or sample["gold_bio"]
    return {**sample, "raw_pred_bio": raw, "pred_bio": loader.project_multinerd_bio(raw), "latency_ms": 2.0}


def test_native_mapping_preserves_adjacent_entity_boundaries_and_eof():
    rows = loader.parse_crossner("Ada\tB-researcher\nLovelace\tI-researcher\nBob\tB-person\nMIT\tB-university\nUK\tB-country\nPython\tB-programlang", "ai")
    assert rows[0]["gold_bio"] == ["B-PER", "I-PER", "B-PER", "B-ORG", "B-LOC", "O"]
    assert rows[0]["tokens"] == ["Ada", "Lovelace", "Bob", "MIT", "UK", "Python"]
    assert rows[0]["native_bio"][-1] == "B-programlang"


@pytest.mark.parametrize("text", ["Ada B-person", "Ada\tB-unrecognized", "Ada\tI-person", "Ada\tB-person\nParis\tI-location", "\tO", ""])
def test_invalid_crossner_annotations_rejected(text):
    with pytest.raises(ValueError):
        loader.parse_crossner(text, "literature")


def test_dataset_hash_checked_before_parsing(tmp_path):
    path = tmp_path / loader.CROSSNER_REVISION / "ai" / "test.txt"
    path.parent.mkdir(parents=True)
    path.write_text("Ada\tB-person\n")
    with pytest.raises(ValueError, match="SHA-256"):
        loader.load_crossner_test("ai", tmp_path)


def test_missing_cache_never_downloaded_implicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="preflight"):
        loader.load_crossner_test("music", tmp_path)


def test_projection_counts_shared_predictions_on_unsupported_gold_as_errors():
    rows = _samples()
    assert rows[1]["gold_bio"] == ["O"]
    record = _record(rows[1], ["B-LOC"])
    metrics = evaluation.compute_ner_metrics([record["gold_bio"]], [record["pred_bio"]])
    assert metrics["precision"] == 0.0
    assert loader.project_multinerd_bio(["B-MEDIA", "I-MEDIA", "I-PER"]) == ["O", "O", "I-PER"]


def test_first_subword_alignment_and_missing_words():
    assert evaluation.first_subword_positions([None, 0, 0, 1, 2, 2, None], 3) == [1, 3, 4]
    with pytest.raises(ValueError, match="every original word"):
        evaluation.first_subword_positions([None, 0, 0, 2], 3)


def test_source_checkpoint_and_baseline_are_read_only(source):
    run, root, directory = source
    before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    checked = evaluation.source_run(run, root)
    assert evaluation.baseline_metrics(checked)["f1"] == 1.0
    assert before == {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [("seed", 123), ("canonical", False), ("checkpoint_used", "/tmp/other"), ("model_name", "other")])
def test_source_ownership_rejects_other_run(source, field, value):
    run, root, directory = source
    path = directory / "run_metadata.yaml"
    data = yaml.safe_load(path.read_text())
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        evaluation.source_run(run, root)


def test_source_label_schema_rejected(source):
    run, root, directory = source
    path = directory / "best_model/config.json"
    path.write_text(json.dumps({"id2label": {"0": "O", "1": "B-OTHER"}}))
    with pytest.raises(ValueError, match="label schema"):
        evaluation.source_run(run, root)


def test_checkpoint_only_recovers_incomplete_trailing_append(tmp_path):
    samples = _samples()
    contract = {"domain": "literature", "seed": 42}
    path = tmp_path / "checkpoint.jsonl"
    header = {"contract": contract, "contract_hash": evaluation.canonical_hash(contract)}
    path.write_text(json.dumps(header) + "\n" + json.dumps(_record(samples[0])) + '\n{"tokens":')
    assert len(evaluation.read_checkpoint(path, contract, samples)) == 1
    with pytest.raises(ValueError, match="contract mismatch"):
        evaluation.read_checkpoint(path, {**contract, "domain": "music"}, samples)
    path.write_text(json.dumps(header) + '\n{"tokens":\n')
    with pytest.raises(ValueError, match="corrupt"):
        evaluation.read_checkpoint(path, contract, samples)


def test_checkpoint_rejects_wrong_words_and_partial_coverage(tmp_path):
    samples = _samples()
    record = _record(samples[0])
    record["raw_pred_bio"] = ["B-PER"]
    with pytest.raises(ValueError, match="coverage"):
        evaluation.validate_record(record, samples[0])
    record = _record(samples[0])
    record["tokens"] = ["different"]
    with pytest.raises(ValueError, match="identity"):
        evaluation.validate_record(record, samples[0])


def test_failed_decoder_parse_is_scored_as_empty_prediction():
    from src.decoder.parse_output import entities_to_bio, parse_llm_output_with_diagnostics
    sample = _samples()[0]
    entities, status, diagnostics = parse_llm_output_with_diagnostics("broken", sample["tokens"], frozenset(MULTINERD_INFO.entity_types))
    record = {**_record(sample, entities_to_bio(sample["tokens"], entities)), "raw_output": "broken", "parse_status": status, "parse_diagnostics": diagnostics}
    contract = {"smoke_limit": None, "expected_samples": 1}
    metrics = evaluation.domain_metrics([record], {"f1": 1.0}, contract)
    assert metrics["parse_failure_rate"] == 1.0
    assert metrics["recall"] == 0.0


def test_output_cannot_overlap_source_tree(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        evaluation.ensure_safe_output(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        evaluation.ensure_safe_output(tmp_path / "multinerd" / "new", tmp_path)


def test_orchestrator_has_six_inference_jobs_and_retired_partition_fallback(tmp_path):
    runner = _runner()
    args = runner.parse_args(["--dry-run", "--source-results-root", str(tmp_path / "source")])
    runs = evaluation.selected_runs(args.model, args.seeds)
    assert len(runs) == 6
    commands = [runner.command_for(run, args) for run in runs]
    assert all("scripts/cluster/job_crossner.sh" in command for command in commands)
    assert all("--partition=H100" in command for command in commands)
    assert not any("train" in item for command in commands for item in command)
    assert sum(len(args.domains) for _ in commands) == 30
    assert all("best_model" not in str(command) for command in commands)  # Source ownership is resolved in the worker.


def test_smoke_default_output_is_separate(tmp_path):
    args = _runner().parse_args(["--dry-run", "--source-results-root", str(tmp_path / "source"), "--smoke-limit", "2"])
    assert args.output_root.name == "smoke"


def test_sample_standard_deviation_and_paired_delta():
    assert aggregation.summarize([0.8, 0.9, 1.0])["std"] == pytest.approx(0.1)
    assert aggregation.summarize([0.8])["std"] is None
    records = [{"seed": s, "precision": f, "recall": f, "f1": f, "baseline_f1": 0.95, "delta_f1": f - 0.95} for s, f in zip((42, 123, 456), (0.8, 0.9, 1.0))]
    row = aggregation.summary_row("model", "ai", records, 3)
    assert row["delta_f1_mean"] == pytest.approx(-0.05)
    assert row["delta_f1_std"] == pytest.approx(0.1)
    assert row["complete"]


def test_evaluation_completion_resume_and_incomplete_aggregation(source, tmp_path, monkeypatch):
    run, root, directory = source
    samples = _samples()
    monkeypatch.setitem(evaluation.TEST_SIZES, "literature", 2)
    monkeypatch.setattr(evaluation, "load_crossner_test", lambda *a, **k: samples)
    monkeypatch.setattr(aggregation, "load_crossner_test", lambda *a, **k: samples)
    monkeypatch.setattr("src.run_metadata.collect_run_metadata", lambda *a: {"test": True})
    calls = []

    class FakePredictor:
        def __init__(self, source):
            self.torch = type("Torch", (), {"cuda": type("Cuda", (), {"max_memory_allocated": staticmethod(lambda: 0)})})

        def predict(self, sample):
            calls.append(sample["sample_id"])
            return _record(sample)

    monkeypatch.setattr(evaluation, "FrozenPredictor", FakePredictor)
    before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    output = tmp_path / "crossner"
    evaluation.run_evaluation(run, root, output, tmp_path, ("literature",))
    assert len(calls) == 3  # warmup + two measured examples
    evaluation.run_evaluation(run, root, output, tmp_path, ("literature",))
    assert len(calls) == 3  # Completed domain is reused without loading a model.
    assert before == {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    report = aggregation.aggregate(root, output, tmp_path, (run.group_key,), (42, 123), ("literature",))
    assert report["successful_evaluations"] == 1
    assert not report["selection_complete"]
    assert not report["complete"]
    assert report["summary"][0]["f1_std"] is None
    assert (output / "aggregate/summary.csv").exists()
    assert (output / "aggregate/summary.md").exists()
    target = output / directory.name / "literature/test_predictions.json"
    target.write_text("[]")
    with pytest.raises(ValueError, match="altered"):
        evaluation.completed_domain(target.parent, evaluation.evaluation_contract(evaluation.source_run(run, root), "literature"), samples)
