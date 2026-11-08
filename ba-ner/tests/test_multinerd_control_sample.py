"""CPU-only tests for the MultiNERD control sample (toy source trees, fake predictor)."""

import copy
import json

import pytest
import yaml

from src.evaluate import multinerd_control_sample as module
from src.evaluate import multinerd_output_limit as mnol
from test_multinerd_output_limit import FAILED, TOY, _create_source, _dataset_rows, _regenerate, _run

QWEN_27B, QWEN_4B, QWEN_08B = mnol.GROUPS


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "source"
    output = root / "multinerd_control_1024"
    monkeypatch.setattr(mnol, "EXPECTED_SAMPLES", len(TOY))
    monkeypatch.setattr(mnol, "load_test_split", _dataset_rows)
    monkeypatch.setattr("src.run_metadata.collect_run_metadata", lambda *a: {
        "fixture": True, "packages": {"torch": {"installed": True, "version": "2.13.0"}},
        "cuda": {"devices": ["NVIDIA H100 80GB HBM3"]}, "git": {"commit": "def456"}})
    sources = {(run.group_key, run.seed): _create_source(run, root) for run in mnol.selected_runs()}
    return root, output, sources


def _fake_predictor(monkeypatch, calls, *, fail_after=None, changed=(), fail_on_init=False):
    class Predictor:
        def __init__(self, source, *, max_new_tokens):
            if fail_on_init:
                raise AssertionError("a completed control sample must not load a model")
            assert max_new_tokens == 1024
            self.torch = type("Torch", (), {"cuda": type("Cuda", (), {"max_memory_allocated": staticmethod(lambda: 512 * 1024 ** 2)})})
            self.total_params = 1234

        def predict(self, index, tokens, stored_raw_output):
            calls.append(index)
            if fail_after is not None and len(calls) > fail_after:
                raise RuntimeError("interrupted generation")
            assert tokens == TOY[index]["tokens"] and stored_raw_output == TOY[index]["raw_output"]
            generated = _regenerate(index, tokens)
            if index in changed:  # a numerically diverging answer on another GPU
                generated = {**generated, "raw_output": "[]", "pred_entities": [], "pred_bio": ["O"] * len(tokens),
                             "stored_prefix_match": False}
            return generated

    monkeypatch.setattr(mnol, "RegenerationPredictor", Predictor)


def test_selection_skips_parse_failures_and_is_seeded():
    selection = module.control_indices(TOY, 3, 7)
    assert selection == sorted(set(range(len(TOY))) - set(FAILED))
    assert module.control_indices(TOY, 2, 7) == module.control_indices(TOY, 2, 7)
    assert set(module.control_indices(TOY, 2, 7)) <= set(selection)
    with pytest.raises(ValueError):
        module.control_indices(TOY, 4, 7)  # only three parsed answers exist


def test_run_records_equality_latency_and_leaves_sources_untouched(study, monkeypatch):
    root, output, _ = study
    before = {path: path.read_bytes() for path in (root / "multinerd").rglob("*") if path.is_file()}
    calls = []
    _fake_predictor(monkeypatch, calls, changed=(3,))
    metrics = module.run_control(_run(QWEN_08B, 42), root, output, sample_size=3, sample_seed=7)
    control = metrics["control"]
    assert calls[0] == calls[1]  # unmeasured warm-up of the first sampled sentence
    assert control["sample_count"] == 3 and control["raw_output_equal_count"] == 2
    assert control["raw_output_mismatch_indices"] == [3]
    assert control["stop_reason_counts"]["eos"] == 3 and control["limit_hit_count"] == 0
    assert control["latency_ms"]["mean"] == 10.0 and control["vram_peak_mb"] == 512.0
    assert control["subset_metrics"]["stored_256"]["f1"] > control["subset_metrics"]["control_1024"]["f1"]
    assert metrics["environment"]["gpu"] == "NVIDIA H100 80GB HBM3"
    assert {path: path.read_bytes() for path in (root / "multinerd").rglob("*") if path.is_file()} == before
    status = json.loads((output / metrics["experiment_name"] / "status.json").read_text())
    assert status["status"] == "COMPLETED" and not (output / metrics["experiment_name"] / module.CHECKPOINT_NAME).exists()


def test_checkpoint_resume_and_completed_skip(study, monkeypatch):
    root, output, _ = study
    calls = []
    _fake_predictor(monkeypatch, calls, fail_after=2)
    with pytest.raises(RuntimeError):
        module.run_control(_run(QWEN_4B, 123), root, output, sample_size=3, sample_seed=7)
    calls.clear()
    _fake_predictor(monkeypatch, calls)
    metrics = module.run_control(_run(QWEN_4B, 123), root, output, sample_size=3, sample_seed=7, resume_failed=True)
    assert len(calls) == 1 + 2  # warm-up plus the two sentences missing from the checkpoint
    assert metrics["control"]["sample_count"] == 3
    _fake_predictor(monkeypatch, [], fail_on_init=True)
    again = module.run_control(_run(QWEN_4B, 123), root, output, sample_size=3, sample_seed=7)
    assert again["predictions_sha256"] == metrics["predictions_sha256"]


def test_rejects_output_limit_root_and_source_root(study):
    root, _, _ = study
    with pytest.raises(ValueError):
        module.run_control(_run(QWEN_27B, 42), root, root / "multinerd_output_limit_1024", sample_size=2)
    with pytest.raises(ValueError):
        module.run_control(_run(QWEN_27B, 42), root, root, sample_size=2)


def test_aggregate_verifies_records_and_estimates_full_test_latency(study, monkeypatch):
    root, output, _ = study
    _fake_predictor(monkeypatch, [])
    for run in mnol.selected_runs():
        module.run_control(run, root, output, sample_size=2, sample_seed=7)
    limit_root = root / "multinerd_output_limit_1024"
    experiment = module.load_experiment_config(module.PROJECT_ROOT / _run(QWEN_27B, 42).config_path)["experiment_name"]
    (limit_root / experiment).mkdir(parents=True)
    (limit_root / experiment / "inference_metrics.yaml").write_text(yaml.safe_dump(
        {"regeneration": {"selected_count": 2, "latency_ms": {"total_seconds": 1.0}}}))
    report = module.aggregate(root, output)
    assert report["complete"] and len(report["rows"]) == 9
    row = next(r for r in report["rows"] if r["experiment_name"] == experiment)
    assert row["full_test_latency_ms_estimate"] == pytest.approx(((len(TOY) - 2) * 10.0 + 1000.0) / len(TOY))
    assert all(r["full_test_latency_ms_estimate"] is None for r in report["rows"] if r is not row)
    assert (output / "aggregate" / "control_summary.md").exists()

    predictions = output / experiment / "control_predictions.json"
    records = json.loads(predictions.read_text())
    tampered = copy.deepcopy(records)
    tampered[0]["latency_ms"] = 99.0
    predictions.write_text(json.dumps(tampered))
    report = module.aggregate(root, output)
    assert not report["complete"] and report["missing_or_failed"][0]["seed"] == 42
