"""CPU-only tests for the MultiNERD output-limit re-generation (toy source trees, fake predictor)."""

import copy
import json
import shutil
import socket
import statistics
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from src.data.dataset_loader import MULTINERD_INFO
from src.data.preprocess_decoder import build_system_prompt, format_numbered_tokens
from src.decoder.parse_output import entities_to_bio, evaluate_llm_predictions
from src.evaluate import crossner as original
from src.evaluate import multinerd_output_limit as module
from src.seed_provenance import scientific_contract
from src.seed_study import full_run_config_hash, scientific_config_hash

ZERO_DIAGNOSTICS = {"invalid_items": 0, "missing_fields": 0, "unknown_types": 0, "wrong_schema": 0,
                    "invalid_offsets": 0, "text_mismatches": 0, "overlaps": 0}
QWEN_27B, QWEN_4B, QWEN_08B = module.GROUPS


def _entity(tokens, start, end, kind):
    return {"start_token": start, "end_token": end, "text": " ".join(tokens[start:end]), "type": kind}


def _toy_records():
    # tokens, gold spans, stored predicted spans (None = parse failure cut off at the 256-token limit)
    specs = [
        (["Ada", "Lovelace", "visited", "Paris", "."], [(0, 2, "PER"), (3, 4, "LOC")], [(0, 2, "PER"), (3, 4, "LOC")]),
        (["Marie", "Curie", "worked", "in", "Warsaw"], [(0, 2, "PER"), (4, 5, "LOC")], None),
        (["Bob", "ate", "pizza"], [(0, 1, "PER"), (2, 3, "FOOD")], [(0, 1, "PER")]),
        (["The", "Nile", "flows", "north"], [(1, 2, "LOC")], [(1, 2, "LOC")]),
        (["Zeus", "loved", "Athens"], [(0, 1, "MYTH"), (2, 3, "LOC")], None),
    ]
    records = []
    for tokens, gold, pred in specs:
        gold_entities = [_entity(tokens, *span) for span in gold]
        pred_entities = [] if pred is None else [_entity(tokens, *span) for span in pred]
        records.append({"tokens": tokens, "gold_entities": gold_entities, "gold_bio": entities_to_bio(tokens, gold_entities),
                        "pred_entities": pred_entities, "pred_bio": entities_to_bio(tokens, pred_entities),
                        "raw_output": '[{"start_token": 0' if pred is None else json.dumps(pred_entities),
                        "parse_status": "failed" if pred is None else "ok", "parse_diagnostics": dict(ZERO_DIAGNOSTICS)})
    return records


TOY = _toy_records()
FAILED = [1, 4]


def _dataset_rows():
    return [{key: copy.deepcopy(record[key]) for key in ("tokens", "gold_entities", "gold_bio")} for record in TOY]


def _regenerate(index, tokens, *, partial=False):
    if TOY[index]["parse_status"] == "ok":  # an answer that ended before 256 tokens is reproduced unchanged
        entities, raw = copy.deepcopy(TOY[index]["pred_entities"]), TOY[index]["raw_output"]
    else:
        entities = copy.deepcopy(TOY[index]["gold_entities"][:1] if partial else TOY[index]["gold_entities"])
        raw = json.dumps(entities)
    return {"index": index, "raw_output": raw, "pred_entities": entities, "parse_status": "ok",
            "parse_diagnostics": dict(ZERO_DIAGNOSTICS), "pred_bio": entities_to_bio(tokens, entities),
            "latency_ms": 10.0, "stored_prefix_match": True, "generated_token_count": 7,
            "token_limit_reached": False, "stop_reason": "eos", "json_complete": True}


def _create_source(run, root):
    cfg = original.load_experiment_config(original.PROJECT_ROOT / run.config_path)
    directory = root / Path(run.output_dir).relative_to("results")
    checkpoint = directory / "best_lora_adapter"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": run.model_name, "inference_mode": True}))
    for filename in ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (checkpoint / filename).write_text(f"fixture {run.group_key} {run.seed}")
    identity = {"seed": run.seed, "canonical": True, "model_name": run.model_name,
                "scientific_config_hash": scientific_config_hash(cfg), "full_run_config_hash": full_run_config_hash(cfg),
                "checkpoint_used": str(checkpoint), "model_revision": original.MODEL_REVISIONS[run.model_name],
                "checkpoint_selection": scientific_contract(cfg)["evaluation"]["checkpoint_selection"]}
    records = copy.deepcopy(TOY)
    metrics = {**identity, **module.llm_metrics(records), "max_new_tokens": 256, "inference_batch_size": 1,
               "precision_mode": "qlora_4bit", "attn_impl": "sdpa", "thinking_enabled": False,
               "latency_ms_mean": 100.0, "latency_ms_p50": 90.0, "latency_ms_p95": 200.0,
               "latency_measurement": "amortized_per_sample", "generation_samples_per_second": 10.0,
               "vram_peak_mb": 1000.0, "total_params": 1234,
               "run_metadata": {"packages": {"torch": {"installed": True, "version": "2.13.0"}},
                                "cuda": {"devices": ["NVIDIA H100 NVL"]}, "git": {"commit": "abc123"}}}
    for filename, content in (("run_metadata.yaml", identity), ("inference_metrics.yaml", metrics),
                              ("scientific_contract.yaml", scientific_contract(cfg)), ("config_resolved.yaml", cfg)):
        (directory / filename).write_text(yaml.safe_dump(content))
    (directory / "status.json").write_text(json.dumps({**identity, "status": "COMPLETED"}))
    original.atomic_write_json(directory / "test_predictions.json", records)
    return original.source_run(run, root)


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "source"
    output = root / "multinerd_output_limit_1024"
    monkeypatch.setattr(module, "EXPECTED_SAMPLES", len(TOY))
    monkeypatch.setattr(module, "load_test_split", _dataset_rows)
    monkeypatch.setattr("src.run_metadata.collect_run_metadata", lambda *a: {
        "fixture": True, "packages": {"torch": {"installed": True, "version": "2.13.0"}},
        "cuda": {"devices": ["NVIDIA H100 80GB HBM3"]}, "git": {"commit": "def456"}})
    sources = {(run.group_key, run.seed): _create_source(run, root) for run in module.selected_runs()}
    return root, output, sources


def _run(group, seed):
    return module.selected_runs((group,), (seed,))[0]


def _fake_predictor(monkeypatch, calls, *, fail_after=None, fail_on_init=False, partial_seeds=()):
    class Predictor:
        def __init__(self, source, *, max_new_tokens):
            if fail_on_init:
                raise AssertionError("a completed evaluation must not load a model")
            assert max_new_tokens == 1024
            self.seed = source["identity"]["seed"]
            self.torch = type("Torch", (), {"cuda": type("Cuda", (), {"max_memory_allocated": staticmethod(lambda: 512 * 1024 ** 2)})})
            self.total_params = 1234

        def predict(self, index, tokens, stored_raw_output):
            calls.append(index)
            if fail_after is not None and len(calls) > fail_after:
                raise RuntimeError("interrupted generation")
            assert tokens == TOY[index]["tokens"] and stored_raw_output == TOY[index]["raw_output"]
            return _regenerate(index, tokens, partial=self.seed in partial_seeds)

    monkeypatch.setattr(module, "RegenerationPredictor", Predictor)


def test_selected_runs_cover_nine_canonical_qwen_runs():
    runs = module.selected_runs()
    assert len(runs) == 9
    assert {(run.group_key, run.seed) for run in runs} == {(group, seed) for group in module.GROUPS for seed in module.SEEDS}
    assert all(run.model_family == "decoder" and run.canonical and not run.historical for run in runs)
    assert all(run.config_path.startswith("configs/qwen35_") for run in runs)
    assert [run.seed for run in module.selected_runs((QWEN_08B,), (42,))] == [42]
    with pytest.raises(ValueError):
        module.selected_runs(("deberta-v3-large",))
    with pytest.raises(ValueError):
        module.selected_runs(seeds=(7,))
    assert module.run_for_config(module.PROJECT_ROOT / "configs/qwen35_27b_3ep_seed456.yaml").seed == 456
    with pytest.raises(ValueError, match="canonical"):
        module.run_for_config(module.PROJECT_ROOT / "configs/qwen35_27b.yaml")  # historical 2ep run


def test_failed_indices_selection_and_smoke_limit():
    assert module.failed_indices(TOY) == FAILED
    assert module.failed_indices(TOY, smoke_limit=1) == [1]
    assert module.failed_indices(TOY, smoke_limit=9) == FAILED
    assert module.failed_indices(list(reversed(TOY))) == [0, 3]
    for bad in (0, -1, 1.0, True):
        with pytest.raises(ValueError):
            module.failed_indices(TOY, smoke_limit=bad)


def test_merge_preserves_untouched_records_byte_for_byte():
    regenerated = [module.build_regenerated_record(TOY[i], i, _regenerate(i, TOY[i]["tokens"])) for i in FAILED]
    merged = module.merge_predictions(TOY, regenerated)
    assert len(merged) == len(TOY)
    for index, record in enumerate(merged):
        if index in FAILED:
            assert record["regenerated"] is True and record["index"] == index
            assert record["baseline_256"] == {k: TOY[index][k] for k in module.REGENERATED_KEYS}
            assert record["parse_status"] == "ok" and record["pred_entities"] == TOY[index]["gold_entities"]
            assert record["gold_bio"] == TOY[index]["gold_bio"] and record["tokens"] == TOY[index]["tokens"]
            assert record["stop_reason"] == "eos" and record["generated_token_count"] == 7
            module.validate_regenerated_record(record, TOY[index], 1024)
        else:
            assert record == {**TOY[index], "regenerated": False}
    untouched = [{k: v for k, v in r.items() if k != "regenerated"} for i, r in enumerate(merged) if i not in FAILED]
    assert json.dumps(untouched, sort_keys=True) == json.dumps([r for i, r in enumerate(TOY) if i not in FAILED], sort_keys=True)
    with pytest.raises(ValueError, match="duplicated"):
        module.merge_predictions(TOY, regenerated + regenerated[:1])
    with pytest.raises(ValueError, match="out of range"):
        module.merge_predictions(TOY, [{**regenerated[0], "index": 99}])
    with pytest.raises(ValueError, match="stored sentence"):
        module.merge_predictions(TOY, [{**regenerated[0], "tokens": ["different"]}])
    for key, bad in (("generated_token_count", 1025), ("token_limit_reached", True), ("stop_reason", "max_new_tokens"),
                     ("json_complete", False), ("latency_ms", -1.0), ("stored_prefix_match", 1), ("regenerated", False),
                     ("pred_bio", ["O"] * 5), ("baseline_256", {})):
        with pytest.raises(ValueError):
            module.validate_regenerated_record({**regenerated[0], key: bad}, TOY[1], 1024)
    with pytest.raises(ValueError, match="unexpected keys"):
        module.validate_regenerated_record({**regenerated[0], "extra": 1}, TOY[1], 1024)


def test_metrics_recomputation_on_toy_set(study, monkeypatch):
    root, output, sources = study
    source = sources[(QWEN_27B, 42)]
    stored = module.load_source_predictions(source)
    baseline = module.baseline_metrics_256(source, stored)
    expected = evaluate_llm_predictions([r["tokens"] for r in stored], [r["gold_entities"] for r in stored],
                                        [r["pred_entities"] for r in stored], [r["parse_status"] for r in stored],
                                        [r["parse_diagnostics"] for r in stored])
    assert {key: baseline[key] for key in expected} == expected
    assert baseline["parse_failed"] == 2 and baseline["recomputed_matches_stored"] and baseline["max_new_tokens"] == 256
    assert baseline["precision"] == 1.0 and baseline["recall"] == pytest.approx(4 / 9)
    assert baseline["vram_peak_mb"] == 1000.0 and baseline["gpu"] == "NVIDIA H100 NVL" and baseline["git_commit"] == "abc123"
    assert set(baseline["per_entity"]) == set(MULTINERD_INFO.entity_types)
    calls = []
    _fake_predictor(monkeypatch, calls)
    metrics = module.run_evaluation(_run(QWEN_27B, 42), root, output)
    assert calls == [1, 1, 4]  # unmeasured warm-up, then the two stored failures
    assert metrics["recall"] > baseline["recall"] and metrics["f1"] > baseline["f1"]
    assert metrics["f1"] == metrics["test_f1"] == pytest.approx(16 / 17)  # tp 8/9: record 2 keeps its missing FOOD
    assert metrics["recall"] == pytest.approx(8 / 9) and metrics["precision"] == 1.0
    assert metrics["parse_failed"] == 0 and metrics["delta_vs_256"]["parse_failed"] == -2
    assert len(metrics["per_entity"]) == 15 and set(metrics["per_entity"]) == set(MULTINERD_INFO.entity_types)
    assert metrics["macro_f1"] == pytest.approx(statistics.fmean(v["f1"] for v in metrics["per_entity"].values()))
    assert metrics["delta_vs_256"]["f1"] == pytest.approx(metrics["f1"] - baseline["f1"])
    assert metrics["delta_vs_256"]["per_entity_f1"]["LOC"] > 0 and metrics["delta_vs_256"]["per_entity_f1"]["PER"] > 0
    assert metrics["baseline_256"] == baseline
    regeneration = metrics["regeneration"]
    assert regeneration["selected_indices"] == FAILED and regeneration["selected_count"] == 2
    assert regeneration["subset_metrics"] == {"baseline_256": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                                              "output_limit_1024": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}
    assert regeneration["truncated_without_closing_bracket"] == 2
    assert regeneration["gold_entities_in_selection"] == 4 and regeneration["gold_entity_share"] == pytest.approx(4 / 9)
    assert regeneration["parse_status_counts_after"] == {"ok": 2, "markdown_stripped": 0, "regex_fallback": 0, "failed": 0}
    assert regeneration["stop_reason_counts"] == {"eos": 2, "max_new_tokens": 0, "other": 0}
    assert regeneration["still_failed_count"] == 0 and regeneration["limit_hit_count"] == 0
    assert regeneration["stored_prefix_match_count"] == 2 and regeneration["stored_prefix_mismatch_indices"] == []
    assert regeneration["generated_tokens"]["mean"] == 7 and regeneration["generated_tokens"]["total"] == 14
    assert regeneration["latency_ms"]["total_seconds"] == pytest.approx(0.02)
    assert metrics["vram_peak_mb"] == 512.0 and metrics["total_params"] == 1234
    assert metrics["environment"]["matches_source_packages"] and not metrics["environment"]["matches_source_gpu"]
    assert metrics["environment"]["gpu"] == "NVIDIA H100 80GB HBM3" and metrics["environment"]["source_gpu"] == "NVIDIA H100 NVL"
    assert metrics["dataset_alignment"] == {"dataset_revision": module.DATASET_REVISION, "verified_samples": 5}
    assert metrics["complete"] and metrics["smoke_limit"] is None and not metrics["canonical"]
    assert metrics["sample_count"] == 5 and metrics["max_new_tokens"] == 1024 and metrics["baseline_max_new_tokens"] == 256
    assert "latency_ms_mean" not in metrics  # no full-set latency can be reported
    assert metrics["contract"]["selection"]["indices"] == FAILED
    assert set(metrics["code_hashes"]) >= set(module.CODE_PATHS) | {"src/evaluate/multinerd_output_limit.py"}
    directory = output / source["config"]["experiment_name"]
    saved = yaml.safe_load((directory / "inference_metrics.yaml").read_text())
    assert saved["f1"] == metrics["f1"] and saved["contract"] == metrics["contract"]
    assert saved["predictions_sha256"] == original.file_hash(directory / "test_predictions.json")
    path = source["directory"] / "inference_metrics.yaml"
    data = yaml.safe_load(path.read_text())
    data["f1"] += 1e-6
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="differs"):
        module.baseline_metrics_256(source, stored)


def test_checkpoint_resume_recovers_prefix_and_is_idempotent(study, monkeypatch, capsys):
    root, output, sources = study
    run = _run(QWEN_27B, 42)
    directory = output / sources[(QWEN_27B, 42)]["config"]["experiment_name"]
    calls = []
    _fake_predictor(monkeypatch, calls, fail_after=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.run_evaluation(run, root, output)
    assert calls == [1, 1, 4]
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "FAILED" and "interrupted" in status["error"]
    assert status["selected_count"] == 2 and status["completed_count"] == 1 and status["baseline_f1"] > 0
    assert (directory / "inference_checkpoint.jsonl").exists() and (directory / "contract.json").exists()
    assert not (directory / module.LOCK_NAME).exists()
    calls.clear()
    _fake_predictor(monkeypatch, calls)
    metrics = module.run_evaluation(run, root, output, resume_failed=True)
    assert calls == [4, 4]  # warm-up on the remaining sentence, then only that sentence
    assert not (directory / "inference_checkpoint.jsonl").exists()
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "COMPLETED" and status["completed_count"] == 2 and status["error"] is None
    assert status["stored_prefix_match_count"] == 2 and status["eta_seconds"] == 0.0
    assert metrics["regeneration"]["selected_indices"] == FAILED
    assert "MNOL_RESULT" in capsys.readouterr().out
    _fake_predictor(monkeypatch, calls, fail_on_init=True)
    again = module.run_evaluation(run, root, output)
    assert again["predictions_sha256"] == metrics["predictions_sha256"]
    assert "SKIP_COMPLETED" in capsys.readouterr().out
    assert json.loads((directory / "status.json").read_text())["status"] == "COMPLETED"
    with pytest.raises(ValueError, match="contract"):
        module.run_evaluation(run, root, output, max_new_tokens=2048)
    assert json.loads((directory / "status.json").read_text())["status"] == "FAILED"
    _fake_predictor(monkeypatch, calls)
    module.run_evaluation(run, root, output / "smoke", smoke_limit=1)
    _fake_predictor(monkeypatch, calls, fail_on_init=True)
    with pytest.raises(ValueError, match="contract"):
        module.run_evaluation(run, root, output / "smoke", smoke_limit=2)
    with pytest.raises(ValueError, match="separate path"):
        module.run_evaluation(run, root, output, smoke_limit=1)


def test_checkpoint_tolerates_only_interrupted_final_append(tmp_path):
    contract = {"decoder": {"max_new_tokens": 1024}, "selection": {"indices": FAILED}}
    header = module.checkpoint_header(contract)
    first = module.build_regenerated_record(TOY[1], 1, _regenerate(1, TOY[1]["tokens"]))
    second = module.build_regenerated_record(TOY[4], 4, _regenerate(4, TOY[4]["tokens"]))
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(first) + '\n{"tokens":')
    assert [r["index"] for r in module.read_regeneration_checkpoint(path, contract, FAILED, TOY)] == [1]
    with pytest.raises(ValueError, match="contract mismatch"):
        module.read_regeneration_checkpoint(path, {**contract, "version": 2}, FAILED, TOY)
    path.write_text(json.dumps(header) + '\n{"tokens":\n')
    with pytest.raises(ValueError, match="corrupt"):
        module.read_regeneration_checkpoint(path, contract, FAILED, TOY)
    path.write_text(json.dumps(header) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="selection order"):
        module.read_regeneration_checkpoint(path, contract, FAILED, TOY)
    path.write_text(json.dumps(header) + "\n" + "\n".join(json.dumps(r) for r in (first, second, first)) + "\n")
    with pytest.raises(ValueError, match="too many"):
        module.read_regeneration_checkpoint(path, contract, FAILED, TOY)
    path.write_text(json.dumps(header) + "\n" + json.dumps({**first, "generated_token_count": 2048}) + "\n")
    with pytest.raises(ValueError, match="token count"):
        module.read_regeneration_checkpoint(path, contract, FAILED, TOY)
    assert module.read_regeneration_checkpoint(tmp_path / "missing.jsonl", contract, FAILED, TOY) == []


def test_refuses_output_inside_source_root(tmp_path):
    root = tmp_path / "source"
    for output, smoke in ((root, None), (root / "multinerd" / "x", None), (root / "multinerd", None), (tmp_path, None),
                          (root / "crossner" / "x", None), (root / "crossner_output_limit_1024", None),
                          (root / "multinerd_output_limit_1024", 2), (root / "multinerd_output_limit_2048", 1)):
        with pytest.raises(ValueError):
            module.ensure_output_safe(output, root, smoke_limit=smoke)
    module.ensure_output_safe(root / "multinerd_output_limit_1024", root)
    module.ensure_output_safe(root / "multinerd_output_limit_1024" / "smoke", root, smoke_limit=2)
    module.ensure_output_safe(tmp_path / "elsewhere", root, smoke_limit=2)


def test_source_tree_is_read_only_after_run(study, monkeypatch):
    root, output, sources = study
    before = {str(p): p.read_bytes() for p in (root / "multinerd").rglob("*") if p.is_file()}
    _fake_predictor(monkeypatch, [])
    module.run_evaluation(_run(QWEN_08B, 123), root, output)
    module.run_evaluation(_run(QWEN_4B, 456), root, output / "smoke", smoke_limit=1, control_limit=2)
    assert before == {str(p): p.read_bytes() for p in (root / "multinerd").rglob("*") if p.is_file()}
    assert {p.name for p in (root / "multinerd").iterdir()} == {Path(s["directory"]).name for s in sources.values()}
    smoke = output / "smoke" / sources[(QWEN_4B, 456)]["config"]["experiment_name"]
    control = json.loads((smoke / "control_predictions.json").read_text())
    assert [row["index"] for row in control] == [0, 2] and all(row["raw_output_equal_to_stored"] for row in control)
    smoke_metrics = yaml.safe_load((smoke / "inference_metrics.yaml").read_text())
    assert smoke_metrics["smoke_limit"] == 1 and not smoke_metrics["complete"]
    assert smoke_metrics["regeneration"]["selected_indices"] == [1]
    assert smoke_metrics["control_sample"] == {"count": 2, "indices": [0, 2], "raw_output_equal_count": 2, "pred_entities_equal_count": 2}


def test_dataset_alignment_mismatch_rejected():
    rows = _dataset_rows()
    assert module.verify_dataset_alignment(TOY, rows) == {"dataset_revision": module.DATASET_REVISION, "verified_samples": 5}
    rows[2]["tokens"][0] = "A"
    with pytest.raises(ValueError, match="mismatch at index 2"):
        module.verify_dataset_alignment(TOY, rows)
    rows = _dataset_rows()
    rows[3]["gold_entities"][0]["type"] = "ORG"
    with pytest.raises(ValueError, match="mismatch at index 3"):
        module.verify_dataset_alignment(TOY, rows)
    with pytest.raises(ValueError, match="mismatch"):
        module.verify_dataset_alignment(TOY, _dataset_rows()[:4])


def test_predictor_keeps_greedy_settings_and_computes_prefix_match(monkeypatch):
    calls, prompts = [], []
    inputs = {"input_ids": type("Input", (), {"shape": (1, 5)})()}
    new_ids = [1] * 299 + [9]
    tokens = TOY[2]["tokens"]
    full = json.dumps(TOY[2]["gold_entities"])
    stored_raw = '[{"start_token": 0'

    class Result:
        def __getitem__(self, item):
            assert item == (0, slice(5, None))
            return type("Ids", (), {"tolist": lambda self: list(new_ids)})()

    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Result()

    class Tokenizer:
        eos_token_id = pad_token_id = 9

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens and ids in (new_ids, new_ids[:256])
            return full if ids == new_ids else stored_raw

    def prepare(tokenizer, messages, device):
        prompts.append(messages)
        return inputs

    monkeypatch.setattr(module, "prepare_generation_inputs", prepare)
    predictor = object.__new__(module.RegenerationPredictor)
    predictor.torch = type("Torch", (), {"cuda": type("Cuda", (), {"synchronize": staticmethod(lambda: None)}),
                                         "no_grad": staticmethod(nullcontext)})()
    predictor.tokenizer, predictor.model, predictor.device, predictor.max_new_tokens = Tokenizer(), Model(), "cpu", 1024
    record = predictor.predict(2, tokens, stored_raw)
    assert calls[0]["max_new_tokens"] == 1024 and calls[0]["do_sample"] is False
    assert calls[0]["temperature"] is None and calls[0]["top_p"] is None
    assert calls[0]["eos_token_id"] == 9 and calls[0]["pad_token_id"] == 9
    assert prompts[0] == [{"role": "system", "content": build_system_prompt(MULTINERD_INFO.entity_types)},
                          {"role": "user", "content": format_numbered_tokens(tokens)}]
    assert record["index"] == 2 and record["generated_token_count"] == 300
    assert record["stop_reason"] == "eos" and record["token_limit_reached"] is False and record["json_complete"] is True
    assert record["parse_status"] == "ok" and record["pred_bio"] == ["B-PER", "O", "B-FOOD"]  # 15-type, not projected
    assert record["stored_prefix_match"] is True and record["latency_ms"] >= 0
    assert predictor.predict(2, tokens, "a different stored answer")["stored_prefix_match"] is False


def test_aggregate_mean_sd_ddof1_and_incomplete_selection(study, monkeypatch):
    root, output, sources = study
    _fake_predictor(monkeypatch, [], partial_seeds=(456,))
    for run in module.selected_runs():
        module.run_evaluation(run, root, output)
    report = module.aggregate(root, output)
    assert report["complete"] and report["selection_complete"] and report["ddof"] == 1
    assert report["successful_runs"] == report["expected_runs"] == 9 and report["missing_or_failed"] == []
    rows = [r for r in report["seed_metrics"] if r["model"] == QWEN_4B and r["evaluation"] == "output_limit_1024"]
    assert [r["seed"] for r in rows] == [42, 123, 456]
    assert rows[0]["f1"] == rows[1]["f1"] == pytest.approx(16 / 17) and rows[2]["f1"] == pytest.approx(0.8)
    summary = next(r for r in report["summary"] if r["model"] == QWEN_4B and r["evaluation"] == "output_limit_1024")
    assert summary["f1_mean"] == pytest.approx(statistics.fmean(r["f1"] for r in rows))
    assert summary["f1_std"] == pytest.approx(statistics.stdev(r["f1"] for r in rows)) and summary["f1_std"] > 0
    assert summary["successful_seeds"] == 3 and summary["complete"] and summary["seeds"] == "42,123,456"
    assert summary["delta_f1_vs_256_mean"] > 0 and summary["regenerated_count_mean"] == 2 and summary["still_failed_count_mean"] == 0
    assert summary["regenerated_latency_ms_mean_mean"] == 10.0 and summary["vram_peak_mb_mean"] == 512.0
    baseline_row = next(r for r in report["summary"] if r["model"] == QWEN_4B and r["evaluation"] == "baseline_256")
    assert baseline_row["f1_std"] == 0 and baseline_row["regenerated_count_mean"] is None
    assert baseline_row["parse_failed_mean"] == 2 and baseline_row["vram_peak_mb_mean"] == 1000.0
    per_entity = next(r for r in report["per_entity_summary"] if r["model"] == QWEN_4B
                      and r["evaluation"] == "output_limit_1024" and r["entity_type"] == "LOC")
    assert per_entity["f1_mean"] > 0.6 and per_entity["delta_f1_vs_256_mean"] > 0 and per_entity["delta_f1_vs_256_std"] > 0
    assert len(report["per_entity_summary"]) == 3 * 2 * 15
    for name in ("comparison.yaml", "comparison.csv", "comparison.md", "seed_metrics.csv", "per_entity_summary.csv"):
        assert (output / "aggregate" / name).is_file()
    assert "Micro-F1" in (output / "aggregate" / "comparison.md").read_text() and "COMPLETE;" in (output / "aggregate" / "comparison.md").read_text()
    assert yaml.safe_load((output / "aggregate" / "comparison.yaml").read_text())["complete"]
    assert module.main(["--aggregate", "--source-results-root", str(root), "--output-root", str(output)]) == 0
    # A damaged untouched record cannot hide behind a refreshed checksum.
    experiment = sources[(QWEN_27B, 123)]["config"]["experiment_name"]
    predictions = output / experiment / "test_predictions.json"
    merged = json.loads(predictions.read_text())
    merged[0]["pred_entities"], merged[0]["pred_bio"] = [], ["O"] * 5
    original.atomic_write_json(predictions, merged)
    metrics_path = output / experiment / "inference_metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text())
    metrics["predictions_sha256"] = original.file_hash(predictions)
    original.atomic_write_yaml(metrics_path, metrics)
    report = module.aggregate(root, output)
    assert not report["selection_complete"] and report["successful_runs"] == 8
    assert [(m["model"], m["seed"]) for m in report["missing_or_failed"]] == [(QWEN_27B, 123)]
    assert "Untouched sentence 0" in report["missing_or_failed"][0]["reason"]
    # A missing seed and a smoke result are both excluded; subsets are never reported as complete.
    shutil.rmtree(output / experiment)
    experiment = sources[(QWEN_08B, 456)]["config"]["experiment_name"]
    shutil.rmtree(output / experiment)
    module.run_evaluation(_run(QWEN_08B, 456), root, output / "smoke", smoke_limit=1)
    shutil.move(output / "smoke" / experiment, output / experiment)
    report = module.aggregate(root, output)
    assert not report["selection_complete"] and not report["complete"] and report["successful_runs"] == 7
    assert {(m["model"], m["seed"]) for m in report["missing_or_failed"]} == {(QWEN_27B, 123), (QWEN_08B, 456)}
    assert any("contract differs" in m["reason"] for m in report["missing_or_failed"])
    assert module.main(["--aggregate", "--source-results-root", str(root), "--output-root", str(output)]) == 1
    subset = module.aggregate(root, output, (QWEN_4B,), (42, 123))
    assert subset["selection_complete"] and not subset["complete"] and subset["expected_runs"] == 2
    assert next(r for r in subset["summary"] if r["evaluation"] == "output_limit_1024")["f1_std"] == 0


def test_cli_rejects_encoder_config_and_low_limit(tmp_path):
    common = ["--source-results-root", str(tmp_path / "source"), "--output-root", str(tmp_path / "out")]
    qwen = str(module.PROJECT_ROOT / "configs/qwen35_27b_3ep.yaml")
    for argv in (["--config", str(module.PROJECT_ROOT / "configs/deberta_large_canonical.yaml")],
                 ["--config", str(module.PROJECT_ROOT / "configs/qwen35_27b.yaml")],
                 ["--config", qwen, "--max-new-tokens", "256"],
                 ["--config", qwen, "--smoke-limit", "0"],
                 ["--config", qwen, "--control-limit", "3"],
                 ["--config", qwen, "--verify-dataset"],
                 ["--aggregate", "--smoke-limit", "2"],
                 ["--aggregate", "--resume-failed"],
                 ["--aggregate", "--models", "deberta-v3-large"],
                 ["--preflight", "--seeds", "7"]):
        with pytest.raises(SystemExit) as info:
            module.main(argv + common)
        assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        module.main(["--help"])
    assert info.value.code == 0


def test_run_lock_recovers_same_requeued_job_and_dead_local_process(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    with module.run_lock(directory):
        assert json.loads((directory / module.LOCK_NAME).read_text())["job_id"] == "123"
        with pytest.raises(RuntimeError, match="locked"):
            with module.run_lock(directory):
                pass
    assert not (directory / module.LOCK_NAME).exists()
    stale = directory / module.LOCK_NAME
    stale.write_text(json.dumps({"pid": 1, "hostname": "elsewhere", "job_id": "123"}))
    with pytest.raises(RuntimeError, match="locked"):
        with module.run_lock(directory):
            pass
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with module.run_lock(directory):  # the same job after a SLURM requeue
        pass
    assert not stale.exists()
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.delenv("SLURM_RESTART_COUNT")
    stale.write_text(json.dumps({"pid": 2 ** 22 + 12345, "hostname": socket.gethostname(), "job_id": None}))
    with pytest.raises(RuntimeError, match="locked"):
        with module.run_lock(directory):
            pass
    with module.run_lock(directory, resume_failed=True):  # dead local process
        pass
    stale.write_text(json.dumps({"pid": 1, "hostname": "elsewhere", "job_id": None}))
    with pytest.raises(RuntimeError, match="remote"):
        with module.run_lock(directory, resume_failed=True):
            pass


def test_preflight_reports_selection_and_refuses_locks(study, monkeypatch, capsys):
    root, output, sources = study
    monkeypatch.setattr(module, "base_model_snapshot", lambda run, source, source_root: f"/snapshots/{run.model_name}")
    report = module.preflight(module.selected_runs(), root, output, verify_dataset=True)
    assert len(report["runs"]) == 9 and all(entry["selected_count"] == 2 and not entry["completed"] for entry in report["runs"])
    assert report["dataset_verified"] and report["runs"][0]["base_model_snapshot"].startswith("/snapshots/Qwen/")
    out = capsys.readouterr().out
    assert "MNOL_PREFLIGHT_OK runs=9" in out and out.count("MNOL_PREFLIGHT ") == 9
    assert module.preflight(module.selected_runs(), root, output / "smoke", smoke_limit=1)["runs"][0]["selected_count"] == 1
    with pytest.raises(ValueError, match="separate path"):
        module.preflight(module.selected_runs(), root, output, smoke_limit=1)
    common = ["--source-results-root", str(root), "--output-root", str(output)]
    assert module.main(["--preflight", "--verify-dataset", *common]) == 0
    experiment = sources[(QWEN_27B, 42)]["config"]["experiment_name"]
    (output / experiment).mkdir(parents=True)
    (output / experiment / module.LOCK_NAME).write_text("{}")
    with pytest.raises(RuntimeError, match="lock"):
        module.preflight(module.selected_runs((QWEN_27B,), (42,)), root, output)
    assert module.main(["--preflight", "--models", QWEN_27B, "--seeds", "42", *common]) == 1
    assert "MNOL_PREFLIGHT_FAILED" in capsys.readouterr().err
    (output / experiment / module.LOCK_NAME).unlink()
    (sources[(QWEN_27B, 42)]["checkpoint"] / "adapter_model.safetensors").write_text("other weights")
    _fake_predictor(monkeypatch, [])
    module.run_evaluation(_run(QWEN_27B, 42), root, output)
    assert module.preflight(module.selected_runs((QWEN_27B,), (42,)), root, output)["runs"][0]["completed"]
    (sources[(QWEN_27B, 42)]["checkpoint"] / "adapter_model.safetensors").write_text("changed weights")
    with pytest.raises(ValueError, match="contract differs"):
        module.preflight(module.selected_runs((QWEN_27B,), (42,)), root, output)
