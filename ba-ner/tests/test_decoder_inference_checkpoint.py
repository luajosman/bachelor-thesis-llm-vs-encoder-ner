import json

import pytest

from src.decoder.inference import (
    _load_inference_checkpoint,
    _write_inference_checkpoint,
)


def _header():
    return {
        "version": 1,
        "experiment_name": "test",
        "model_name": "model",
        "regime": "llm_zeroshot",
        "thinking_enabled": False,
        "max_new_tokens": 256,
        "seed": 42,
        "total_samples": 2,
    }


def _record(index):
    return {
        "index": index,
        "pred_entities": [],
        "raw_output": "[]",
        "parse_status": "ok",
        "parse_diagnostics": {},
        "latency_ms": 12.5,
        "elapsed_seconds": 20.0 + index,
    }


def test_inference_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    records = [_record(0), _record(1)]

    _write_inference_checkpoint(path, _header(), records)

    assert _load_inference_checkpoint(path, _header()) == records


def test_inference_checkpoint_ignores_partial_trailing_record(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    records = [_record(0)]
    _write_inference_checkpoint(path, _header(), records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"index": 1, "pred_entities":')

    loaded = _load_inference_checkpoint(path, _header())
    _write_inference_checkpoint(path, _header(), loaded)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert loaded == records
    assert len(lines) == 2
    assert json.loads(lines[1]) == records[0]


def test_inference_checkpoint_rejects_different_run(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    _write_inference_checkpoint(path, _header(), [])
    other_header = {**_header(), "thinking_enabled": True}

    with pytest.raises(RuntimeError, match="does not match"):
        _load_inference_checkpoint(path, other_header)
