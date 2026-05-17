from __future__ import annotations

import pytest

from src.decoder.parse_output import (
    entities_to_bio,
    evaluate_llm_predictions,
    parse_llm_output_with_diagnostics,
)


VALID_TYPES = frozenset({"PER", "LOC"})


def test_parse_direct_json_and_normalize_entity_type() -> None:
    entities, status, diagnostics = parse_llm_output_with_diagnostics(
        '[{"entity": "Barack Obama", "type": "per"}]',
        VALID_TYPES,
    )

    assert status == "ok"
    assert entities == [{"entity": "Barack Obama", "type": "PER"}]
    assert diagnostics == {
        "invalid_items": 0,
        "missing_fields": 0,
        "unknown_types": 0,
        "wrong_schema": 0,
    }


def test_parse_markdown_json_reports_diagnostics_before_filtering() -> None:
    output = """```json
[
  {"entity": "Paris", "type": "LOC"},
  {"entity": "Atlantis", "type": "MYTH"},
  {"entity": "", "type": "LOC"},
  {"foo": "bar"},
  7
]
```"""

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, VALID_TYPES)

    assert status == "markdown_stripped"
    assert entities == [{"entity": "Paris", "type": "LOC"}]
    assert diagnostics["unknown_types"] == 1
    assert diagnostics["missing_fields"] == 2
    assert diagnostics["invalid_items"] == 1


def test_entities_to_bio_and_llm_metrics_are_strict_span_comparable() -> None:
    pytest.importorskip("seqeval")

    tokens = [["Barack", "Obama", "visited", "Paris"]]
    gold = [[
        {"entity": "Barack Obama", "type": "PER"},
        {"entity": "Paris", "type": "LOC"},
    ]]
    pred = [[
        {"entity": "Barack Obama", "type": "PER"},
        {"entity": "Paris", "type": "LOC"},
    ]]

    assert entities_to_bio(tokens[0], pred[0]) == ["B-PER", "I-PER", "O", "B-LOC"]

    metrics = evaluate_llm_predictions(
        tokens_list=tokens,
        gold_entities=gold,
        pred_entities=pred,
        parse_statuses=["ok"],
        parse_diagnostics=[{
            "invalid_items": 0,
            "missing_fields": 0,
            "unknown_types": 0,
            "wrong_schema": 0,
        }],
    )

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["parse_failure_rate"] == 0.0
