from __future__ import annotations

import pytest

from src.decoder.parse_output import (
    entities_to_bio,
    evaluate_llm_predictions,
    parse_llm_output_with_diagnostics,
)


VALID_TYPES = frozenset({"PER", "LOC"})


def test_parse_direct_json_and_normalize_entity_type() -> None:
    tokens = ["Barack", "Obama", "visited", "Berlin", "."]

    entities, status, diagnostics = parse_llm_output_with_diagnostics(
        '[{"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "per"}]',
        tokens,
        VALID_TYPES,
    )

    assert status == "ok"
    assert entities == [{
        "start_token": 0,
        "end_token": 2,
        "text": "Barack Obama",
        "type": "PER",
    }]
    assert diagnostics == {
        "invalid_items": 0,
        "missing_fields": 0,
        "unknown_types": 0,
        "wrong_schema": 0,
        "invalid_offsets": 0,
        "text_mismatches": 0,
        "overlaps": 0,
    }


def test_parse_markdown_json_reports_diagnostics_before_filtering() -> None:
    tokens = ["Paris", "is", "old", "."]
    output = """```json
[
  {"start_token": 0, "end_token": 1, "text": "Paris", "type": "LOC"},
  {"start_token": 0, "end_token": 1, "text": "Paris", "type": "MYTH"},
  {"start_token": 0, "end_token": 1, "text": "", "type": "LOC"},
  {"foo": "bar"},
  7
]
```"""

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "markdown_stripped"
    assert entities == [{"start_token": 0, "end_token": 1, "text": "Paris", "type": "LOC"}]
    assert diagnostics["unknown_types"] == 1
    assert diagnostics["missing_fields"] == 2
    assert diagnostics["invalid_items"] == 1


def test_entities_to_bio_and_llm_metrics_are_strict_span_comparable() -> None:
    pytest.importorskip("seqeval")

    tokens = [["Barack", "Obama", "visited", "Paris"]]
    gold = [[
        {"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PER"},
        {"start_token": 3, "end_token": 4, "text": "Paris", "type": "LOC"},
    ]]
    pred = [[
        {"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PER"},
        {"start_token": 3, "end_token": 4, "text": "Paris", "type": "LOC"},
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
            "invalid_offsets": 0,
            "text_mismatches": 0,
            "overlaps": 0,
        }],
    )

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["parse_failure_rate"] == 0.0


def test_empty_entity_list_produces_only_o_tags() -> None:
    tokens = ["No", "entities", "."]

    entities, status, diagnostics = parse_llm_output_with_diagnostics("[]", tokens, VALID_TYPES)

    assert status == "ok"
    assert entities == []
    assert diagnostics["wrong_schema"] == 0
    assert entities_to_bio(tokens, entities) == ["O", "O", "O"]


def test_repeated_text_is_resolved_by_offsets_not_string_matching() -> None:
    tokens = ["Berlin", "is", "larger", "than", "Berlin", "."]
    output = '[{"start_token": 4, "end_token": 5, "text": "Berlin", "type": "LOC"}]'

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "ok"
    assert diagnostics["text_mismatches"] == 0
    assert entities_to_bio(tokens, entities) == ["O", "O", "O", "O", "B-LOC", "O"]


def test_multiword_entity_uses_bio_continuation_tags() -> None:
    tokens = ["New", "York", "City", "is", "large", "."]
    entities = [{"start_token": 0, "end_token": 3, "text": "New York City", "type": "LOC"}]

    assert entities_to_bio(tokens, entities) == ["B-LOC", "I-LOC", "I-LOC", "O", "O", "O"]


@pytest.mark.parametrize(
    "output",
    [
        '[{"start_token": -1, "end_token": 1, "text": "Berlin", "type": "LOC"}]',
        '[{"start_token": 0, "end_token": 9, "text": "Berlin", "type": "LOC"}]',
        '[{"start_token": 1, "end_token": 1, "text": "Berlin", "type": "LOC"}]',
        '[{"start_token": true, "end_token": 1, "text": "Berlin", "type": "LOC"}]',
    ],
)
def test_invalid_offsets_are_discarded(output: str) -> None:
    tokens = ["Berlin", "."]

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "ok"
    assert entities == []
    assert diagnostics["invalid_offsets"] == 1


def test_invalid_label_is_discarded() -> None:
    tokens = ["Barack", "Obama"]
    output = '[{"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PERSON"}]'

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "ok"
    assert entities == []
    assert diagnostics["unknown_types"] == 1


def test_invalid_json_or_wrong_top_level_fails_cleanly() -> None:
    tokens = ["Berlin"]

    bad_json_entities, bad_json_status, _ = parse_llm_output_with_diagnostics(
        '[{"start_token": 0',
        tokens,
        VALID_TYPES,
    )
    wrong_schema_entities, wrong_schema_status, wrong_schema_diag = parse_llm_output_with_diagnostics(
        '{"start_token": 0, "end_token": 1, "text": "Berlin", "type": "LOC"}',
        tokens,
        VALID_TYPES,
    )

    assert bad_json_entities == []
    assert bad_json_status == "failed"
    assert wrong_schema_entities == []
    assert wrong_schema_status == "failed"
    assert wrong_schema_diag["wrong_schema"] == 1


def test_text_mismatch_is_discarded() -> None:
    tokens = ["Barack", "Obama"]
    output = '[{"start_token": 0, "end_token": 2, "text": "Obama", "type": "PER"}]'

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "ok"
    assert entities == []
    assert diagnostics["text_mismatches"] == 1


def test_overlapping_spans_are_resolved_deterministically() -> None:
    tokens = ["New", "York", "City"]
    output = """[
      {"start_token": 0, "end_token": 2, "text": "New York", "type": "LOC"},
      {"start_token": 1, "end_token": 3, "text": "York City", "type": "LOC"}
    ]"""

    entities, status, diagnostics = parse_llm_output_with_diagnostics(output, tokens, VALID_TYPES)

    assert status == "ok"
    assert entities == [{"start_token": 0, "end_token": 2, "text": "New York", "type": "LOC"}]
    assert diagnostics["overlaps"] == 1
    assert entities_to_bio(tokens, entities) == ["B-LOC", "I-LOC", "O"]
