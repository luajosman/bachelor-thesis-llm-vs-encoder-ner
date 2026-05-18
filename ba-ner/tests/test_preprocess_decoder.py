from __future__ import annotations

from src.data.preprocess_decoder import (
    build_system_prompt,
    extract_entities_from_bio,
    format_for_llm,
    format_numbered_tokens,
)


def test_format_numbered_tokens_matches_prompt_contract() -> None:
    assert format_numbered_tokens(["Barack", "Obama", "."]) == "Tokens:\n0: Barack\n1: Obama\n2: ."


def test_extract_entities_from_bio_returns_token_offsets() -> None:
    tokens = ["Barack", "Obama", "visited", "Berlin", "."]
    id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC"}
    ner_tags = [1, 2, 0, 3, 0]

    assert extract_entities_from_bio(tokens, ner_tags, id2label) == [
        {"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PER"},
        {"start_token": 3, "end_token": 4, "text": "Berlin", "type": "LOC"},
    ]


def test_format_for_llm_uses_numbered_tokens_and_offset_json() -> None:
    sample = {
        "tokens": ["Barack", "Obama", "visited", "Berlin", "."],
        "ner_tags": [1, 2, 0, 3, 0],
    }
    id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC"}
    system_prompt = build_system_prompt(["PER", "LOC"])

    result = format_for_llm(sample, system_prompt, id2label)

    assert result["messages"][0]["role"] == "system"
    assert "start_token" in result["messages"][0]["content"]
    assert result["messages"][1] == {
        "role": "user",
        "content": "Tokens:\n0: Barack\n1: Obama\n2: visited\n3: Berlin\n4: .",
    }
    assert result["messages"][2]["content"] == (
        '[{"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PER"}, '
        '{"start_token": 3, "end_token": 4, "text": "Berlin", "type": "LOC"}]'
    )
