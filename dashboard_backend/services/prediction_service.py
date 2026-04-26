"""Prediction loading and normalization across encoder and LLM outputs."""

from __future__ import annotations

from typing import Any

from dashboard_backend.models.schemas import PredictionSample
from dashboard_backend.utils.paths import DEFAULT_DATASET, experiment_dir
from dashboard_backend.utils.yaml_json import load_json_file


def _bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    start: int | None = None
    current_type: str | None = None

    def _close(end: int) -> None:
        nonlocal start, current_type
        if start is None or current_type is None:
            return
        entities.append(
            {
                "entity": " ".join(tokens[start:end]),
                "type": current_type,
                "start": start,
                "end": end,
            }
        )
        start = None
        current_type = None

    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            _close(index)
            start = index
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            _close(index)
    _close(len(tags))
    return entities


def _normalize_entities(tokens: list[str], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    sentence = " ".join(tokens)
    for entity in entities:
        record = dict(entity)
        record.setdefault("entity", str(record.get("text", record.get("entity", ""))))
        record.setdefault("type", str(record.get("label", record.get("type", ""))))
        if "start" not in record or "end" not in record:
            text = str(record.get("entity", ""))
            if text and text in sentence:
                token_parts = text.split()
                for index in range(len(tokens)):
                    if tokens[index:index + len(token_parts)] == token_parts:
                        record["start"] = index
                        record["end"] = index + len(token_parts)
                        break
        normalized.append(record)
    return normalized


def _normalize_sample(index: int, raw_sample: dict[str, Any]) -> PredictionSample:
    tokens = [str(token) for token in raw_sample.get("tokens", [])]
    gold_bio = [str(tag) for tag in raw_sample.get("gold_bio", raw_sample.get("gold", []))]
    pred_bio = [str(tag) for tag in raw_sample.get("pred_bio", raw_sample.get("pred", []))]

    gold_entities_raw = raw_sample.get("gold_entities")
    pred_entities_raw = raw_sample.get("pred_entities")

    gold_entities = _normalize_entities(tokens, gold_entities_raw) if isinstance(gold_entities_raw, list) else _bio_to_entities(tokens, gold_bio)
    pred_entities = _normalize_entities(tokens, pred_entities_raw) if isinstance(pred_entities_raw, list) else _bio_to_entities(tokens, pred_bio)

    return PredictionSample(
        sample_id=str(raw_sample.get("sample_id", index)),
        tokens=tokens,
        gold_bio=gold_bio,
        pred_bio=pred_bio,
        gold_entities=gold_entities,
        pred_entities=pred_entities,
        raw_output=raw_sample.get("raw_output"),
        parse_status=raw_sample.get("parse_status"),
    )


def list_predictions(experiment_id: str, dataset: str = DEFAULT_DATASET) -> list[PredictionSample]:
    """Load normalized predictions for one experiment."""
    pred_path = experiment_dir(experiment_id, dataset) / "test_predictions.json"
    raw_predictions = load_json_file(pred_path, default=[])
    if not isinstance(raw_predictions, list):
        return []
    return [_normalize_sample(index, sample if isinstance(sample, dict) else {}) for index, sample in enumerate(raw_predictions)]


def get_prediction(experiment_id: str, sample_id: str, dataset: str = DEFAULT_DATASET) -> PredictionSample | None:
    """Return one normalized prediction sample by id."""
    for sample in list_predictions(experiment_id, dataset):
        if sample.sample_id == sample_id:
            return sample
    return None
