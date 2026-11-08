"""CPU-only tests for the blind check sample (span extraction, evaluation statistics)."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("check_sample", Path(__file__).resolve().parents[1] / "scripts/check_sample.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def test_strict_iob2_spans_ignore_orphan_inside_tags():
    tags = ["B-PER", "I-PER", "O", "I-LOC", "B-ANIM", "B-ANIM", "I-ANIM", "I-PER"]
    assert check.spans(tags) == {(0, 2, "PER"), (4, 5, "ANIM"), (5, 7, "ANIM")}


def test_case_region_covers_overlapping_alternative():
    row = {"gold": {(0, 1, "PLANT")}, "enc": {(0, 1, "PLANT")}, "llm": {(0, 2, "PLANT")}, "regenerated": False}
    case = check.make_case(3, row, "S1", (0, 1, "PLANT"))
    assert case["region"] == [0, 2] and case["reference"] == [[0, 1, "PLANT"]] and case["model"] == [[0, 2, "PLANT"]]


def test_wilson_and_kappa():
    low, high = check.wilson(30, 60)
    assert low == pytest.approx(0.3773, abs=1e-3) and high == pytest.approx(0.6227, abs=1e-3)
    assert check.cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert check.cohen_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)


def test_evaluate_maps_blind_judgments_to_reference_or_model(tmp_path, monkeypatch):
    key = {"F001": {"stratum": "S1", "type_group": "noisy", "reference_is": "A", "repeat_of": None},
           "F002": {"stratum": "S3", "type_group": "other", "reference_is": "B", "repeat_of": None},
           "W01": {"stratum": "S3", "type_group": "other", "reference_is": "A", "repeat_of": "F002", "swapped": True}}
    monkeypatch.setattr(check, "OUT", tmp_path)
    (tmp_path / "key.json").write_text(json.dumps(key))
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"F001": "A", "F002": "A", "W01": "B"}))
    check.evaluate(str(annotations))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["strata"]["S1"]["reference_correct"]["count"] == 1
    assert summary["strata"]["S3"]["model_correct"]["count"] == 1
    assert summary["intra_rater"] == {"pairs": 1, "agreement": 1.0, "cohen_kappa": 1.0}
