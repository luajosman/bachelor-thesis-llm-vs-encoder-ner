from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.decoder import train
from src.decoder.train import (
    GENERATIVE_EVAL_STATE_FILENAME,
    GenerativeDevEvalCallback,
)


class _SavingObject:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.save_calls = 0

    def save_pretrained(self, output_dir: str) -> None:
        self.save_calls += 1
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / self.filename).write_text("saved", encoding="utf-8")


def _callback(tmp_path: Path) -> GenerativeDevEvalCallback:
    return GenerativeDevEvalCallback(
        tokenizer=_SavingObject("tokenizer.json"),
        dev_prompts=[],
        dev_gold_entities=[],
        dev_tokens=[],
        valid_types=frozenset(),
        output_dir=tmp_path,
    )


def test_eval_state_survives_callback_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback(tmp_path)
    model = _SavingObject("adapter_model.safetensors")
    callback.set_trainer(SimpleNamespace(model=model))
    monkeypatch.setattr(
        train,
        "_run_generative_eval",
        lambda **kwargs: {
            "f1": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "parse_failure_rate": 0.05,
        },
    )

    callback.on_evaluate(None, SimpleNamespace(epoch=1.0), None)

    state_file = tmp_path / GENERATIVE_EVAL_STATE_FILENAME
    persisted = yaml.safe_load(state_file.read_text(encoding="utf-8"))
    assert persisted == {
        "version": 1,
        "best_f1": 0.75,
        "best_epoch": 1,
        "epoch_results": [{
            "epoch": 1,
            "dev_f1": 0.75,
            "dev_precision": 0.8,
            "dev_recall": 0.7,
            "parse_failure_rate": 0.05,
        }],
    }
    assert not list(tmp_path.glob(f".{GENERATIVE_EVAL_STATE_FILENAME}.*.tmp"))

    restarted = _callback(tmp_path)
    assert restarted.best_f1 == 0.75
    assert restarted.best_epoch == 1
    assert restarted.epoch_results == persisted["epoch_results"]

    restarted.set_trainer(SimpleNamespace(model=_SavingObject("adapter_model.safetensors")))
    monkeypatch.setattr(
        train,
        "_run_generative_eval",
        lambda **kwargs: {
            "f1": 0.7,
            "precision": 0.72,
            "recall": 0.68,
            "parse_failure_rate": 0.1,
        },
    )
    restarted.on_evaluate(None, SimpleNamespace(epoch=2.0), None)

    restarted_again = _callback(tmp_path)
    assert restarted_again.best_f1 == 0.75
    assert restarted_again.best_epoch == 1
    assert [result["epoch"] for result in restarted_again.epoch_results] == [1, 2]


def test_restarted_callback_skips_an_already_persisted_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback(tmp_path)
    callback.best_f1 = 0.75
    callback.best_epoch = 1
    callback.epoch_results = [{
        "epoch": 1,
        "dev_f1": 0.75,
        "dev_precision": 0.8,
        "dev_recall": 0.7,
        "parse_failure_rate": 0.05,
    }]
    callback._save_state()

    restarted = _callback(tmp_path)
    model = _SavingObject("adapter_model.safetensors")
    restarted.set_trainer(SimpleNamespace(model=model))

    def unexpected_eval(**kwargs):
        raise AssertionError("persisted epoch must not be evaluated again")

    monkeypatch.setattr(train, "_run_generative_eval", unexpected_eval)
    restarted.on_evaluate(None, SimpleNamespace(epoch=1.0), None)

    assert model.save_calls == 0
    assert len(restarted.epoch_results) == 1


@pytest.mark.parametrize(
    "contents",
    [
        "not: [valid",
        """
version: 1
best_f1: .nan
best_epoch: 1
epoch_results:
  - epoch: 1
    dev_f1: .nan
    dev_precision: 0.8
    dev_recall: 0.7
    parse_failure_rate: 0.0
""",
    ],
)
def test_corrupt_eval_state_is_ignored_conservatively(
    tmp_path: Path,
    contents: str,
) -> None:
    state_file = tmp_path / GENERATIVE_EVAL_STATE_FILENAME
    state_file.write_text(contents, encoding="utf-8")

    callback = _callback(tmp_path)

    assert callback.best_f1 == -1.0
    assert callback.best_epoch == -1
    assert callback.epoch_results == []
    assert state_file.read_text(encoding="utf-8") == contents
