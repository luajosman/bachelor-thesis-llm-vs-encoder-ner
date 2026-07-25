from __future__ import annotations

import torch
from transformers import BatchEncoding

from src.decoder import train


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["return_dict"] is True
        return BatchEncoding({
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]]),
        })

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return "[]"


class _Model:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.generate_kwargs = None

    def parameters(self):
        yield self.parameter

    def eval(self):
        return self

    def train(self):
        return self

    def generate(self, *args, **kwargs):
        assert not args
        self.generate_kwargs = kwargs
        return torch.tensor([[4, 5, 6]])


def test_generative_eval_passes_batch_encoding_as_keyword_inputs(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        train,
        "parse_llm_output_with_diagnostics",
        lambda *args: ([], "ok", {}),
    )
    monkeypatch.setattr(
        train,
        "evaluate_llm_predictions",
        lambda **kwargs: {"f1": 0.0},
    )

    metrics = train._run_generative_eval(
        model=model,
        tokenizer=_Tokenizer(),
        prompts=[[{"role": "user", "content": "example"}]],
        gold_entities=[[]],
        tokens_list=[["example"]],
        valid_types=frozenset(),
        max_samples=1,
    )

    assert metrics == {"f1": 0.0}
    assert set(model.generate_kwargs) >= {"input_ids", "attention_mask"}
