from __future__ import annotations

import torch
from transformers import BatchEncoding

from src.decoder import generation, inference, train


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["return_dict"] is True
        return BatchEncoding({
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]]),
        })

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return "[]"


class _BatchTokenizer:
    padding_side = "right"

    def __init__(self) -> None:
        self.rendered = []
        self.tokenizer_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tokenize": False,
        }
        rendered = messages[0]["content"]
        self.rendered.append(rendered)
        return rendered

    def __call__(self, prompts, **kwargs):
        self.tokenizer_kwargs = kwargs
        assert prompts == ["short", "long prompt"]
        return BatchEncoding({
            "input_ids": torch.tensor([[0, 4, 5], [6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        })


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


def test_inference_warmup_uses_direct_answer_generation():
    model = _Model()

    inference._warmup(
        model=model,
        tokenizer=_Tokenizer(),
        sample_messages=[{"role": "user", "content": "example"}],
        device=torch.device("cpu"),
        max_new_tokens=256,
    )

    assert set(model.generate_kwargs) >= {"input_ids", "attention_mask"}


def test_generation_batch_inputs_render_without_thinking_and_left_pad():
    tokenizer = _BatchTokenizer()

    result = generation.prepare_generation_batch_inputs(
        tokenizer,
        [
            [{"role": "user", "content": "short"}],
            [{"role": "user", "content": "long prompt"}],
        ],
        torch.device("cpu"),
    )

    assert tokenizer.padding_side == "left"
    assert tokenizer.tokenizer_kwargs == {
        "add_special_tokens": False,
        "padding": True,
        "return_tensors": "pt",
    }
    assert result["input_ids"].shape == (2, 3)


def test_generation_one_item_batch_preserves_original_path():
    tokenizer = _Tokenizer()

    result = generation.prepare_generation_batch_inputs(
        tokenizer,
        [[{"role": "user", "content": "example"}]],
        torch.device("cpu"),
    )

    assert result["input_ids"].tolist() == [[4, 5]]
