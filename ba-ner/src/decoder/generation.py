"""Shared helpers for deterministic, structured decoder generation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


THINKING_ENABLED = False


def _render_generation_prompt(
    tokenizer,
    messages: List[Dict[str, Any]],
) -> str:
    """Render one prompt without tokenizing it a second time."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=THINKING_ENABLED,
        tokenize=False,
    )


def prepare_generation_inputs(
    tokenizer,
    messages: List[Dict[str, Any]],
    device,
):
    """Render a direct-answer prompt with Qwen thinking explicitly disabled.

    Qwen3.5 chat-template defaults vary by model size. Passing the flag
    explicitly keeps generative validation and inference aligned with the
    supervised JSON targets used during training.
    """
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=THINKING_ENABLED,
        return_tensors="pt",
        return_dict=True,
    ).to(device)


def prepare_generation_batch_inputs(
    tokenizer,
    messages_batch: Sequence[List[Dict[str, Any]]],
    device,
):
    """Tokenize a left-padded batch of direct-answer chat prompts.

    Decoder-only generation must use left padding: generated tokens are
    appended after the shared padded prompt width for every row. A one-item
    batch deliberately follows the original single-sample path so existing
    batch-size-one runs remain byte-for-byte compatible.
    """
    if not messages_batch:
        raise ValueError("messages_batch must contain at least one prompt")
    if len(messages_batch) == 1:
        return prepare_generation_inputs(tokenizer, messages_batch[0], device)

    rendered_prompts = [
        _render_generation_prompt(tokenizer, messages)
        for messages in messages_batch
    ]
    tokenizer.padding_side = "left"
    return tokenizer(
        rendered_prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(device)
