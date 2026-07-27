"""Shared helpers for deterministic, structured decoder generation."""

from __future__ import annotations

from typing import Any, Dict, List


THINKING_ENABLED = False


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
