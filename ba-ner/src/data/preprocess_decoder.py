"""
Preprocessing for decoder-based NER (generative, JSON output format).

The LLM receives a system prompt + user message (the raw sentence) and is
expected to output a JSON list of entity dicts:
    [{"entity": "Barack Obama", "type": "person"}, ...]

Usage:
    from src.data.preprocess_decoder import prepare_decoder_dataset
    dataset = prepare_decoder_dataset()
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datasets import DatasetDict, Dataset

from src.data.load_wnut17 import ID2LABEL, LABEL_LIST, load_wnut17

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "Du bist ein NER-System. Extrahiere alle Named Entities aus dem gegebenen Text.\n"
    "Gib das Ergebnis als JSON-Liste zurück. Jedes Element hat die Felder "
    "\"entity\" (der Text) und \"type\" (einer von: person, location, corporation, "
    "creative-work, group, product).\n"
    "Wenn keine Entities vorhanden sind, gib eine leere Liste [] zurück.\n"
    "Antworte NUR mit dem JSON, ohne zusätzlichen Text."
)

# ---------------------------------------------------------------------------
# BIO → entity list conversion
# ---------------------------------------------------------------------------


def extract_entities_from_bio(
    tokens: List[str],
    ner_tags: List[int],
) -> List[Dict[str, str]]:
    """Convert BIO integer tags into a list of entity dicts.

    Parameters
    ----------
    tokens:
        Word tokens of the sentence.
    ner_tags:
        Corresponding integer BIO tags (using ID2LABEL mapping).

    Returns
    -------
    List[Dict[str, str]]
        Each dict has keys ``entity`` (surface string) and ``type``.

    Examples
    --------
    >>> extract_entities_from_bio(["New", "York", "is", "great"], [7, 8, 0, 0])
    [{"entity": "New York", "type": "location"}]
    """
    entities: List[Dict[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, ner_tags):
        label = ID2LABEL[tag]
        if label.startswith("B-"):
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type is not None:
            current_tokens.append(token)
        else:
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []

    if current_type is not None:
        entities.append({"entity": " ".join(current_tokens), "type": current_type})

    return entities


# ---------------------------------------------------------------------------
# Chat format builder
# ---------------------------------------------------------------------------


def format_for_llm(sample: Dict) -> Dict:
    """Convert a WNUT-17 sample into chat-format messages for SFT.

    Produces a ``messages`` field (list of role/content dicts) following the
    ChatML convention (compatible with Qwen).  The assistant turn contains the
    gold JSON.

    Parameters
    ----------
    sample:
        Dataset row with ``tokens`` and ``ner_tags`` fields.

    Returns
    -------
    Dict
        Original sample augmented with a ``messages`` key.
    """
    tokens: List[str] = sample["tokens"]
    ner_tags: List[int] = sample["ner_tags"]

    sentence: str = " ".join(tokens)
    entities: List[Dict[str, str]] = extract_entities_from_bio(tokens, ner_tags)
    assistant_answer: str = json.dumps(entities, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
        {"role": "assistant", "content": assistant_answer},
    ]

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_decoder_dataset() -> DatasetDict:
    """Load WNUT-17 and convert all splits to chat-format for SFT.

    Returns
    -------
    DatasetDict
        Dataset with splits train/validation/test, each containing a
        ``messages`` column.
    """
    raw: DatasetDict = load_wnut17()

    formatted: DatasetDict = raw.map(
        format_for_llm,
        remove_columns=raw["train"].column_names,
    )
    return formatted


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def prepare_test_inputs(
    dataset_split: Dataset,
) -> Tuple[List[List[Dict]], List[List[Dict[str, str]]]]:
    """Build prompt-only messages (no assistant turn) and gold entities.

    Used during inference: the model should predict the assistant turn.

    Parameters
    ----------
    dataset_split:
        A single split of the raw WNUT-17 dataset (with ``tokens`` and
        ``ner_tags`` columns).

    Returns
    -------
    Tuple[List[List[Dict]], List[List[Dict[str, str]]]]
        ``(prompts, gold_entities)``

        * ``prompts``: list of message lists [system, user] — no assistant
        * ``gold_entities``: list of entity-dict lists (ground truth)
    """
    prompts: List[List[Dict]] = []
    gold_entities: List[List[Dict[str, str]]] = []

    for sample in dataset_split:
        tokens: List[str] = sample["tokens"]
        ner_tags: List[int] = sample["ner_tags"]
        sentence: str = " ".join(tokens)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ]
        prompts.append(messages)
        gold_entities.append(extract_entities_from_bio(tokens, ner_tags))

    return prompts, gold_entities
