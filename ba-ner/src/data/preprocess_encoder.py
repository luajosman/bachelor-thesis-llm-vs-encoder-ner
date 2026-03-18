"""
Preprocessing for encoder-based NER (token classification with BIO tagging).

Usage:
    from src.data.preprocess_encoder import prepare_encoder_dataset
    tokenized_dataset, tokenizer, label_list = prepare_encoder_dataset("microsoft/deberta-v3-large")
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.load_wnut17 import LABEL_LIST, LABEL2ID, load_wnut17


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


def tokenize_and_align_labels(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> Dict:
    """Tokenize a batch of word-tokenized sentences and align BIO labels.

    Sub-word tokens beyond the first receive label ``-100`` so that the loss
    function (and seqeval) ignores them.  Special tokens ([CLS], [SEP],
    padding) also get ``-100``.

    Parameters
    ----------
    examples:
        Batch dict with keys ``tokens`` (List[List[str]]) and
        ``ner_tags`` (List[List[int]]).
    tokenizer:
        HuggingFace tokenizer, should be created with ``add_prefix_space=True``
        for models that require it (e.g. DeBERTa, RoBERTa).
    max_length:
        Maximum token length for truncation.

    Returns
    -------
    Dict
        Tokenized batch with ``input_ids``, ``attention_mask``, ``labels``,
        and (if applicable) ``token_type_ids``.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels: List[List[int]] = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx: int | None = None
        label_ids: List[int] = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special token — ignore
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-word of a new word — use real label
                label_ids.append(labels[word_idx])
            else:
                # Continuation sub-word — ignore
                label_ids.append(-100)
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_encoder_dataset(
    model_name: str,
    max_length: int = 128,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase, List[str]]:
    """Load WNUT-17, tokenize and align labels for encoder token classification.

    Parameters
    ----------
    model_name:
        HuggingFace model name or path (e.g. ``"microsoft/deberta-v3-large"``).
    max_length:
        Maximum sequence length for truncation.

    Returns
    -------
    Tuple[DatasetDict, PreTrainedTokenizerBase, List[str]]
        ``(tokenized_dataset, tokenizer, LABEL_LIST)``
    """
    dataset: DatasetDict = load_wnut17()

    # DeBERTa / RoBERTa need add_prefix_space=True for correct tokenization
    # of word-split inputs.  For BERT it is harmless.
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        use_fast=True,
    )

    tokenized_dataset: DatasetDict = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer, LABEL_LIST
