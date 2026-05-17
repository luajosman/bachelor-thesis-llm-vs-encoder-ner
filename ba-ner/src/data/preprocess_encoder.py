"""Encoder preprocessing for MultiNERD English token classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.dataset_loader import DatasetInfo, load_ner_dataset

if TYPE_CHECKING:
    from datasets import DatasetDict


# ---------------------------------------------------------------------------
# Label-Alignment: Subword-Tokens <-> BIO-Labels
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> Dict:
    """Tokenisiert eine Batch von Saetzen und richtet die BIO-Labels aus.

    Das Alignment ist der kritische Schritt beim Encoder-Preprocessing:
    Ein Wort wie "London" wird z.B. zu ["Lon", "##don"] aufgeteilt.
    "Lon" bekommt das echte Label "B-location", "##don" bekommt -100.

    Warum -100? HuggingFace's CrossEntropyLoss ignoriert Positionen mit
    label_id == -100 automatisch. Seqeval filtert sie ebenfalls heraus.

    Args:
        examples:   Batch-Dict mit Schluesseln 'tokens' und 'ner_tags'.
        tokenizer:  HuggingFace-Tokenizer (muss mit use_fast=True geladen sein,
                    damit word_ids() verfuegbar ist).
        max_length: Maximale Sequenzlaenge; laengere Saetze werden abgeschnitten.

    Returns:
        Dict mit 'input_ids', 'attention_mask', 'labels' (und ggf. 'token_type_ids').
    """
    # Tokenisierung: is_split_into_words=True, weil der Datensatz
    # bereits wortweise vorliegt (Liste von Strings pro Satz)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels: List[List[int]] = []

    for i, labels in enumerate(examples["ner_tags"]):
        # word_ids() gibt fuer jedes Subword-Token den Index des
        # urspruenglichen Wortes zurueck; None steht fuer Sondertokens.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx: int | None = None
        label_ids: List[int] = []

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], Padding → wird beim Training ignoriert
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Erstes Subword-Token eines neuen Wortes → echtes Label
                label_ids.append(labels[word_idx])
            else:
                # Weiteres Subword-Token desselben Wortes → ignorieren
                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    # Labels dem tokenisierten Dict hinzufuegen
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ---------------------------------------------------------------------------
# Datensatz-Vorbereitung
# ---------------------------------------------------------------------------

def prepare_encoder_dataset(
    model_name: str,
    max_length: int = 256,
) -> Tuple["DatasetDict", PreTrainedTokenizerBase, DatasetInfo]:
    """Load MultiNERD English, tokenize all splits, and align word-level labels.

    Only the first subword of each original word receives the BIO label.
    Continuation subwords and special tokens are set to -100.
    """
    dataset, info = load_ner_dataset()

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        use_fast=True,
    )

    tokenized_dataset: "DatasetDict" = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer, info
