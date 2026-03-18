"""
preprocess_encoder.py — Vorverarbeitung für Encoder-basierte NER

Für die Token-Klassifikation (BERT, DeBERTa) muss jedes Wort-Token
in Subword-Tokens aufgeteilt und das BIO-Label korrekt übertragen werden.

Wichtige Designentscheidung:
    - Nur das erste Subword-Token eines Wortes erhält das echte Label.
    - Alle weiteren Subword-Tokens (und Sondertokens wie [CLS], [SEP])
      bekommen -100, damit sie vom Loss und von seqeval ignoriert werden.

Verwendung:
    from src.data.preprocess_encoder import prepare_encoder_dataset
    tokenized_ds, tokenizer, label_list = prepare_encoder_dataset("microsoft/deberta-v3-large")
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.load_wnut17 import LABEL_LIST, LABEL2ID, load_wnut17


# ---------------------------------------------------------------------------
# Label-Alignment: Subword-Tokens ↔ BIO-Labels
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> Dict:
    """Tokenisiert eine Batch von Sätzen und richtet die BIO-Labels aus.

    Das Alignment ist der kritische Schritt beim Encoder-Preprocessing:
    Ein Wort wie "London" wird z.B. zu ["Lon", "##don"] aufgeteilt.
    "Lon" bekommt das echte Label "B-location", "##don" bekommt -100.

    Warum -100? HuggingFace's CrossEntropyLoss ignoriert Positionen mit
    label_id == -100 automatisch. Seqeval filtert sie ebenfalls heraus.

    Args:
        examples:   Batch-Dict mit Schlüsseln 'tokens' und 'ner_tags'.
        tokenizer:  HuggingFace-Tokenizer (muss mit use_fast=True geladen sein,
                    damit word_ids() verfügbar ist).
        max_length: Maximale Sequenzlänge; längere Sätze werden abgeschnitten.

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
        # word_ids() gibt für jedes Subword-Token den Index des
        # ursprünglichen Wortes zurück; None steht für Sondertokens.
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

    # Labels dem tokenisierten Dict hinzufügen
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ---------------------------------------------------------------------------
# Datensatz-Vorbereitung
# ---------------------------------------------------------------------------

def prepare_encoder_dataset(
    model_name: str,
    max_length: int = 128,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase, List[str]]:
    """Lädt WNUT-2017, tokenisiert alle Splits und richtet Labels aus.

    Der Tokenizer wird mit add_prefix_space=True geladen, weil DeBERTa und
    RoBERTa beim Verarbeiten von vorher tokenisierten Wörtern ein führendes
    Leerzeichen erwarten. Für BERT ist die Option harmlos.

    Args:
        model_name: HuggingFace Model-ID (z.B. 'microsoft/deberta-v3-large').
        max_length: Maximale Sequenzlänge für den Tokenizer.

    Returns:
        Tuple aus (tokenized_dataset, tokenizer, LABEL_LIST).
    """
    # Rohen Datensatz von HuggingFace laden
    dataset: DatasetDict = load_wnut17()

    # Tokenizer laden; add_prefix_space für DeBERTa/RoBERTa-Kompatibilität
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        use_fast=True,  # use_fast=True wird für word_ids() benötigt
    )

    # Tokenisierung und Label-Alignment über alle Splits mappen
    # batched=True beschleunigt die Verarbeitung erheblich
    tokenized_dataset: DatasetDict = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length),
        batched=True,
        # Originalspalten entfernen, da sie durch Tokenisierung ersetzt werden
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer, LABEL_LIST
