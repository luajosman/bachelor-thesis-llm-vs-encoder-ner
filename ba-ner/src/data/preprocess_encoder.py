"""
preprocess_encoder.py — Vorverarbeitung fuer Encoder-basierte NER

Fuer die Token-Klassifikation (DeBERTa) muss jedes Wort-Token
in Subword-Tokens aufgeteilt und das BIO-Label korrekt uebertragen werden.

Wichtige Designentscheidung:
    - Nur das erste Subword-Token eines Wortes erhaelt das echte Label.
    - Alle weiteren Subword-Tokens (und Sondertokens wie [CLS], [SEP])
      bekommen -100, damit sie vom Loss und von seqeval ignoriert werden.

Verwendung:
    from src.data.preprocess_encoder import prepare_encoder_dataset
    tokenized_ds, tokenizer, info = prepare_encoder_dataset(
        model_name="microsoft/deberta-v3-base",
        dataset_name="multinerd",
    )
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.dataset_loader import DatasetInfo, load_ner_dataset


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
    dataset_name: str = "multinerd",
    dataset_language: str = "en",
    max_length: int = 256,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase, DatasetInfo]:
    """Laedt einen NER-Datensatz, tokenisiert alle Splits und richtet Labels aus.

    Der Tokenizer wird mit add_prefix_space=True geladen, weil DeBERTa
    beim Verarbeiten von vorher tokenisierten Woertern ein fuehrendes
    Leerzeichen erwartet.

    Args:
        model_name:       HuggingFace Model-ID (z.B. 'microsoft/deberta-v3-base').
        dataset_name:     "multinerd" oder "wnut_17".
        dataset_language: Sprachfilter fuer MultiNERD (Standard: "en").
        max_length:       Maximale Sequenzlaenge fuer den Tokenizer.

    Returns:
        Tuple aus (tokenized_dataset, tokenizer, DatasetInfo).
    """
    # Datensatz und Metadaten laden
    dataset, info = load_ner_dataset(dataset_name, language=dataset_language)

    # Tokenizer laden; add_prefix_space fuer DeBERTa-Kompatibilitaet
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        use_fast=True,  # use_fast=True wird fuer word_ids() benoetigt
    )

    # Tokenisierung und Label-Alignment ueber alle Splits mappen
    # batched=True beschleunigt die Verarbeitung erheblich
    tokenized_dataset: DatasetDict = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length),
        batched=True,
        # Originalspalten entfernen, da sie durch Tokenisierung ersetzt werden
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer, info
