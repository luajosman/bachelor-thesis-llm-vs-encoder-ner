"""Decoder preprocessing for generative NER on MultiNERD English."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from src.data.dataset_loader import DatasetInfo, load_ner_dataset

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


# ---------------------------------------------------------------------------
# System-Prompt (dynamisch je nach Datensatz)
# ---------------------------------------------------------------------------

def build_system_prompt(entity_types: List[str]) -> str:
    """Baut den System-Prompt mit den Entity-Typen des jeweiligen Datensatzes.

    Args:
        entity_types: Liste der Entity-Typen (z.B. ["PER", "ORG", "LOC", ...]).

    Returns:
        Vollstaendiger System-Prompt als String.
    """
    types_str = ", ".join(entity_types)
    return (
        "You are a Named Entity Recognition (NER) system. "
        "Extract all named entities from the numbered token list.\n"
        "Return ONLY a JSON array. Do not use Markdown code blocks and do not "
        "add explanations.\n"
        "Each entity object must have exactly these fields: "
        "\"start_token\" (integer, inclusive), "
        "\"end_token\" (integer, exclusive), "
        "\"text\" (the exact text covered by the token span), and "
        f"\"type\" (one of: {types_str}).\n"
        "Token indices must refer exactly to the numbered token list shown by "
        "the user. If no entities are present, return exactly []."
    )


def format_numbered_tokens(tokens: List[str]) -> str:
    """Format tokens as the numbered list used in decoder prompts."""
    token_lines = [f"{i}: {token}" for i, token in enumerate(tokens)]
    return "Tokens:\n" + "\n".join(token_lines)


# ---------------------------------------------------------------------------
# BIO-Tags → Entity-Liste
# ---------------------------------------------------------------------------

def extract_entities_from_bio(
    tokens: List[str],
    ner_tags: List[int],
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Konvertiert eine BIO-Tag-Sequenz in eine Liste von Entity-Dicts.

    Wird genutzt um die Gold-Labels aus dem BIO-Format in das
    JSON-Format fuer den Assistent-Turn zu uebersetzen.

    Args:
        tokens:   Wortliste des Satzes.
        ner_tags: Integer-BIO-Tags (Index in id2label).
        id2label: Mapping von Integer-ID zu Label-String.

    Returns:
        Liste von Dicts:
        [{"start_token": 0, "end_token": 2, "text": "New York", "type": "LOC"}, ...]
    """
    entities: List[Dict[str, Any]] = []
    current_tokens: List[str] = []
    current_type: str | None = None
    current_start: int | None = None

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        label = id2label[tag]

        if label.startswith("B-"):
            # Offene Entity zuerst abschliessen
            if current_type is not None and current_start is not None:
                entities.append({
                    "start_token": current_start,
                    "end_token": i,
                    "text": " ".join(current_tokens),
                    "type": current_type,
                })
            # Neue Entity starten; "B-PER" -> type = "PER"
            current_type = label[2:]
            current_tokens = [token]
            current_start = i

        elif label.startswith("I-") and current_type is not None:
            # Aktuelle Entity um das naechste Wort erweitern
            current_tokens.append(token)

        else:
            # "O"-Tag: Entity abschliessen und Zustand zuruecksetzen
            if current_type is not None and current_start is not None:
                entities.append({
                    "start_token": current_start,
                    "end_token": i,
                    "text": " ".join(current_tokens),
                    "type": current_type,
                })
            current_type = None
            current_tokens = []
            current_start = None

    # Letzte Entity am Satzende abschliessen
    if current_type is not None and current_start is not None:
        entities.append({
            "start_token": current_start,
            "end_token": len(tokens),
            "text": " ".join(current_tokens),
            "type": current_type,
        })

    return entities


# ---------------------------------------------------------------------------
# Chat-Format fuer SFT
# ---------------------------------------------------------------------------

def format_for_llm(
    sample: Dict,
    system_prompt: str,
    id2label: Dict[int, str],
) -> Dict:
    """Konvertiert ein NER-Sample in das ChatML-Format fuer SFTTrainer.

    Erzeugt ein 'messages'-Feld mit drei Turns:
      - system: der NER-Instruktions-Prompt
      - user:   die nummerierte Tokenliste
      - assistant: die Gold-Entities mit tokenbasierten Offsets als JSON-String

    Args:
        sample:        Einzelne Zeile des Datensatzes.
        system_prompt: Der datensatz-spezifische System-Prompt.
        id2label:      Mapping Integer → Label-String.

    Returns:
        Dict mit Schluessel 'messages' (Liste von role/content-Dicts).
    """
    tokens: List[str] = sample["tokens"]
    ner_tags: List[int] = sample["ner_tags"]

    # Gold-Entities aus BIO-Tags extrahieren und als JSON serialisieren
    entities: List[Dict[str, Any]] = extract_entities_from_bio(tokens, ner_tags, id2label)
    assistant_answer: str = json.dumps(entities, ensure_ascii=False)

    # ChatML-Struktur aufbauen
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": format_numbered_tokens(tokens)},
        {"role": "assistant", "content": assistant_answer},
    ]

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Datensatz-Vorbereitung fuer Training
# ---------------------------------------------------------------------------

def prepare_decoder_dataset(
) -> Tuple["DatasetDict", DatasetInfo]:
    """Load MultiNERD English and convert all splits to chat messages.

    Jedes Sample bekommt ein 'messages'-Feld (system + user + assistant).
    Die Original-Spalten 'tokens' und 'ner_tags' werden entfernt.
    """
    raw, info = load_ner_dataset()

    system_prompt = build_system_prompt(info.entity_types)
    id2label = info.id2label

    # format_for_llm auf alle Splits anwenden
    formatted: "DatasetDict" = raw.map(
        lambda sample: format_for_llm(sample, system_prompt, id2label),
        remove_columns=raw["train"].column_names,
    )
    return formatted, info


# ---------------------------------------------------------------------------
# Inferenz-Hilfsfunktion
# ---------------------------------------------------------------------------

def prepare_test_inputs(
    dataset_split: "Dataset",
    info: DatasetInfo,
) -> Tuple[List[List[Dict]], List[List[Dict[str, Any]]]]:
    """Baut Prompt-Only-Nachrichten (ohne Assistent-Turn) fuer die Inferenz.

    Bei der Inferenz darf der Assistent-Turn nicht uebergeben werden —
    das Modell soll ihn selbst generieren. Diese Funktion gibt
    Prompts (system + user) und Gold-Entities (Referenz) zurueck.

    Args:
        dataset_split: Ein einzelner Split (mit 'tokens' und 'ner_tags').
        info:          DatasetInfo mit Entity-Typen und Label-Mappings.

    Returns:
        Tuple aus:
          - prompts:       Liste von [system, user]-Nachrichtenlisten
          - gold_entities: Liste von Entity-Dicts pro Satz (Referenz fuer Evaluation)
    """
    system_prompt = build_system_prompt(info.entity_types)
    id2label = info.id2label

    prompts: List[List[Dict]] = []
    gold_entities: List[List[Dict[str, Any]]] = []

    for sample in dataset_split:
        tokens: List[str] = sample["tokens"]
        ner_tags: List[int] = sample["ner_tags"]

        # Nur System- und User-Turn — der Assistent-Turn fehlt bewusst
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": format_numbered_tokens(tokens)},
        ]
        prompts.append(messages)

        # Gold-Entities aus BIO-Tags fuer die spaetere Evaluation extrahieren
        gold_entities.append(extract_entities_from_bio(tokens, ner_tags, id2label))

    return prompts, gold_entities
