"""
preprocess_decoder.py — Vorverarbeitung fuer LLM-basierte NER (generativer Ansatz)

Beim Decoder-Ansatz wird NER als Generierungsaufgabe formuliert:
Das Modell erhaelt einen System-Prompt + den Eingabesatz und soll eine
JSON-Liste der erkannten Entities ausgeben.

Trainingsformat (ChatML, kompatibel mit Qwen):
    <|im_start|>system
    Du bist ein NER-System ...
    <|im_start|>user
    EU rejects German call ...
    <|im_start|>assistant
    [{"entity": "EU", "type": "ORG"}, ...]

Der System-Prompt wird dynamisch aus den Entity-Typen des jeweiligen
Datensatzes generiert (MultiNERD oder WNUT-2017).

Verwendung:
    from src.data.preprocess_decoder import prepare_decoder_dataset
    dataset, info = prepare_decoder_dataset("multinerd")
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict

from src.data.dataset_loader import DatasetInfo, load_ner_dataset


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
        "Extract all named entities from the given text.\n"
        "Return the result as a JSON list. Each element has the fields "
        f"\"entity\" (the text span) and \"type\" (one of: {types_str}).\n"
        "If no entities are present, return an empty list [].\n"
        "Respond ONLY with the JSON, without any additional text."
    )


# ---------------------------------------------------------------------------
# BIO-Tags → Entity-Liste
# ---------------------------------------------------------------------------

def extract_entities_from_bio(
    tokens: List[str],
    ner_tags: List[int],
    id2label: Dict[int, str],
) -> List[Dict[str, str]]:
    """Konvertiert eine BIO-Tag-Sequenz in eine Liste von Entity-Dicts.

    Wird genutzt um die Gold-Labels aus dem BIO-Format in das
    JSON-Format fuer den Assistent-Turn zu uebersetzen.

    Args:
        tokens:   Wortliste des Satzes.
        ner_tags: Integer-BIO-Tags (Index in id2label).
        id2label: Mapping von Integer-ID zu Label-String.

    Returns:
        Liste von Dicts: [{"entity": "New York", "type": "LOC"}, ...]
    """
    entities: List[Dict[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, ner_tags):
        label = id2label[tag]

        if label.startswith("B-"):
            # Offene Entity zuerst abschliessen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            # Neue Entity starten; "B-PER" → type = "PER"
            current_type = label[2:]
            current_tokens = [token]

        elif label.startswith("I-") and current_type is not None:
            # Aktuelle Entity um das naechste Wort erweitern
            current_tokens.append(token)

        else:
            # "O"-Tag: Entity abschliessen und Zustand zuruecksetzen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []

    # Letzte Entity am Satzende abschliessen
    if current_type is not None:
        entities.append({"entity": " ".join(current_tokens), "type": current_type})

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
      - user:   der Eingabesatz (Woerter zu String zusammengesetzt)
      - assistant: die Gold-Entities als JSON-String

    Args:
        sample:        Einzelne Zeile des Datensatzes.
        system_prompt: Der datensatz-spezifische System-Prompt.
        id2label:      Mapping Integer → Label-String.

    Returns:
        Dict mit Schluessel 'messages' (Liste von role/content-Dicts).
    """
    tokens: List[str] = sample["tokens"]
    ner_tags: List[int] = sample["ner_tags"]

    # Woerter zu einem einzigen String zusammensetzen
    sentence: str = " ".join(tokens)

    # Gold-Entities aus BIO-Tags extrahieren und als JSON serialisieren
    entities: List[Dict[str, str]] = extract_entities_from_bio(tokens, ner_tags, id2label)
    assistant_answer: str = json.dumps(entities, ensure_ascii=False)

    # ChatML-Struktur aufbauen
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": sentence},
        {"role": "assistant", "content": assistant_answer},
    ]

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Datensatz-Vorbereitung fuer Training
# ---------------------------------------------------------------------------

def prepare_decoder_dataset(
    dataset_name: str = "multinerd",
    dataset_language: str = "en",
) -> Tuple[DatasetDict, DatasetInfo]:
    """Laedt einen NER-Datensatz und konvertiert alle Splits in das ChatML-Format.

    Jedes Sample bekommt ein 'messages'-Feld (system + user + assistant).
    Die Original-Spalten 'tokens' und 'ner_tags' werden entfernt.

    Args:
        dataset_name:     "multinerd" oder "wnut_17".
        dataset_language: Sprachfilter fuer MultiNERD (Standard: "en").

    Returns:
        Tuple (DatasetDict mit 'messages'-Spalte, DatasetInfo).
    """
    raw, info = load_ner_dataset(dataset_name, language=dataset_language)

    system_prompt = build_system_prompt(info.entity_types)
    id2label = info.id2label

    # format_for_llm auf alle Splits anwenden
    formatted: DatasetDict = raw.map(
        lambda sample: format_for_llm(sample, system_prompt, id2label),
        remove_columns=raw["train"].column_names,
    )
    return formatted, info


# ---------------------------------------------------------------------------
# Inferenz-Hilfsfunktion
# ---------------------------------------------------------------------------

def prepare_test_inputs(
    dataset_split: Dataset,
    info: DatasetInfo,
) -> Tuple[List[List[Dict]], List[List[Dict[str, str]]]]:
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
    gold_entities: List[List[Dict[str, str]]] = []

    for sample in dataset_split:
        tokens: List[str] = sample["tokens"]
        ner_tags: List[int] = sample["ner_tags"]
        sentence: str = " ".join(tokens)

        # Nur System- und User-Turn — der Assistent-Turn fehlt bewusst
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": sentence},
        ]
        prompts.append(messages)

        # Gold-Entities aus BIO-Tags fuer die spaetere Evaluation extrahieren
        gold_entities.append(extract_entities_from_bio(tokens, ner_tags, id2label))

    return prompts, gold_entities
