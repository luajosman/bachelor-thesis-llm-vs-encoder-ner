"""
preprocess_decoder.py — Vorverarbeitung für LLM-basierte NER (generativer Ansatz)

Beim Decoder-Ansatz wird NER als Generierungsaufgabe formuliert:
Das Modell erhält einen System-Prompt + den Eingabesatz und soll eine
JSON-Liste der erkannten Entities ausgeben.

Trainingsformat (ChatML, kompatibel mit Qwen):
    <|im_start|>system
    Du bist ein NER-System ...
    <|im_start|>user
    EU rejects German call ...
    <|im_start|>assistant
    [{"entity": "EU", "type": "corporation"}, ...]

Der Assistent-Turn enthält die Gold-Entities als JSON-String.
Bei der Inferenz wird das Modell nach dem User-Turn aufgefordert zu generieren.

Verwendung:
    from src.data.preprocess_decoder import prepare_decoder_dataset
    dataset = prepare_decoder_dataset()
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datasets import DatasetDict, Dataset

from src.data.load_wnut17 import ID2LABEL, LABEL_LIST, load_wnut17

# ---------------------------------------------------------------------------
# System-Prompt
# ---------------------------------------------------------------------------

# Der System-Prompt erklärt dem Modell seine Aufgabe und das erwartete
# Ausgabeformat. Er wird bei jedem Sample wiederholt (kein separates
# Fine-Tuning des System-Prompts — der bleibt eingefroren).
SYSTEM_PROMPT: str = (
    "Du bist ein NER-System. Extrahiere alle Named Entities aus dem gegebenen Text.\n"
    "Gib das Ergebnis als JSON-Liste zurück. Jedes Element hat die Felder "
    "\"entity\" (der Text) und \"type\" (einer von: person, location, corporation, "
    "creative-work, group, product).\n"
    "Wenn keine Entities vorhanden sind, gib eine leere Liste [] zurück.\n"
    "Antworte NUR mit dem JSON, ohne zusätzlichen Text."
)


# ---------------------------------------------------------------------------
# BIO-Tags → Entity-Liste
# ---------------------------------------------------------------------------

def extract_entities_from_bio(
    tokens: List[str],
    ner_tags: List[int],
) -> List[Dict[str, str]]:
    """Konvertiert eine BIO-Tag-Sequenz in eine Liste von Entity-Dicts.

    Wird genutzt um die Gold-Labels aus dem WNUT-2017-Format in das
    JSON-Format für den Assistent-Turn zu übersetzen.

    Args:
        tokens:   Wortliste des Satzes.
        ner_tags: Integer-BIO-Tags (Index in ID2LABEL).

    Returns:
        Liste von Dicts: [{"entity": "New York", "type": "location"}, ...]

    Beispiel:
        >>> extract_entities_from_bio(["New", "York", "is", "great"], [7, 8, 0, 0])
        [{"entity": "New York", "type": "location"}]
    """
    entities: List[Dict[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, ner_tags):
        label = ID2LABEL[tag]

        if label.startswith("B-"):
            # Offene Entity zuerst abschließen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            # Neue Entity starten; "B-person" → type = "person"
            current_type = label[2:]
            current_tokens = [token]

        elif label.startswith("I-") and current_type is not None:
            # Aktuelle Entity um das nächste Wort erweitern
            current_tokens.append(token)

        else:
            # "O"-Tag: Entity abschließen und Zustand zurücksetzen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []

    # Letzte Entity am Satzende abschließen
    if current_type is not None:
        entities.append({"entity": " ".join(current_tokens), "type": current_type})

    return entities


# ---------------------------------------------------------------------------
# Chat-Format für SFT
# ---------------------------------------------------------------------------

def format_for_llm(sample: Dict) -> Dict:
    """Konvertiert ein WNUT-2017-Sample in das ChatML-Format für SFTTrainer.

    Erzeugt ein 'messages'-Feld mit drei Turns:
      - system: der NER-Instruktions-Prompt
      - user:   der Eingabesatz (Wörter zu String zusammengesetzt)
      - assistant: die Gold-Entities als JSON-String

    Das SFT-Training optimiert nur auf dem Assistent-Turn,
    die anderen Turns werden beim Loss maskiert.

    Args:
        sample: Einzelne Zeile des WNUT-2017-Datensatzes.

    Returns:
        Dict mit Schlüssel 'messages' (Liste von role/content-Dicts).
    """
    tokens: List[str] = sample["tokens"]
    ner_tags: List[int] = sample["ner_tags"]

    # Wörter zu einem einzigen String zusammensetzen
    sentence: str = " ".join(tokens)

    # Gold-Entities aus BIO-Tags extrahieren und als JSON serialisieren
    entities: List[Dict[str, str]] = extract_entities_from_bio(tokens, ner_tags)
    assistant_answer: str = json.dumps(entities, ensure_ascii=False)

    # ChatML-Struktur aufbauen
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sentence},
        {"role": "assistant", "content": assistant_answer},
    ]

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Datensatz-Vorbereitung für Training
# ---------------------------------------------------------------------------

def prepare_decoder_dataset() -> DatasetDict:
    """Lädt WNUT-2017 und konvertiert alle Splits in das ChatML-Format.

    Jedes Sample bekommt ein 'messages'-Feld (system + user + assistant).
    Die Original-Spalten 'tokens' und 'ner_tags' werden entfernt.

    Returns:
        DatasetDict mit Splits train/validation/test, jeweils mit 'messages'-Spalte.
    """
    raw: DatasetDict = load_wnut17()

    # format_for_llm auf alle Splits anwenden
    formatted: DatasetDict = raw.map(
        format_for_llm,
        remove_columns=raw["train"].column_names,
    )
    return formatted


# ---------------------------------------------------------------------------
# Inferenz-Hilfsfunktion
# ---------------------------------------------------------------------------

def prepare_test_inputs(
    dataset_split: Dataset,
) -> Tuple[List[List[Dict]], List[List[Dict[str, str]]]]:
    """Baut Prompt-Only-Nachrichten (ohne Assistent-Turn) für die Inferenz.

    Bei der Inferenz darf der Assistent-Turn nicht übergeben werden —
    das Modell soll ihn selbst generieren. Diese Funktion gibt
    Prompts (system + user) und Gold-Entities (Referenz) zurück.

    Args:
        dataset_split: Ein einzelner WNUT-2017-Split (mit 'tokens' und 'ner_tags').

    Returns:
        Tuple aus:
          - prompts:       Liste von [system, user]-Nachrichtenlisten
          - gold_entities: Liste von Entity-Dicts pro Satz (Referenz für Evaluation)
    """
    prompts: List[List[Dict]] = []
    gold_entities: List[List[Dict[str, str]]] = []

    for sample in dataset_split:
        tokens: List[str] = sample["tokens"]
        ner_tags: List[int] = sample["ner_tags"]
        sentence: str = " ".join(tokens)

        # Nur System- und User-Turn — der Assistent-Turn fehlt bewusst
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": sentence},
        ]
        prompts.append(messages)

        # Gold-Entities aus BIO-Tags für die spätere Evaluation extrahieren
        gold_entities.append(extract_entities_from_bio(tokens, ner_tags))

    return prompts, gold_entities
