"""
parse_output.py — JSON-Parsing und Evaluation fuer LLM-generierte NER-Ausgaben

Das LLM soll Ausgaben im tokenbasierten Offset-Format produzieren:
    [{"start_token": 0, "end_token": 2, "text": "Barack Obama", "type": "PER"}, ...]

In der Praxis weichen generative Modelle manchmal von diesem Format ab:
  - Markdown-Code-Bloecke (```json ... ```)
  - Denk-Bloecke von Qwen3 (<think>...</think>)
  - Abgeschnittene oder invalide JSON-Strings
  - Falsche Entity-Typen, fehlende Felder, ungueltige Offsets

Dieses Modul implementiert einen dreistufigen Fallback-Parser und
konvertiert die geparsten Entities fuer die seqeval-Evaluation in BIO-Tags.

Die erlaubten Entity-Typen werden dynamisch uebergeben. In der finalen
Codebasis ist das die MultiNERD-English-Taxonomie.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# JSON-Parser mit Fallback-Strategien
# ---------------------------------------------------------------------------

def parse_llm_output(
    output_text: str,
    tokens: List[str],
    valid_types: Optional[FrozenSet[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Parst den tokenbasierten JSON-Entity-Output des LLMs.

    Strategie 1: Direktes json.loads() nach Strip des Textes.
    Strategie 2: Markdown-Code-Fence entfernen (```json ... ```) und parsen.
    Strategie 3: Regex-Suche nach dem ersten [...]-Block im Text.

    Zusaetzlich werden <think>...</think>-Bloecke von Qwen3 vor dem Parsing
    herausgefiltert, da der Thinking-Mode fuer strukturierte Ausgaben
    nicht geeignet ist.

    Args:
        output_text: Roher Text, den das LLM generiert hat.
        tokens: Tokenliste des Samples. Die Offsets im JSON beziehen sich auf
                genau diese Liste.
        valid_types: Erlaubte Entity-Typen (frozenset). Wenn None, wird
                     keine Typ-Validierung durchgefuehrt.

    Returns:
        Tuple (entities, parse_status):
          - entities:     Liste von Entity-Dicts (leer bei Versagen).
          - parse_status: "ok", "markdown_stripped", "regex_fallback" oder "failed".
    """
    entities, status, _ = parse_llm_output_with_diagnostics(output_text, tokens, valid_types)
    return entities, status


def parse_llm_output_with_diagnostics(
    output_text: str,
    tokens: List[str],
    valid_types: Optional[FrozenSet[str]] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, int]]:
    """Parse an LLM output and return parser diagnostics.

    Diagnostics count discarded or malformed structures before validation drops
    them, so error analysis can report unknown types and missing fields.
    """
    text = output_text.strip()
    diagnostics = _empty_diagnostics()

    # Qwen3 Thinking-Mode: <think>...</think>-Bloecke entfernen
    # Diese enthalten den Denkprozess des Modells, nicht die eigentliche Antwort
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # --- Strategie 1: Direktes JSON-Parsing ---
    try:
        result = json.loads(text)
        if isinstance(result, list):
            entities, validation = _validate_entities(result, tokens, valid_types)
            _merge_diagnostics(diagnostics, validation)
            return entities, "ok", diagnostics
        diagnostics["wrong_schema"] += 1
    except json.JSONDecodeError:
        pass  # Weiter zur naechsten Strategie

    # --- Strategie 2: Markdown-Code-Fence entfernen ---
    # Manche Modelle umschliessen JSON mit ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                entities, validation = _validate_entities(result, tokens, valid_types)
                _merge_diagnostics(diagnostics, validation)
                return entities, "markdown_stripped", diagnostics
            diagnostics["wrong_schema"] += 1
        except json.JSONDecodeError:
            pass

    # --- Strategie 3: Regex-Suche nach [...]-Block ---
    # Als letzter Ausweg: das erste JSON-Array im Text suchen
    array_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if array_match:
        try:
            result = json.loads(array_match.group(0))
            if isinstance(result, list):
                entities, validation = _validate_entities(result, tokens, valid_types)
                _merge_diagnostics(diagnostics, validation)
                return entities, "regex_fallback", diagnostics
            diagnostics["wrong_schema"] += 1
        except json.JSONDecodeError:
            pass

    # Alle Strategien gescheitert → leere Liste zurueckgeben
    return [], "failed", diagnostics


def _validate_entities(
    raw: List,
    tokens: List[str],
    valid_types: Optional[FrozenSet[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filtert eine geparste JSON-Liste auf valide tokenbasierte Entity-Dicts.

    Entfernt Eintraege ohne Pflichtfelder, mit unbekannten Entity-Typen,
    ungueltigen Offsets, Text/Token-Mismatches oder ueberlappenden Spans.

    Die Typ-Validierung ist case-insensitive: das LLM-Output wird
    gegen die lowercase-Versionen der valid_types geprueft.

    Args:
        raw:         Geparste JSON-Liste (kann Dicts oder ungueltige Eintraege enthalten).
        tokens:      Tokenliste, auf die start_token/end_token referenzieren.
        valid_types: Erlaubte Entity-Typen (frozenset). None = keine Validierung.

    Returns:
        Bereinigte Liste mit nur validen Entity-Dicts.
    """
    # Case-insensitive Lookup-Set aufbauen
    valid_lower: Optional[Set[str]] = None
    # Mapping von lowercase -> original case (fuer konsistente Ausgabe)
    lower_to_original: Optional[Dict[str, str]] = None
    if valid_types is not None:
        valid_lower = {t.lower() for t in valid_types}
        lower_to_original = {t.lower(): t for t in valid_types}

    candidates: List[Dict[str, Any]] = []
    diagnostics = _empty_diagnostics()
    required_fields = {"start_token", "end_token", "text", "type"}

    for original_order, item in enumerate(raw):
        if not isinstance(item, dict):
            diagnostics["invalid_items"] += 1
            continue  # Kein Dict -> ueberspringen

        if any(field not in item for field in required_fields):
            diagnostics["missing_fields"] += 1
            continue

        start_token = item.get("start_token")
        end_token = item.get("end_token")
        text = item.get("text", "")
        etype = item.get("type", "")

        # Text und Typ muessen Strings sein.
        if not isinstance(text, str) or not text.strip():
            diagnostics["missing_fields"] += 1
            continue
        if not isinstance(etype, str) or not etype.strip():
            diagnostics["missing_fields"] += 1
            continue

        # bool ist eine int-Subclass in Python, wird hier aber nicht als
        # gueltiger Tokenindex akzeptiert.
        if type(start_token) is not int or type(end_token) is not int:
            diagnostics["invalid_offsets"] += 1
            continue
        if not (0 <= start_token < end_token <= len(tokens)):
            diagnostics["invalid_offsets"] += 1
            continue

        etype_clean = etype.strip()

        # Entity-Typ validieren (case-insensitive)
        if valid_lower is not None:
            if etype_clean.lower() not in valid_lower:
                diagnostics["unknown_types"] += 1
                continue
            # Typ auf die kanonische Schreibweise normalisieren
            etype_clean = lower_to_original[etype_clean.lower()]

        expected_text = " ".join(tokens[start_token:end_token])
        if text.strip() != expected_text:
            diagnostics["text_mismatches"] += 1
            continue

        candidates.append({
            "start_token": start_token,
            "end_token": end_token,
            "text": expected_text,
            "type": etype_clean,
            "_original_order": original_order,
        })

    valid: List[Dict[str, Any]] = []
    occupied = [False] * len(tokens)
    for item in sorted(
        candidates,
        key=lambda ent: (ent["start_token"], ent["end_token"], ent["_original_order"]),
    ):
        start = item["start_token"]
        end = item["end_token"]
        if any(occupied[start:end]):
            diagnostics["overlaps"] += 1
            continue
        for i in range(start, end):
            occupied[i] = True
        valid.append({
            "start_token": start,
            "end_token": end,
            "text": item["text"],
            "type": item["type"],
        })
    return valid, diagnostics


def _empty_diagnostics() -> Dict[str, int]:
    return {
        "invalid_items": 0,
        "missing_fields": 0,
        "unknown_types": 0,
        "wrong_schema": 0,
        "invalid_offsets": 0,
        "text_mismatches": 0,
        "overlaps": 0,
    }


def _merge_diagnostics(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


# ---------------------------------------------------------------------------
# Entity-Liste → BIO-Tag-Sequenz
# ---------------------------------------------------------------------------

def entities_to_bio(
    tokens: List[str],
    entities: List[Dict[str, Any]],
) -> List[str]:
    """Konvertiert tokenbasierte Entity-Dicts in eine BIO-Tag-Sequenz.

    Die Rueckabbildung nutzt ausschliesslich start_token/end_token. Es findet
    kein String-Matching statt, damit wiederholte gleiche Textspans eindeutig
    bleiben.

    Args:
        tokens:   Woerter des Satzes.
        entities: Entity-Dicts aus dem LLM-Output (nach parse_llm_output).

    Returns:
        BIO-Tag-Liste gleicher Laenge wie tokens,
        z.B. ["O", "B-PER", "I-PER", "O"].
    """
    bio_tags = ["O"] * len(tokens)

    for ent in entities:
        start = ent.get("start_token")
        end = ent.get("end_token")
        etype = ent.get("type")
        if type(start) is not int or type(end) is not int or not isinstance(etype, str):
            continue
        if not (0 <= start < end <= len(tokens)):
            continue
        if any(tag != "O" for tag in bio_tags[start:end]):
            continue

        bio_tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            bio_tags[i] = f"I-{etype}"

    return bio_tags


# ---------------------------------------------------------------------------
# Evaluation-Wrapper
# ---------------------------------------------------------------------------

def evaluate_llm_predictions(
    tokens_list:    List[List[str]],
    gold_entities:  List[List[Dict[str, Any]]],
    pred_entities:  List[List[Dict[str, Any]]],
    parse_statuses: List[str],
    parse_diagnostics: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Berechnet seqeval-Metriken fuer LLM-Vorhersagen.

    Konvertiert sowohl Gold- als auch Pred-Entities in BIO-Sequenzen
    und berechnet dann entity-level F1, Precision und Recall.
    Zusaetzlich wird die Parse-Fehlerrate als eigene Metrik ausgegeben.

    Args:
        tokens_list:    Token-Listen (eine pro Satz).
        gold_entities:  Gold-Entity-Dicts pro Satz.
        pred_entities:  Vorhergesagte Entity-Dicts pro Satz.
        parse_statuses: Parse-Ergebnis pro Satz (fuer Fehlerrate).

    Returns:
        Dict mit precision, recall, f1, parse_failure_rate und
        Zaehlung der einzelnen Parse-Status-Kategorien.
    """
    from src.evaluate.metrics import compute_ner_metrics

    true_bio: List[List[str]] = []
    pred_bio: List[List[str]] = []

    # Alle Samples in BIO-Sequenzen umwandeln
    for tokens, gold, pred in zip(tokens_list, gold_entities, pred_entities):
        true_bio.append(entities_to_bio(tokens, gold))
        pred_bio.append(entities_to_bio(tokens, pred))

    # Anteil der Samples, bei denen kein valides JSON geparst werden konnte
    parse_fail_rate = parse_statuses.count("failed") / max(len(parse_statuses), 1)

    metrics = compute_ner_metrics(true_bio, pred_bio)
    diagnostics = parse_diagnostics or []

    return {
        "precision":              metrics["precision"],
        "recall":                 metrics["recall"],
        "f1":                     metrics["f1"],
        "parse_failure_rate":     parse_fail_rate,
        # Aufschluesselung nach Parse-Strategie (fuer Fehleranalyse-Kapitel)
        "parse_ok":               parse_statuses.count("ok"),
        "parse_markdown_stripped": parse_statuses.count("markdown_stripped"),
        "parse_regex_fallback":   parse_statuses.count("regex_fallback"),
        "parse_failed":           parse_statuses.count("failed"),
        "parse_invalid_items":     _sum_diagnostics(diagnostics, "invalid_items"),
        "parse_missing_fields":    _sum_diagnostics(diagnostics, "missing_fields"),
        "parse_unknown_types":     _sum_diagnostics(diagnostics, "unknown_types"),
        "parse_wrong_schema":      _sum_diagnostics(diagnostics, "wrong_schema"),
        "parse_invalid_offsets":   _sum_diagnostics(diagnostics, "invalid_offsets"),
        "parse_text_mismatches":   _sum_diagnostics(diagnostics, "text_mismatches"),
        "parse_overlaps":          _sum_diagnostics(diagnostics, "overlaps"),
    }


def _sum_diagnostics(diagnostics: List[Dict[str, int]], key: str) -> int:
    return int(sum(item.get(key, 0) for item in diagnostics))
