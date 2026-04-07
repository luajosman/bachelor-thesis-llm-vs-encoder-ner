"""
parse_output.py — JSON-Parsing und Evaluation fuer LLM-generierte NER-Ausgaben

Das LLM soll Ausgaben im Format produzieren:
    [{"entity": "Barack Obama", "type": "PER"}, ...]

In der Praxis weichen generative Modelle manchmal von diesem Format ab:
  - Markdown-Code-Bloecke (```json ... ```)
  - Denk-Bloecke von Qwen3 (<think>...</think>)
  - Abgeschnittene oder invalide JSON-Strings
  - Falsche Entity-Typen, fehlende Felder

Dieses Modul implementiert einen dreistufigen Fallback-Parser und
konvertiert die geparsten Entities fuer die seqeval-Evaluation in BIO-Tags.

Die erlaubten Entity-Typen werden dynamisch uebergeben, damit das Modul
sowohl mit MultiNERD als auch mit WNUT-2017 funktioniert.
"""

from __future__ import annotations

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# JSON-Parser mit Fallback-Strategien
# ---------------------------------------------------------------------------

def parse_llm_output(
    output_text: str,
    valid_types: Optional[FrozenSet[str]] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """Parst den JSON-Entity-Output des LLMs mit drei Fallback-Strategien.

    Strategie 1: Direktes json.loads() nach Strip des Textes.
    Strategie 2: Markdown-Code-Fence entfernen (```json ... ```) und parsen.
    Strategie 3: Regex-Suche nach dem ersten [...]-Block im Text.

    Zusaetzlich werden <think>...</think>-Bloecke von Qwen3 vor dem Parsing
    herausgefiltert, da der Thinking-Mode fuer strukturierte Ausgaben
    nicht geeignet ist.

    Args:
        output_text: Roher Text, den das LLM generiert hat.
        valid_types: Erlaubte Entity-Typen (frozenset). Wenn None, wird
                     keine Typ-Validierung durchgefuehrt.

    Returns:
        Tuple (entities, parse_status):
          - entities:     Liste von Entity-Dicts (leer bei Versagen).
          - parse_status: "ok", "markdown_stripped", "regex_fallback" oder "failed".
    """
    text = output_text.strip()

    # Qwen3 Thinking-Mode: <think>...</think>-Bloecke entfernen
    # Diese enthalten den Denkprozess des Modells, nicht die eigentliche Antwort
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # --- Strategie 1: Direktes JSON-Parsing ---
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return _validate_entities(result, valid_types), "ok"
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
                return _validate_entities(result, valid_types), "markdown_stripped"
        except json.JSONDecodeError:
            pass

    # --- Strategie 3: Regex-Suche nach [...]-Block ---
    # Als letzter Ausweg: das erste JSON-Array im Text suchen
    array_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if array_match:
        try:
            result = json.loads(array_match.group(0))
            if isinstance(result, list):
                return _validate_entities(result, valid_types), "regex_fallback"
        except json.JSONDecodeError:
            pass

    # Alle Strategien gescheitert → leere Liste zurueckgeben
    return [], "failed"


def _validate_entities(
    raw: List,
    valid_types: Optional[FrozenSet[str]] = None,
) -> List[Dict[str, str]]:
    """Filtert eine geparste JSON-Liste auf valide Entity-Dicts.

    Entfernt Eintraege ohne 'entity'- oder 'type'-Feld und solche mit
    unbekannten Entity-Typen (falls valid_types angegeben).

    Die Typ-Validierung ist case-insensitive: das LLM-Output wird
    gegen die lowercase-Versionen der valid_types geprueft.

    Args:
        raw:         Geparste JSON-Liste (kann Dicts oder ungueltige Eintraege enthalten).
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

    valid: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue  # Kein Dict → ueberspringen

        entity = item.get("entity", "")
        etype = item.get("type", "")

        # Beide Felder muessen nicht-leere Strings sein
        if not isinstance(entity, str) or not entity.strip():
            continue
        if not isinstance(etype, str) or not etype.strip():
            continue

        etype_clean = etype.strip()

        # Entity-Typ validieren (case-insensitive)
        if valid_lower is not None:
            if etype_clean.lower() not in valid_lower:
                continue
            # Typ auf die kanonische Schreibweise normalisieren
            etype_clean = lower_to_original[etype_clean.lower()]

        valid.append({"entity": entity.strip(), "type": etype_clean})
    return valid


# ---------------------------------------------------------------------------
# Entity-Liste → BIO-Tag-Sequenz
# ---------------------------------------------------------------------------

def entities_to_bio(
    tokens: List[str],
    entities: List[Dict[str, str]],
) -> List[str]:
    """Konvertiert Entity-Dicts in eine BIO-Tag-Sequenz fuer seqeval.

    Verwendet einfaches String-Matching: Die Woerter der Entity werden
    im Token-Fenster gesucht. Beim ersten Treffer werden B-/I-Tags gesetzt.
    Ueberlappungen werden vermieden (bereits getaggte Positionen werden
    nicht ueberschrieben).

    Args:
        tokens:   Woerter des Satzes.
        entities: Entity-Dicts aus dem LLM-Output (nach parse_llm_output).

    Returns:
        BIO-Tag-Liste gleicher Laenge wie tokens,
        z.B. ["O", "B-PER", "I-PER", "O"].
    """
    # Alle Positionen zunaechst mit "O" initialisieren
    bio_tags = ["O"] * len(tokens)

    for ent in entities:
        ent_text = ent["entity"]
        etype = ent["type"]

        # Entity-Text auf Leerzeichen aufteilen (konsistent mit Tokenisierung)
        ent_tokens = ent_text.split()
        n = len(ent_tokens)

        # Sliding-Window-Suche: jeden moeglichen Startpunkt pruefen
        for start in range(len(tokens) - n + 1):
            window = tokens[start : start + n]
            if window == ent_tokens:
                # Ueberlappungs-Check: nur taggen, wenn alle Positionen noch "O" sind
                if all(bio_tags[start + j] == "O" for j in range(n)):
                    bio_tags[start] = f"B-{etype}"              # Beginn der Entity
                    for j in range(1, n):
                        bio_tags[start + j] = f"I-{etype}"     # Fortsetzung
                break  # Ersten Treffer nutzen, keine weiteren Matches suchen

    return bio_tags


# ---------------------------------------------------------------------------
# Evaluation-Wrapper
# ---------------------------------------------------------------------------

def evaluate_llm_predictions(
    tokens_list:    List[List[str]],
    gold_entities:  List[List[Dict[str, str]]],
    pred_entities:  List[List[Dict[str, str]]],
    parse_statuses: List[str],
) -> Dict[str, float]:
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
    from seqeval.metrics import f1_score, precision_score, recall_score

    true_bio: List[List[str]] = []
    pred_bio: List[List[str]] = []

    # Alle Samples in BIO-Sequenzen umwandeln
    for tokens, gold, pred in zip(tokens_list, gold_entities, pred_entities):
        true_bio.append(entities_to_bio(tokens, gold))
        pred_bio.append(entities_to_bio(tokens, pred))

    # Anteil der Samples, bei denen kein valides JSON geparst werden konnte
    parse_fail_rate = parse_statuses.count("failed") / max(len(parse_statuses), 1)

    return {
        "precision":              float(precision_score(true_bio, pred_bio, zero_division=0)),
        "recall":                 float(recall_score(true_bio, pred_bio, zero_division=0)),
        "f1":                     float(f1_score(true_bio, pred_bio, zero_division=0)),
        "parse_failure_rate":     parse_fail_rate,
        # Aufschluesselung nach Parse-Strategie (fuer Fehleranalyse-Kapitel)
        "parse_ok":               parse_statuses.count("ok"),
        "parse_markdown_stripped": parse_statuses.count("markdown_stripped"),
        "parse_regex_fallback":   parse_statuses.count("regex_fallback"),
        "parse_failed":           parse_statuses.count("failed"),
    }
