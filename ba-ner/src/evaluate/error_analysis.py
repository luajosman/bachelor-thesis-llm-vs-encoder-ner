"""
Qualitative error analysis for MultiNERD English encoder and decoder NER.

Analyzes saved test_predictions.json files and categorizes errors for the two
final paradigms:

Encoder errors from BIO tag sequences:
  - Boundary Errors:      wrong span boundaries
  - Type Errors:          correct span with wrong entity type
  - Missed Entities:      gold entity not predicted
  - Hallucinations:       predicted entity without a gold overlap

Decoder structural errors from LLM output:
  - JSON Parse Failures:  no valid JSON array extracted
  - Incomplete JSON:      output likely truncated
  - Wrong Schema:         valid JSON with the wrong top-level shape
  - Missing Fields:       missing start_token/end_token/text/type fields
  - Unknown Types:        type outside the MultiNERD taxonomy
  - Invalid Offsets:      token offsets outside the presented token list
  - Text Mismatches:      text does not match the referenced token span
  - Overlapping Spans:    output contains spans that overlap in BIO space

Usage:
    python -m src.evaluate.error_analysis \\
        --encoder-preds results/multinerd/deberta-v3-large/test_predictions.json \\
        --decoder-preds results/multinerd/qwen35-27b-qlora/test_predictions.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from rich.console import Console
from rich.table import Table

from src.data.dataset_loader import get_dataset_info

console = Console()


# ---------------------------------------------------------------------------
# Datenklassen für Fehler-Statistiken
# ---------------------------------------------------------------------------

@dataclass
class EncoderErrorStats:
    """Fehler-Zähler für Encoder-Modelle (aus BIO-Tag-Vergleich).

    Attributes:
        boundary_errors:       Falsche Span-Grenzen (überlappende Spans gefunden).
        type_errors:           Exakt gleiche Span, aber anderer Entity-Typ.
        missed_entities:       Gold-Entity ohne jegliche Vorhersage (False Negative).
        hallucinated_entities: Vorhersage ohne Gold-Entsprechung (False Positive).
        total_gold:            Gesamtzahl der Gold-Entities im Test-Set.
        total_pred:            Gesamtzahl der vorhergesagten Entities.
        examples:              Gespeicherte Beispiele pro Fehlerkategorie.
    """
    boundary_errors:       int       = 0
    type_errors:           int       = 0
    missed_entities:       int       = 0
    hallucinated_entities: int       = 0
    total_gold:            int       = 0
    total_pred:            int       = 0
    examples:              List[Dict] = field(default_factory=list)


@dataclass
class DecoderErrorStats:
    """Fehler-Zähler für Decoder-Modelle (strukturelle LLM-Output-Fehler).

    Attributes:
        json_parse_failures:  Kein valides JSON im generierten Text gefunden.
        incomplete_json:      JSON endet nicht mit ']' (wurde abgeschnitten).
        wrong_schema:         Valides JSON, aber kein Array (z.B. Dict).
        missing_fields:       Entities ohne start_token/end_token/text/type-Felder.
        unknown_entity_types: 'type'-Wert nicht in der Datensatz-Taxonomie.
        invalid_items:        Nicht-Dict-Eintraege im JSON-Array.
        invalid_offsets:      Ungueltige tokenbasierte Offsets.
        text_mismatches:      Text stimmt nicht zur referenzierten Tokenrange.
        overlapping_spans:    Ueberlappende Spans, die nicht in BIO darstellbar sind.
        total_samples:        Gesamtanzahl analysierter Test-Samples.
        examples:             Gespeicherte Fehler-Beispiele.
    """
    json_parse_failures:  int       = 0
    incomplete_json:      int       = 0
    wrong_schema:         int       = 0
    missing_fields:       int       = 0
    unknown_entity_types: int       = 0
    invalid_items:        int       = 0
    invalid_offsets:      int       = 0
    text_mismatches:      int       = 0
    overlapping_spans:    int       = 0
    total_samples:        int       = 0
    examples:             List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hilfsfunktion: BIO-Sequenz → Span-Liste
# ---------------------------------------------------------------------------

def _extract_spans(bio_tags: List[str]) -> List[Dict[str, Any]]:
    """Extrahiert Entity-Spans (Start, Ende, Typ) aus einer BIO-Tag-Sequenz.

    Gibt für jede erkannte Entity ein Dict mit 'start' (inklusiv),
    'end' (exklusiv) und 'type' zurück. Damit kann man Spans auf
    Überlappung prüfen ohne Strings vergleichen zu müssen.

    Args:
        bio_tags: BIO-Tag-Liste, z.B. ["O", "B-person", "I-person", "O"].

    Returns:
        Liste von Span-Dicts: [{"start": 1, "end": 3, "type": "person"}].
    """
    spans: List[Dict[str, Any]] = []
    current_type: Optional[str] = None
    start: int = -1

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            # Offene Entity abschließen
            if current_type is not None:
                spans.append({"start": start, "end": i, "type": current_type})
            current_type = tag[2:]  # "B-location" → "location"
            start = i
        elif tag.startswith("I-") and current_type is not None:
            pass  # Fortsetzung — nichts tun
        else:
            if current_type is not None:
                spans.append({"start": start, "end": i, "type": current_type})
            current_type = None
            start = -1

    # Letzte Entity am Sequenzende abschließen
    if current_type is not None:
        spans.append({"start": start, "end": len(bio_tags), "type": current_type})

    return spans


# ---------------------------------------------------------------------------
# Encoder-Fehleranalyse
# ---------------------------------------------------------------------------

def analyze_encoder_errors(
    tokens_list:  List[List[str]],
    gold_tags:    List[List[str]],
    pred_tags:    List[List[str]],
    max_examples: int = 10,
) -> EncoderErrorStats:
    """Analysiert Vorhersagefehler für Encoder-Modelle (BIO-Tag-Vergleich).

    Vergleicht Gold- und Pred-Spans auf drei Ebenen:
      1. Exakt übereinstimmende Spans mit falschem Typ → Type Error
      2. Überlappende aber nicht deckungsgleiche Spans → Boundary Error
      3. Gold-Span ohne jegliche Überlappung → Missed Entity
      4. Pred-Span ohne jegliche Überlappung mit Gold → Hallucination

    Args:
        tokens_list:  Token-Listen (für lesbare Beispiel-Ausgaben).
        gold_tags:    Gold-BIO-Sequenzen.
        pred_tags:    Vorhergesagte BIO-Sequenzen.
        max_examples: Maximale Anzahl gespeicherter Beispiele pro Kategorie.

    Returns:
        EncoderErrorStats mit allen Zählern und Beispielen.
    """
    stats = EncoderErrorStats()

    for tokens, gold, pred in zip(tokens_list, gold_tags, pred_tags):
        gold_spans = _extract_spans(gold)
        pred_spans = _extract_spans(pred)

        stats.total_gold += len(gold_spans)
        stats.total_pred += len(pred_spans)

        # Span-Positionen für schnelle Suche als Dict indexieren
        gold_by_pos = {(s["start"], s["end"]): s["type"] for s in gold_spans}
        pred_by_pos = {(s["start"], s["end"]): s["type"] for s in pred_spans}

        # --- Für jede Gold-Entity den Fehlertyp bestimmen ---
        for pos, g_type in gold_by_pos.items():
            if pos in pred_by_pos:
                # Gleiche Position gefunden → Typ-Fehler?
                p_type = pred_by_pos[pos]
                if g_type != p_type:
                    stats.type_errors += 1
                    if len([e for e in stats.examples if e.get("error") == "type"]) < max_examples:
                        stats.examples.append({
                            "error":      "type",
                            "tokens":     tokens,
                            "span":       " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type":  g_type,
                            "pred_type":  p_type,
                        })
            else:
                # Keine exakte Position → auf Überlappung prüfen
                overlap = any(
                    not (ps[1] <= pos[0] or ps[0] >= pos[1])
                    for ps in pred_by_pos
                )
                if overlap:
                    # Überlappung vorhanden → Boundary Error (falsche Grenzen)
                    stats.boundary_errors += 1
                    if len([e for e in stats.examples if e.get("error") == "boundary"]) < max_examples:
                        stats.examples.append({
                            "error":      "boundary",
                            "tokens":     tokens,
                            "gold_span":  " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type":  g_type,
                        })
                else:
                    # Keine Überlappung → Entity komplett übersehen
                    stats.missed_entities += 1
                    if len([e for e in stats.examples if e.get("error") == "missed"]) < max_examples:
                        stats.examples.append({
                            "error":     "missed",
                            "tokens":    tokens,
                            "gold_span": " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type": g_type,
                        })

        # --- Halluzinationen: Pred-Spans ohne jegliche Gold-Überlappung ---
        for pos, p_type in pred_by_pos.items():
            overlap = any(
                not (gs[1] <= pos[0] or gs[0] >= pos[1])
                for gs in gold_by_pos
            )
            if not overlap:
                stats.hallucinated_entities += 1
                if len([e for e in stats.examples if e.get("error") == "hallucinated"]) < max_examples:
                    stats.examples.append({
                        "error":     "hallucinated",
                        "tokens":    tokens,
                        "pred_span": " ".join(tokens[pos[0]:pos[1]]),
                        "pred_type": p_type,
                    })

    return stats


# ---------------------------------------------------------------------------
# Decoder-Fehleranalyse
# ---------------------------------------------------------------------------

def analyze_decoder_errors(
    gold_entities:  List[List[Dict]],
    pred_entities:  List[List[Dict]],
    raw_outputs:    List[str],
    parse_statuses: List[str],
    tokens_list:    List[List[str]],
    valid_types:    Optional[FrozenSet[str]] = None,
    parse_diagnostics: Optional[List[Dict[str, int]]] = None,
    max_examples:   int = 10,
) -> DecoderErrorStats:
    """Analysiert strukturelle und inhaltliche Fehler im LLM-Output.

    Unterscheidet zwei Fehlerebenen:
      - Parsing-Ebene: Konnte überhaupt valides JSON extrahiert werden?
      - Inhalts-Ebene: Sind die geparsten Entities korrekt strukturiert?

    Args:
        gold_entities:  Gold-Entity-Dicts pro Satz.
        pred_entities:  Vorhergesagte Entity-Dicts pro Satz (bereits geparst).
        raw_outputs:    Roher LLM-Output pro Satz (für Debugging).
        parse_statuses: Parse-Ergebnis pro Satz ("ok", "failed", ...).
        tokens_list:    Token-Listen fuer Offset- und Textkonsistenz-Checks.
        valid_types:    Erlaubte Entity-Typen (datensatzabhaengig). Wenn None,
                        wird der Unknown-Type-Check uebersprungen.
        max_examples:   Max. Anzahl gespeicherter Fehler-Beispiele.

    Returns:
        DecoderErrorStats mit allen Zählern und Beispielen.
    """
    import json as _json
    import re

    stats = DecoderErrorStats()
    stats.total_samples = len(raw_outputs)

    diagnostics_list = parse_diagnostics or [{} for _ in raw_outputs]

    for i, (raw, status, tokens, diagnostics) in enumerate(
        zip(raw_outputs, parse_statuses, tokens_list, diagnostics_list)
    ):
        sentence = " ".join(tokens)

        stats.wrong_schema += int(diagnostics.get("wrong_schema", 0))
        stats.missing_fields += int(diagnostics.get("missing_fields", 0))
        stats.unknown_entity_types += int(diagnostics.get("unknown_types", 0))
        stats.invalid_items += int(diagnostics.get("invalid_items", 0))
        stats.invalid_offsets += int(diagnostics.get("invalid_offsets", 0))
        stats.text_mismatches += int(diagnostics.get("text_mismatches", 0))
        stats.overlapping_spans += int(diagnostics.get("overlaps", 0))

        # --- Parse-Fehler-Kategorien ---
        if status == "failed":
            stats.json_parse_failures += 1
            stripped = raw.strip()
            # Abgeschnittenes JSON erkennen: endet nicht mit ']' oder '}'
            if stripped and stripped[-1] not in ("]", "}"):
                stats.incomplete_json += 1
                if len([e for e in stats.examples if e.get("error") == "incomplete"]) < max_examples:
                    stats.examples.append({
                        "error":      "incomplete",
                        "raw_output": raw[:200],
                        "sentence":   sentence[:100],
                    })
            elif len([e for e in stats.examples if e.get("error") == "failed"]) < max_examples:
                stats.examples.append({
                    "error":      "failed",
                    "raw_output": raw[:200],
                    "sentence":   sentence[:100],
                })
            continue  # Kein valides JSON → Inhalts-Checks überspringen

        if not diagnostics:
            # Backward-compatible best effort for prediction files without
            # parser diagnostics.
            text_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_clean, re.DOTALL)
            candidate = fence_match.group(1).strip() if fence_match else text_clean
            try:
                parsed_raw = _json.loads(candidate)
                if not isinstance(parsed_raw, list):
                    stats.wrong_schema += 1
                    if len([e for e in stats.examples if e.get("error") == "wrong_schema"]) < max_examples:
                        stats.examples.append({
                            "error":      "wrong_schema",
                            "raw_output": raw[:200],
                            "sentence":   sentence[:100],
                        })
            except Exception:
                pass
        elif diagnostics.get("wrong_schema", 0):
            if len([e for e in stats.examples if e.get("error") == "wrong_schema"]) < max_examples:
                stats.examples.append({
                    "error":      "wrong_schema",
                    "raw_output": raw[:200],
                    "sentence":   sentence[:100],
                })

        for key, label in (
            ("invalid_offsets", "invalid_offsets"),
            ("text_mismatches", "text_mismatch"),
            ("overlaps", "overlap"),
            ("unknown_types", "unknown_type"),
            ("missing_fields", "missing_fields"),
            ("invalid_items", "invalid_item"),
        ):
            if (
                diagnostics.get(key, 0)
                and len([e for e in stats.examples if e.get("error") == label]) < max_examples
            ):
                stats.examples.append({
                    "error":      label,
                    "count":      int(diagnostics.get(key, 0)),
                    "raw_output": raw[:200],
                    "sentence":   sentence[:100],
                })

        if not diagnostics:
            # Backward-compatible best effort for prediction files without
            # parser diagnostics. Active decoder outputs are validated by the
            # parser before they reach pred_entities.
            occupied = [False] * len(tokens)
            required = {"start_token", "end_token", "text", "type"}
            for ent in pred_entities[i]:
                if not isinstance(ent, dict) or any(field not in ent for field in required):
                    stats.missing_fields += 1
                    continue
                start = ent.get("start_token")
                end = ent.get("end_token")
                etype = ent.get("type")
                if valid_types is not None and etype not in valid_types:
                    stats.unknown_entity_types += 1
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not (0 <= start < end <= len(tokens))
                ):
                    stats.invalid_offsets += 1
                    continue
                if ent.get("text") != " ".join(tokens[start:end]):
                    stats.text_mismatches += 1
                if any(occupied[start:end]):
                    stats.overlapping_spans += 1
                    continue
                for pos in range(start, end):
                    occupied[pos] = True

    return stats


# ---------------------------------------------------------------------------
# Vergleichstabelle ausgeben
# ---------------------------------------------------------------------------

def print_error_summary(
    encoder_stats: Optional[EncoderErrorStats] = None,
    decoder_stats: Optional[DecoderErrorStats] = None,
    encoder_name:  str = "Encoder",
    decoder_name:  str = "Decoder",
) -> None:
    """Gibt eine Vergleichstabelle der Fehlertypen für Encoder und Decoder aus.

    Hilft beim Verfassen des Fehleranalyse-Kapitels der Bachelorarbeit:
    zeigt auf einen Blick, welche Fehlertypen bei welchem Paradigma dominieren.

    Args:
        encoder_stats: Fehler-Statistiken des Encoder-Modells (optional).
        decoder_stats: Fehler-Statistiken des Decoder-Modells (optional).
        encoder_name:  Anzeigename des Encoder-Experiments.
        decoder_name:  Anzeigename des Decoder-Experiments.
    """
    table = Table(
        title="Fehleranalyse: Encoder vs. Decoder",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Fehlerkategorie", style="bold")
    if encoder_stats is not None:
        table.add_column(encoder_name, justify="right")
    if decoder_stats is not None:
        table.add_column(decoder_name, justify="right")

    def _row(label: str, enc_val, dec_val):
        """Hilfsfunktion: Tabellenzeile mit optionalen Encoder-/Decoder-Werten."""
        row = [label]
        if encoder_stats is not None:
            row.append(str(enc_val))
        if decoder_stats is not None:
            row.append(str(dec_val))
        table.add_row(*row)

    # Encoder-spezifische Fehler
    _row("Missed Entities",          encoder_stats.missed_entities if encoder_stats else "-",          "-")
    _row("Halluzinierte Entities",   encoder_stats.hallucinated_entities if encoder_stats else "-",    "-")
    _row("Boundary Errors",          encoder_stats.boundary_errors if encoder_stats else "-",          "-")
    _row("Type Errors",              encoder_stats.type_errors if encoder_stats else "-",              "-")
    # Decoder-spezifische Fehler
    _row("JSON Parse Failures",      "-",  decoder_stats.json_parse_failures if decoder_stats else "-")
    _row("Incomplete JSON",          "-",  decoder_stats.incomplete_json if decoder_stats else "-")
    _row("Wrong Schema",             "-",  decoder_stats.wrong_schema if decoder_stats else "-")
    _row("Missing Fields",           "-",  decoder_stats.missing_fields if decoder_stats else "-")
    _row("Unknown Entity Types",     "-",  decoder_stats.unknown_entity_types if decoder_stats else "-")
    _row("Invalid JSON Items",       "-",  decoder_stats.invalid_items if decoder_stats else "-")
    _row("Invalid Token Offsets",    "-",  decoder_stats.invalid_offsets if decoder_stats else "-")
    _row("Text/Token Mismatches",    "-",  decoder_stats.text_mismatches if decoder_stats else "-")
    _row("Overlapping Spans",        "-",  decoder_stats.overlapping_spans if decoder_stats else "-")

    console.print(table)


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

def _load_preds(path: str) -> List[Dict]:
    """Lädt eine gespeicherte test_predictions.json-Datei."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER-Fehleranalyse (Encoder vs. Decoder)")
    parser.add_argument("--encoder-preds", help="Pfad zur Encoder test_predictions.json")
    parser.add_argument("--decoder-preds", help="Pfad zur Decoder test_predictions.json")
    args = parser.parse_args()

    info = get_dataset_info()
    valid_types: FrozenSet[str] = frozenset(info.entity_types)
    console.print(f"[cyan]Valid MultiNERD entity types: {sorted(valid_types)}[/cyan]")

    enc_stats = None
    dec_stats = None

    if args.encoder_preds:
        enc_data     = _load_preds(args.encoder_preds)
        tokens_list  = [s["tokens"] for s in enc_data]
        gold_tags    = [s["gold"] for s in enc_data]
        pred_tags    = [s["pred"] for s in enc_data]
        enc_stats    = analyze_encoder_errors(tokens_list, gold_tags, pred_tags)
        console.print(f"[green]Encoder: {len(enc_data)} Samples analysiert[/green]")

    if args.decoder_preds:
        dec_data       = _load_preds(args.decoder_preds)
        tokens_list_d  = [s["tokens"] for s in dec_data]
        gold_entities  = [s["gold_entities"] for s in dec_data]
        pred_entities  = [s["pred_entities"] for s in dec_data]
        raw_outputs    = [s["raw_output"] for s in dec_data]
        parse_statuses = [s["parse_status"] for s in dec_data]
        parse_diagnostics = [s.get("parse_diagnostics", {}) for s in dec_data]
        dec_stats      = analyze_decoder_errors(
            gold_entities,
            pred_entities,
            raw_outputs,
            parse_statuses,
            tokens_list_d,
            valid_types=valid_types,
            parse_diagnostics=parse_diagnostics,
        )
        console.print(f"[green]Decoder: {len(dec_data)} Samples analysiert[/green]")

    print_error_summary(enc_stats, dec_stats)
