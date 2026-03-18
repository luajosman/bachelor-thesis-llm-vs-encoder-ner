"""
Error analysis for both encoder and decoder NER systems.

Encoder errors are categorized from BIO tag sequences.
Decoder errors cover JSON parsing failures and structural issues.

Usage:
    python -m src.evaluate.error_analysis \\
        --encoder-preds results/deberta-v3-large/test_predictions.json \\
        --decoder-preds results/qwen35-27b-lora/test_predictions.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EncoderErrorStats:
    """Categorized error counts for an encoder model."""

    boundary_errors: int = 0      # Wrong B-/I- boundary (span found but wrong extent)
    type_errors: int = 0          # Right span, wrong entity type
    missed_entities: int = 0      # Gold entity not found in predictions
    hallucinated_entities: int = 0  # Predicted entity not in gold
    total_gold: int = 0
    total_pred: int = 0
    examples: List[Dict] = field(default_factory=list)


@dataclass
class DecoderErrorStats:
    """Categorized error counts for a decoder model."""

    json_parse_failures: int = 0      # No valid JSON at all
    incomplete_json: int = 0          # JSON truncated / not closed
    wrong_schema: int = 0             # Valid JSON but not a list
    missing_fields: int = 0           # entity/type fields missing
    unknown_entity_types: int = 0     # type not in valid set
    span_mismatches: int = 0          # Entity text not found in input
    total_samples: int = 0
    examples: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: extract spans from BIO sequence
# ---------------------------------------------------------------------------


def _extract_spans(bio_tags: List[str]) -> List[Dict[str, Any]]:
    """Extract entity spans (start, end, type) from a BIO tag sequence."""
    spans: List[Dict[str, Any]] = []
    current_type: Optional[str] = None
    start: int = -1

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.append({"start": start, "end": i, "type": current_type})
            current_type = tag[2:]
            start = i
        elif tag.startswith("I-") and current_type is not None:
            pass  # continuation
        else:
            if current_type is not None:
                spans.append({"start": start, "end": i, "type": current_type})
            current_type = None
            start = -1

    if current_type is not None:
        spans.append({"start": start, "end": len(bio_tags), "type": current_type})

    return spans


# ---------------------------------------------------------------------------
# Encoder error analysis
# ---------------------------------------------------------------------------


def analyze_encoder_errors(
    tokens_list: List[List[str]],
    gold_tags: List[List[str]],
    pred_tags: List[List[str]],
    max_examples: int = 10,
) -> EncoderErrorStats:
    """Analyze prediction errors for encoder (BIO tag sequences).

    Categories:
    - **Boundary errors**: A span overlaps with a gold span but has different
      boundaries (off-by-one B/I errors).
    - **Type errors**: Exact span match but wrong entity type label.
    - **Missed entities**: Gold span not covered by any prediction.
    - **Hallucinated entities**: Predicted span not covered by any gold span.

    Parameters
    ----------
    tokens_list:
        Token sequences (for building example strings).
    gold_tags:
        Gold BIO tag sequences.
    pred_tags:
        Predicted BIO tag sequences.
    max_examples:
        Maximum number of error examples to save per category.

    Returns
    -------
    EncoderErrorStats
    """
    stats = EncoderErrorStats()

    for tokens, gold, pred in zip(tokens_list, gold_tags, pred_tags):
        gold_spans = _extract_spans(gold)
        pred_spans = _extract_spans(pred)

        stats.total_gold += len(gold_spans)
        stats.total_pred += len(pred_spans)

        # Index by (start, end) for fast lookup
        gold_by_pos = {(s["start"], s["end"]): s["type"] for s in gold_spans}
        pred_by_pos = {(s["start"], s["end"]): s["type"] for s in pred_spans}

        # Exact type errors (same span, wrong type)
        for pos, g_type in gold_by_pos.items():
            if pos in pred_by_pos:
                p_type = pred_by_pos[pos]
                if g_type != p_type:
                    stats.type_errors += 1
                    if len([e for e in stats.examples if e.get("error") == "type"]) < max_examples:
                        stats.examples.append({
                            "error": "type",
                            "tokens": tokens,
                            "span": " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type": g_type,
                            "pred_type": p_type,
                        })
            else:
                # Check for boundary errors: any pred span that overlaps?
                overlap = any(
                    not (ps[1] <= pos[0] or ps[0] >= pos[1])
                    for ps in pred_by_pos
                )
                if overlap:
                    stats.boundary_errors += 1
                    if len([e for e in stats.examples if e.get("error") == "boundary"]) < max_examples:
                        stats.examples.append({
                            "error": "boundary",
                            "tokens": tokens,
                            "gold_span": " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type": g_type,
                        })
                else:
                    stats.missed_entities += 1
                    if len([e for e in stats.examples if e.get("error") == "missed"]) < max_examples:
                        stats.examples.append({
                            "error": "missed",
                            "tokens": tokens,
                            "gold_span": " ".join(tokens[pos[0]:pos[1]]),
                            "gold_type": g_type,
                        })

        # Hallucinations: pred spans not in gold at all
        for pos, p_type in pred_by_pos.items():
            overlap = any(
                not (gs[1] <= pos[0] or gs[0] >= pos[1])
                for gs in gold_by_pos
            )
            if not overlap:
                stats.hallucinated_entities += 1
                if len([e for e in stats.examples if e.get("error") == "hallucinated"]) < max_examples:
                    stats.examples.append({
                        "error": "hallucinated",
                        "tokens": tokens,
                        "pred_span": " ".join(tokens[pos[0]:pos[1]]),
                        "pred_type": p_type,
                    })

    return stats


# ---------------------------------------------------------------------------
# Decoder error analysis
# ---------------------------------------------------------------------------

VALID_TYPES = frozenset(
    ["person", "location", "corporation", "creative-work", "group", "product"]
)


def analyze_decoder_errors(
    gold_entities: List[List[Dict]],
    pred_entities: List[List[Dict]],
    raw_outputs: List[str],
    parse_statuses: List[str],
    tokens_list: List[List[str]],
    max_examples: int = 10,
) -> DecoderErrorStats:
    """Analyze structural and content errors for decoder (LLM) predictions.

    Categories:
    - **JSON parse failures**: Status == "failed" (no valid JSON found).
    - **Incomplete JSON**: Output ends mid-JSON (last char not ``]``).
    - **Wrong schema**: Valid JSON but not a list.
    - **Missing fields**: Entries without "entity" or "type".
    - **Unknown entity types**: "type" not in valid set.
    - **Span mismatches**: Entity text not found in the input sentence.

    Parameters
    ----------
    gold_entities:
        Gold entity dicts per sentence.
    pred_entities:
        Predicted entity dicts per sentence (already parsed).
    raw_outputs:
        Raw LLM text output per sentence.
    parse_statuses:
        Parse outcome per sentence.
    tokens_list:
        Token sequences (to check span matching).
    max_examples:
        Max examples per category to save.

    Returns
    -------
    DecoderErrorStats
    """
    import json as _json
    import re

    stats = DecoderErrorStats()
    stats.total_samples = len(raw_outputs)

    for i, (raw, status, tokens) in enumerate(zip(raw_outputs, parse_statuses, tokens_list)):
        sentence = " ".join(tokens)

        # 1. Parse failures
        if status == "failed":
            stats.json_parse_failures += 1
            # Check if it looks like truncated JSON
            stripped = raw.strip()
            if stripped and stripped[-1] not in ("]", "}"):
                stats.incomplete_json += 1
                if len([e for e in stats.examples if e.get("error") == "incomplete"]) < max_examples:
                    stats.examples.append({
                        "error": "incomplete",
                        "raw_output": raw[:200],
                        "sentence": sentence[:100],
                    })
            elif len([e for e in stats.examples if e.get("error") == "failed"]) < max_examples:
                stats.examples.append({
                    "error": "failed",
                    "raw_output": raw[:200],
                    "sentence": sentence[:100],
                })
            continue

        # 2. Wrong schema (already parsed: if pred_entities[i] is empty but
        #    raw looks like a dict or scalar)
        text_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_clean, re.DOTALL)
        candidate = fence_match.group(1).strip() if fence_match else text_clean
        try:
            parsed_raw = _json.loads(candidate)
            if not isinstance(parsed_raw, list):
                stats.wrong_schema += 1
                if len([e for e in stats.examples if e.get("error") == "wrong_schema"]) < max_examples:
                    stats.examples.append({
                        "error": "wrong_schema",
                        "raw_output": raw[:200],
                        "sentence": sentence[:100],
                    })
        except Exception:
            pass

        # 3. Inspect individual predicted entities
        for ent in pred_entities[i]:
            if "entity" not in ent or "type" not in ent:
                stats.missing_fields += 1
                continue
            if ent["type"] not in VALID_TYPES:
                stats.unknown_entity_types += 1
                if len([e for e in stats.examples if e.get("error") == "unknown_type"]) < max_examples:
                    stats.examples.append({
                        "error": "unknown_type",
                        "entity": ent.get("entity"),
                        "type": ent.get("type"),
                        "sentence": sentence[:100],
                    })
            # Span mismatch: entity text not found in sentence
            if ent.get("entity", "") and ent["entity"] not in sentence:
                stats.span_mismatches += 1
                if len([e for e in stats.examples if e.get("error") == "span_mismatch"]) < max_examples:
                    stats.examples.append({
                        "error": "span_mismatch",
                        "entity": ent.get("entity"),
                        "type": ent.get("type"),
                        "sentence": sentence[:100],
                    })

    return stats


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_error_summary(
    encoder_stats: Optional[EncoderErrorStats] = None,
    decoder_stats: Optional[DecoderErrorStats] = None,
    encoder_name: str = "Encoder",
    decoder_name: str = "Decoder",
) -> None:
    """Print a comparative error summary table using rich.

    Parameters
    ----------
    encoder_stats:
        Error statistics from the encoder model.
    decoder_stats:
        Error statistics from the decoder model.
    encoder_name:
        Display name for the encoder.
    decoder_name:
        Display name for the decoder.
    """
    table = Table(
        title="Error Analysis Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Error Category", style="bold")
    if encoder_stats is not None:
        table.add_column(encoder_name, justify="right")
    if decoder_stats is not None:
        table.add_column(decoder_name, justify="right")

    def _row(label: str, enc_val, dec_val):
        row = [label]
        if encoder_stats is not None:
            row.append(str(enc_val))
        if decoder_stats is not None:
            row.append(str(dec_val))
        table.add_row(*row)

    _row("Missed Entities",
         encoder_stats.missed_entities if encoder_stats else "-",
         "-")
    _row("Hallucinated Entities",
         encoder_stats.hallucinated_entities if encoder_stats else "-",
         "-")
    _row("Boundary Errors",
         encoder_stats.boundary_errors if encoder_stats else "-",
         "-")
    _row("Type Errors",
         encoder_stats.type_errors if encoder_stats else "-",
         "-")
    _row("JSON Parse Failures",
         "-",
         decoder_stats.json_parse_failures if decoder_stats else "-")
    _row("Incomplete JSON",
         "-",
         decoder_stats.incomplete_json if decoder_stats else "-")
    _row("Wrong Schema",
         "-",
         decoder_stats.wrong_schema if decoder_stats else "-")
    _row("Missing Fields (entity/type)",
         "-",
         decoder_stats.missing_fields if decoder_stats else "-")
    _row("Unknown Entity Types",
         "-",
         decoder_stats.unknown_entity_types if decoder_stats else "-")
    _row("Span Mismatches",
         "-",
         decoder_stats.span_mismatches if decoder_stats else "-")

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_preds(path: str) -> Dict:
    """Load test_predictions.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER error analysis")
    parser.add_argument("--encoder-preds", help="Path to encoder test_predictions.json")
    parser.add_argument("--decoder-preds", help="Path to decoder test_predictions.json")
    args = parser.parse_args()

    enc_stats = None
    dec_stats = None

    if args.encoder_preds:
        enc_data = _load_preds(args.encoder_preds)
        tokens_list = [s["tokens"] for s in enc_data]
        gold_tags = [s["gold"] for s in enc_data]
        pred_tags = [s["pred"] for s in enc_data]
        enc_stats = analyze_encoder_errors(tokens_list, gold_tags, pred_tags)
        console.print(f"[green]Encoder: {len(enc_data)} samples analyzed[/green]")

    if args.decoder_preds:
        dec_data = _load_preds(args.decoder_preds)
        tokens_list_d = [s["tokens"] for s in dec_data]
        gold_entities = [s["gold_entities"] for s in dec_data]
        pred_entities = [s["pred_entities"] for s in dec_data]
        raw_outputs = [s["raw_output"] for s in dec_data]
        parse_statuses = [s["parse_status"] for s in dec_data]
        dec_stats = analyze_decoder_errors(
            gold_entities, pred_entities, raw_outputs, parse_statuses, tokens_list_d
        )
        console.print(f"[green]Decoder: {len(dec_data)} samples analyzed[/green]")

    print_error_summary(enc_stats, dec_stats)
