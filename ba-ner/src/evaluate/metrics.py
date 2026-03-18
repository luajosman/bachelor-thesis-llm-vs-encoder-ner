"""
metrics.py — NER-Evaluations-Metriken (seqeval, Entity-Level)

Alle Metriken werden auf Span-Ebene berechnet, NICHT auf Token-Ebene.
Token-Level-Accuracy wäre irreführend, weil "O"-Tags dominieren und
selbst ein Modell, das alles als "O" vorhersagt, ~80% Accuracy erreicht.

seqeval-Standard für NER:
  - Eine Entity gilt nur als korrekt, wenn Typ UND beide Grenzen exakt stimmen.
  - Precision = korrekte Predictions / alle Predictions
  - Recall    = korrekte Predictions / alle Gold-Entities
  - F1        = harmonisches Mittel aus P und R
"""

from __future__ import annotations

from typing import Dict, List

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    performance_measure,
)
from seqeval.scheme import IOB2


# ---------------------------------------------------------------------------
# Globale Metriken (Micro-Average über alle Entities)
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, float]:
    """Berechnet entity-level Precision, Recall und F1 mit seqeval.

    Micro-Average: jede einzelne Entity trägt gleichmäßig bei
    (unabhängig von ihrem Typ). Das ist der Standard-F1 in NER-Papers.

    Args:
        y_true: Gold-BIO-Tag-Sequenzen (eine Liste pro Satz).
        y_pred: Vorhergesagte BIO-Tag-Sequenzen.

    Returns:
        Dict mit 'precision', 'recall', 'f1' und 'report' (formatierter String).
    """
    p      = precision_score(y_true, y_pred, zero_division=0)
    r      = recall_score(y_true, y_pred, zero_division=0)
    f      = f1_score(y_true, y_pred, zero_division=0)
    # classification_report gibt den vollständigen Report als formatierten String
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f),
        "report":    report,   # Für die Ausgabe in der Konsole / Arbeit
    }


# ---------------------------------------------------------------------------
# Metriken pro Entity-Typ
# ---------------------------------------------------------------------------

def compute_per_entity_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """Berechnet Precision, Recall und F1 separat für jeden Entity-Typ.

    Wird für die Heatmap in compare_all.py und für die detaillierte
    Fehleranalyse im Bachelorarbeit-Kapitel 5 benötigt.

    Args:
        y_true: Gold-BIO-Sequenzen.
        y_pred: Vorhergesagte BIO-Sequenzen.

    Returns:
        Dict der Form:
            {
                "person":   {"precision": 0.85, "recall": 0.80, "f1": 0.82, "support": 42},
                "location": {...},
                ...
            }

    Beispiel:
        >>> metrics = compute_per_entity_metrics(y_true, y_pred)
        >>> metrics["person"]["f1"]
        0.812
    """
    # output_dict=True gibt den Report als Dict zurück statt als String
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    per_entity: Dict[str, Dict[str, float]] = {}
    for key, val in report_dict.items():
        # Durchschnittswerte überspringen; nur einzelne Typen behalten
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(val, dict):
            per_entity[key] = {
                "precision": float(val.get("precision", 0.0)),
                "recall":    float(val.get("recall", 0.0)),
                "f1":        float(val.get("f1-score", 0.0)),
                "support":   float(val.get("support", 0)),
            }

    return per_entity


# ---------------------------------------------------------------------------
# Macro-F1 (ungewichteter Durchschnitt über Entity-Typen)
# ---------------------------------------------------------------------------

def compute_macro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """Berechnet den Macro-F1 als ungewichteten Durchschnitt über Entity-Typen.

    Im Gegensatz zum Micro-F1 gewichtet Macro-F1 alle Typen gleich,
    unabhängig von ihrer Häufigkeit. Sinnvoll wenn seltene Typen
    (z.B. "creative-work") nicht von häufigen (z.B. "person") dominiert
    werden sollen.

    Args:
        y_true: Gold-BIO-Sequenzen.
        y_pred: Vorhergesagte BIO-Sequenzen.

    Returns:
        Macro-F1 als float.
    """
    per_entity = compute_per_entity_metrics(y_true, y_pred)
    if not per_entity:
        return 0.0  # Keine Entities im Datensatz → F1 = 0

    # Arithmetisches Mittel der F1-Werte über alle Typen
    return float(sum(v["f1"] for v in per_entity.values()) / len(per_entity))
