"""
NER evaluation metrics wrappers (seqeval entity-level).

All evaluation uses span-level (entity-level) metrics, NOT token-level
accuracy, because token-level accuracy is misleading when O-tags dominate.
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
# Global metrics
# ---------------------------------------------------------------------------


def compute_ner_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, float]:
    """Compute entity-level Precision, Recall, F1 using seqeval.

    Parameters
    ----------
    y_true:
        List of gold BIO tag sequences.
    y_pred:
        List of predicted BIO tag sequences.

    Returns
    -------
    Dict[str, float]
        Keys: precision, recall, f1.
        Also includes the full ``classification_report`` as a string under
        the key ``report``.
    """
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "report": report,
    }


# ---------------------------------------------------------------------------
# Per-entity-type metrics
# ---------------------------------------------------------------------------


def compute_per_entity_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """Compute Precision, Recall, F1 per entity type.

    Parameters
    ----------
    y_true:
        Gold BIO sequences.
    y_pred:
        Predicted BIO sequences.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Outer key: entity type (e.g. ``"person"``).
        Inner keys: ``precision``, ``recall``, ``f1``, ``support``.

    Examples
    --------
    >>> metrics = compute_per_entity_metrics(y_true, y_pred)
    >>> metrics["person"]["f1"]
    0.812
    """
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    per_entity: Dict[str, Dict[str, float]] = {}
    for key, val in report_dict.items():
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(val, dict):
            per_entity[key] = {
                "precision": float(val.get("precision", 0.0)),
                "recall": float(val.get("recall", 0.0)),
                "f1": float(val.get("f1-score", 0.0)),
                "support": float(val.get("support", 0)),
            }

    return per_entity


# ---------------------------------------------------------------------------
# Macro metrics (unweighted average over entity types)
# ---------------------------------------------------------------------------


def compute_macro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """Compute macro-averaged F1 (unweighted average over entity types).

    Parameters
    ----------
    y_true:
        Gold BIO sequences.
    y_pred:
        Predicted BIO sequences.

    Returns
    -------
    float
        Macro F1 score.
    """
    per_entity = compute_per_entity_metrics(y_true, y_pred)
    if not per_entity:
        return 0.0
    return float(sum(v["f1"] for v in per_entity.values()) / len(per_entity))
