"""Shared strict span-level NER metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_ner_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    *,
    include_report: bool = False,
) -> Dict[str, Any]:
    """Compute strict entity-level precision, recall, and F1."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2

    metrics: Dict[str, Any] = {
        "precision": float(
            precision_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
    }
    if include_report:
        metrics["report"] = classification_report(
            y_true,
            y_pred,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
    return metrics


def compute_per_entity_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """Compute strict precision, recall, F1, and support by entity type."""
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        mode="strict",
        scheme=IOB2,
        zero_division=0,
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


def compute_macro_f1(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    """Compute unweighted macro-F1 over entity types."""
    per_entity = compute_per_entity_metrics(y_true, y_pred)
    if not per_entity:
        return 0.0
    return float(sum(v["f1"] for v in per_entity.values()) / len(per_entity))


def build_token_classification_compute_metrics(id2label: Dict[int, str]):
    """Build a Hugging Face Trainer-compatible metric callback."""

    def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
        import numpy as np

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)
        true_labels, true_predictions = decode_token_classification_predictions(
            predictions=predictions,
            labels=labels,
            id2label=id2label,
        )
        return compute_ner_metrics(true_labels, true_predictions)

    return compute_metrics


def decode_token_classification_predictions(
    *,
    predictions: Any,
    labels: Any,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Drop ignored token positions and map token-classification ids to BIO labels."""
    true_labels: List[List[str]] = []
    true_predictions: List[List[str]] = []

    for prediction, label in zip(predictions, labels):
        label_seq: List[str] = []
        pred_seq: List[str] = []
        for pred_id, label_id in zip(prediction, label):
            if int(label_id) == -100:
                continue
            label_seq.append(id2label[int(label_id)])
            pred_seq.append(id2label[int(pred_id)])
        true_labels.append(label_seq)
        true_predictions.append(pred_seq)

    return true_labels, true_predictions
