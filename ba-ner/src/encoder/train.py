"""
Encoder-based NER training (token classification with BIO tagging).

Supports DeBERTa-v3-large, DeBERTa-v3-base, and BERT via YAML configs.

Usage:
    python -m src.encoder.train configs/deberta_large.yaml
    python -m src.encoder.train configs/bert_base.yaml
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from rich.console import Console
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.load_wnut17 import ID2LABEL, LABEL2ID, LABEL_LIST
from src.data.preprocess_encoder import prepare_encoder_dataset

console = Console()

# ---------------------------------------------------------------------------
# Seqeval-based metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """Compute entity-level Precision, Recall, F1 using seqeval.

    Sub-word tokens and special tokens have label -100 and are skipped.

    Parameters
    ----------
    eval_pred:
        Tuple of (logits, labels) as provided by HuggingFace Trainer.

    Returns
    -------
    Dict[str, float]
        Keys: precision, recall, f1, accuracy (all entity-level).
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_labels: List[List[str]] = []
    true_predictions: List[List[str]] = []

    for prediction, label in zip(predictions, labels):
        true_label_seq: List[str] = []
        true_pred_seq: List[str] = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            true_label_seq.append(ID2LABEL[label_id])
            true_pred_seq.append(ID2LABEL[pred_id])
        true_labels.append(true_label_seq)
        true_predictions.append(true_pred_seq)

    report = classification_report(true_labels, true_predictions, output_dict=False, zero_division=0)
    console.print(f"\n[dim]{report}[/dim]")

    return {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_encoder(config_path: str) -> Dict[str, Any]:
    """Train an encoder model for NER based on a YAML config.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file (e.g. ``configs/deberta_large.yaml``).

    Returns
    -------
    Dict[str, Any]
        Results dict with test_f1, train_runtime, and model output dir.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold green]Training: {cfg['experiment_name']}[/bold green]")

    # ---- Reproducibility ----
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Data ----
    console.print("[cyan]Loading and tokenizing WNUT-2017...[/cyan]")
    tokenized_dataset, tokenizer, label_list = prepare_encoder_dataset(
        model_name=cfg["model_name"],
        max_length=cfg.get("max_length", 128),
    )

    # ---- Model ----
    console.print(f"[cyan]Loading model: {cfg['model_name']}[/cyan]")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ---- Output dirs ----
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("epochs", 10),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 32),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=cfg.get("fp16", True) and torch.cuda.is_available(),
        seed=seed,
        logging_steps=50,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=cfg.get("experiment_name"),
    )

    # ---- Data collator (dynamic padding) ----
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ---- Train ----
    console.print("[bold yellow]Starting training...[/bold yellow]")
    train_start = time.perf_counter()
    train_result = trainer.train()
    train_runtime = time.perf_counter() - train_start

    # ---- Save best model ----
    console.print(f"[green]Saving best model to {best_model_dir}[/green]")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # ---- Evaluate on test set ----
    console.print("[cyan]Evaluating on test split...[/cyan]")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    test_f1 = test_metrics.get("eval_f1", 0.0)
    test_precision = test_metrics.get("eval_precision", 0.0)
    test_recall = test_metrics.get("eval_recall", 0.0)

    console.print(f"\n[bold green]Test F1: {test_f1:.4f}[/bold green]")
    console.print(f"Precision: {test_precision:.4f}  Recall: {test_recall:.4f}")
    console.print(f"Training time: {train_runtime:.1f}s")

    # ---- Save results ----
    results: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_name": cfg["model_name"],
        "model_type": "encoder",
        "test_f1": float(test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "train_runtime_seconds": float(train_runtime),
        "best_model_dir": str(best_model_dir),
    }

    results_file = output_dir / "results.yaml"
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    console.print(f"Results saved to {results_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.encoder.train <config.yaml>[/red]")
        sys.exit(1)
    train_encoder(sys.argv[1])
