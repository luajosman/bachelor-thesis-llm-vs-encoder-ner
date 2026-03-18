"""
train.py — Encoder-Training für NER (Token-Klassifikation mit BIO-Tagging)

Unterstützte Modelle: DeBERTa-v3-large, DeBERTa-v3-base, BERT-base-cased.
Die Konfiguration (Lernrate, Batch-Größe, Epochen etc.) kommt aus einer YAML-Datei.

Beim Encoder-Ansatz wird NER als Sequenz-Labeling formuliert: Jedes Token
bekommt ein Label aus {O, B-person, I-person, B-location, ...}. Der
Klassifikationskopf (Linear-Schicht) sitzt auf dem letzten Hidden-State
jedes Tokens.

Verwendung:
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
# Evaluations-Metrik: Entity-Level F1 via seqeval
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """Berechnet entity-level Precision, Recall und F1 mit seqeval.

    Wird vom HuggingFace Trainer nach jeder Evaluierungs-Epoche aufgerufen.
    Tokens mit label_id == -100 (Subword-Fortsetzungen, Sondertokens) werden
    herausgefiltert, da sie kein echtes Label tragen.

    Wichtig: seqeval wertet auf Span-Ebene aus, nicht auf Token-Ebene.
    Eine Entity gilt nur als korrekt, wenn Typ UND beide Grenzen stimmen.

    Args:
        eval_pred: Tuple (logits, labels) vom HuggingFace Trainer.

    Returns:
        Dict mit 'precision', 'recall', 'f1' (alle Entity-Level).
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    logits, labels = eval_pred
    # Aus Logits die wahrscheinlichste Klasse pro Token wählen
    predictions = np.argmax(logits, axis=2)

    true_labels: List[List[str]] = []
    true_predictions: List[List[str]] = []

    for prediction, label in zip(predictions, labels):
        true_label_seq: List[str] = []
        true_pred_seq: List[str] = []
        for pred_id, label_id in zip(prediction, label):
            # -100 bedeutet: Subword-Token oder Sondertoken → überspringen
            if label_id == -100:
                continue
            # Integer-IDs in Label-Strings umwandeln (z.B. 9 → "B-person")
            true_label_seq.append(ID2LABEL[label_id])
            true_pred_seq.append(ID2LABEL[pred_id])
        true_labels.append(true_label_seq)
        true_predictions.append(true_pred_seq)

    # Vollständigen Classification-Report ausgeben (nur zur Information)
    report = classification_report(true_labels, true_predictions, output_dict=False, zero_division=0)
    console.print(f"\n[dim]{report}[/dim]")

    return {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall":    recall_score(true_labels, true_predictions, zero_division=0),
        "f1":        f1_score(true_labels, true_predictions, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_encoder(config_path: str) -> Dict[str, Any]:
    """Trainiert ein Encoder-Modell für NER auf Basis einer YAML-Config.

    Ablauf:
      1. Config laden und Seeds setzen (Reproduzierbarkeit)
      2. Datensatz tokenisieren und Labels ausrichten
      3. Modell mit Token-Klassifikationskopf laden
      4. HuggingFace Trainer mit Early Stopping konfigurieren
      5. Training starten, bestes Modell speichern
      6. Test-Set auswerten und Ergebnisse als YAML speichern

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Dict mit test_f1, test_precision, test_recall, train_runtime_seconds.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold green]Training: {cfg['experiment_name']}[/bold green]")

    # --- Reproduzierbarkeit: alle Seeds auf denselben Wert setzen ---
    seed: int = cfg.get("seed", 42)
    set_seed(seed)          # transformers-interner Seed
    random.seed(seed)       # Python-Zufallsgenerator
    np.random.seed(seed)    # NumPy
    torch.manual_seed(seed) # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU (alle Devices)

    # --- Datensatz laden und tokenisieren ---
    console.print("[cyan]Lade und tokenisiere WNUT-2017...[/cyan]")
    tokenized_dataset, tokenizer, label_list = prepare_encoder_dataset(
        model_name=cfg["model_name"],
        max_length=cfg.get("max_length", 128),
    )

    # --- Modell laden ---
    # ignore_mismatched_sizes=True erlaubt es, einen vortrainierten
    # Encoder mit einem neuen Klassifikationskopf (13 Klassen) zu laden.
    console.print(f"[cyan]Lade Modell: {cfg['model_name']}[/cyan]")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(LABEL_LIST),   # 13 BIO-Labels für WNUT-2017
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # --- Ausgabeverzeichnisse anlegen ---
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    # --- TrainingArguments konfigurieren ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("epochs", 10),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 32),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        # Nach jeder Epoche evaluieren und Checkpoint speichern
        eval_strategy="epoch",
        save_strategy="epoch",
        # Am Ende wird das Modell mit dem besten Validation-F1 geladen
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # fp16 nur aktivieren, wenn eine GPU verfügbar ist
        fp16=cfg.get("fp16", True) and torch.cuda.is_available(),
        seed=seed,
        logging_steps=50,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=cfg.get("experiment_name"),
    )

    # --- DataCollator: dynamisches Padding pro Batch ---
    # Statt alle Sätze auf max_length zu padden, wird nur bis zum
    # längsten Satz im jeweiligen Batch gepaddet → weniger Rechenaufwand.
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # --- Trainer zusammenbauen ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Early Stopping: Training stoppt, wenn sich F1 3 Epochen nicht verbessert
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- Training starten ---
    console.print("[bold yellow]Starte Training...[/bold yellow]")
    train_start = time.perf_counter()
    train_result = trainer.train()
    train_runtime = time.perf_counter() - train_start

    # --- Bestes Modell speichern ---
    console.print(f"[green]Speichere bestes Modell nach {best_model_dir}[/green]")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # --- Evaluation auf dem Test-Set ---
    console.print("[cyan]Evaluiere auf Test-Split...[/cyan]")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    test_f1        = test_metrics.get("eval_f1", 0.0)
    test_precision = test_metrics.get("eval_precision", 0.0)
    test_recall    = test_metrics.get("eval_recall", 0.0)

    console.print(f"\n[bold green]Test F1: {test_f1:.4f}[/bold green]")
    console.print(f"Precision: {test_precision:.4f}  Recall: {test_recall:.4f}")
    console.print(f"Trainingszeit: {train_runtime:.1f}s")

    # --- Ergebnisse als YAML speichern (für compare_all.py) ---
    results: Dict[str, Any] = {
        "experiment_name":      cfg["experiment_name"],
        "model_name":           cfg["model_name"],
        "model_type":           "encoder",
        "test_f1":              float(test_f1),
        "test_precision":       float(test_precision),
        "test_recall":          float(test_recall),
        "train_runtime_seconds": float(train_runtime),
        "best_model_dir":       str(best_model_dir),
    }

    results_file = output_dir / "results.yaml"
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    console.print(f"Ergebnisse gespeichert: {results_file}")

    return results


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Verwendung: python -m src.encoder.train <config.yaml>[/red]")
        sys.exit(1)
    train_encoder(sys.argv[1])
