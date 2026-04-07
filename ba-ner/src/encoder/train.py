"""
train.py — Encoder-Training fuer NER (Token-Klassifikation mit BIO-Tagging)

Unterstuetzte Modelle: DeBERTa-v3-base, DeBERTa-v3-large.
Unterstuetzte Datensaetze: MultiNERD (englisch), WNUT-2017.
Die Konfiguration (Lernrate, Batch-Groesse, Epochen etc.) kommt aus einer YAML-Datei.

Beim Encoder-Ansatz wird NER als Sequenz-Labeling formuliert: Jedes Token
bekommt ein Label aus {O, B-PER, I-PER, B-LOC, ...}. Der
Klassifikationskopf (Linear-Schicht) sitzt auf dem letzten Hidden-State
jedes Tokens.

Verwendung:
    python -m src.encoder.train configs/deberta_base.yaml
    python -m src.encoder.train configs/deberta_large.yaml
    python -m src.encoder.train configs/deberta_base.yaml --dataset wnut_17
"""

from __future__ import annotations

import argparse
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

from src.data.dataset_loader import DatasetInfo, get_dataset_info
from src.data.preprocess_encoder import prepare_encoder_dataset

console = Console()


# ---------------------------------------------------------------------------
# Evaluations-Metrik: Entity-Level F1 via seqeval
# ---------------------------------------------------------------------------

def build_compute_metrics(info: DatasetInfo):
    """Baut eine compute_metrics-Funktion mit dem richtigen id2label-Mapping.

    Args:
        info: DatasetInfo mit id2label-Mapping fuer den aktiven Datensatz.

    Returns:
        Callable fuer den HuggingFace Trainer (eval_pred -> dict).
    """
    id2label = info.id2label

    def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
        """Berechnet entity-level Precision, Recall und F1 mit seqeval.

        Tokens mit label_id == -100 (Subword-Fortsetzungen, Sondertokens) werden
        herausgefiltert, da sie kein echtes Label tragen.

        seqeval wertet auf Span-Ebene aus, nicht auf Token-Ebene.
        Eine Entity gilt nur als korrekt, wenn Typ UND beide Grenzen stimmen.
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
                true_label_seq.append(id2label[label_id])
                true_pred_seq.append(id2label[pred_id])
            true_labels.append(true_label_seq)
            true_predictions.append(true_pred_seq)

        report = classification_report(
            true_labels, true_predictions, output_dict=False, zero_division=0
        )
        console.print(f"\n[dim]{report}[/dim]")

        return {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall":    recall_score(true_labels, true_predictions, zero_division=0),
            "f1":        f1_score(true_labels, true_predictions, zero_division=0),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_encoder(config_path: str, dataset_override: str | None = None) -> Dict[str, Any]:
    """Trainiert ein Encoder-Modell fuer NER auf Basis einer YAML-Config.

    Ablauf:
      1. Config laden und Seeds setzen (Reproduzierbarkeit)
      2. Datensatz tokenisieren und Labels ausrichten
      3. Modell mit Token-Klassifikationskopf laden
      4. HuggingFace Trainer mit Early Stopping konfigurieren
      5. Training starten, bestes Modell speichern
      6. Test-Set auswerten und Ergebnisse als YAML speichern

    Args:
        config_path:      Pfad zur YAML-Konfigurationsdatei.
        dataset_override: Optionaler Datensatz-Override (z.B. "wnut_17").

    Returns:
        Dict mit test_f1, test_precision, test_recall, train_runtime_seconds.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Datensatz bestimmen: CLI-Override > Config > Default
    dataset_name = dataset_override or cfg.get("dataset", "multinerd")
    dataset_language = cfg.get("dataset_language", "en")

    console.rule(f"[bold green]Training: {cfg['experiment_name']} auf {dataset_name}[/bold green]")

    # --- Reproduzierbarkeit: alle Seeds auf denselben Wert setzen ---
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Datensatz laden und tokenisieren ---
    max_length = cfg.get("max_length", 256)
    console.print(f"[cyan]Lade und tokenisiere {dataset_name}...[/cyan]")
    tokenized_dataset, tokenizer, info = prepare_encoder_dataset(
        model_name=cfg["model_name"],
        dataset_name=dataset_name,
        dataset_language=dataset_language,
        max_length=max_length,
    )

    # --- Modell laden ---
    console.print(f"[cyan]Lade Modell: {cfg['model_name']}[/cyan]")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"],
        num_labels=info.num_labels,
        id2label=info.id2label,
        label2id=info.label2id,
        ignore_mismatched_sizes=True,
    )

    # --- Ausgabeverzeichnisse anlegen ---
    # Bei Dataset-Override den Ausgabepfad anpassen
    output_dir = Path(cfg.get("output_dir", f"results/{dataset_name}/{cfg['experiment_name']}"))
    if dataset_override:
        # Output-Pfad auf den Override-Datensatz umschreiben
        output_dir = Path(f"results/{dataset_override}/{cfg['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    # --- TrainingArguments konfigurieren ---
    # Hardware-Erkennung: bf16 bevorzugen wenn verfuegbar, sonst fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", cfg.get("epochs", 5)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", cfg.get("batch_size", 16)),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", cfg.get("eval_batch_size", 32)),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        # Nach jeder Epoche evaluieren und Checkpoint speichern
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        # Am Ende wird das Modell mit dem besten Validation-F1 geladen
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        greater_is_better=cfg.get("greater_is_better", True),
        # Mixed Precision
        bf16=use_bf16,
        fp16=use_fp16,
        seed=seed,
        logging_steps=cfg.get("logging_steps", 50),
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=f"{cfg.get('experiment_name')}_{dataset_name}",
    )

    # --- DataCollator: dynamisches Padding pro Batch ---
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # --- compute_metrics mit dem richtigen id2label bauen ---
    compute_metrics = build_compute_metrics(info)

    # --- Early Stopping Patience aus Config ---
    early_stopping_patience = cfg.get("early_stopping_patience", 1)

    # --- Trainer zusammenbauen ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
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

    # --- Ergebnisse als YAML speichern (fuer compare_all.py) ---
    results: Dict[str, Any] = {
        "experiment_name":       cfg["experiment_name"],
        "model_name":            cfg["model_name"],
        "model_type":            "encoder",
        "dataset":               dataset_name,
        "test_f1":               float(test_f1),
        "test_precision":        float(test_precision),
        "test_recall":           float(test_recall),
        "train_runtime_seconds": float(train_runtime),
        "best_model_dir":        str(best_model_dir),
        "seed":                  seed,
        "num_train_epochs":      training_args.num_train_epochs,
        "learning_rate":         training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "max_length":            max_length,
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
    parser = argparse.ArgumentParser(description="Encoder-NER Training")
    parser.add_argument("config", help="Pfad zur YAML-Config")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Datensatz-Override (z.B. 'wnut_17'). Ueberschreibt den Wert aus der Config.",
    )
    args = parser.parse_args()
    train_encoder(args.config, dataset_override=args.dataset)
