"""Encoder training for MultiNERD English token classification."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict

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

from src.config import DATASET_LANGUAGE, DATASET_NAME, load_experiment_config, output_dir_from_config
from src.data.preprocess_encoder import prepare_encoder_dataset
from src.evaluate.metrics import build_token_classification_compute_metrics
from src.run_metadata import collect_run_metadata

console = Console()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_encoder(config_path: str) -> Dict[str, Any]:
    """Trainiert ein Encoder-Modell fuer NER auf Basis einer YAML-Config.

    Ablauf:
      1. Config laden und Seeds setzen (Reproduzierbarkeit)
      2. Datensatz tokenisieren und Labels ausrichten
      3. Modell mit Token-Klassifikationskopf laden
      4. HuggingFace Trainer mit Early Stopping konfigurieren
      5. Training starten und bestes Modell speichern
      6. Trainingszusammenfassung als YAML speichern

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Dict mit Trainingslaufzeit, bestem Validation-F1 und Artefaktpfaden.
    """
    config_path = Path(config_path)
    cfg = load_experiment_config(config_path, expected_model_type="encoder", expected_regime="encoder")

    console.rule(f"[bold green]Training: {cfg['experiment_name']} on MultiNERD English[/bold green]")

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
    console.print("[cyan]Loading and tokenizing MultiNERD English...[/cyan]")
    tokenized_dataset, tokenizer, info = prepare_encoder_dataset(
        model_name=cfg["model_name"],
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

    output_dir = output_dir_from_config(cfg)
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
        run_name=f"{cfg.get('experiment_name')}_{DATASET_NAME}",
    )

    # --- DataCollator: dynamisches Padding pro Batch ---
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    compute_metrics = build_token_classification_compute_metrics(info.id2label)

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
    trainer.train()
    train_runtime = time.perf_counter() - train_start

    # --- Bestes Modell speichern ---
    console.print(f"[green]Speichere bestes Modell nach {best_model_dir}[/green]")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    best_validation_f1 = float(trainer.state.best_metric or 0.0)
    console.print(f"\n[bold green]Best validation F1: {best_validation_f1:.4f}[/bold green]")
    console.print(f"Trainingszeit: {train_runtime:.1f}s")

    # --- Ergebnisse als YAML speichern ---
    results: Dict[str, Any] = {
        "experiment_name":       cfg["experiment_name"],
        "model_name":            cfg["model_name"],
        "model_type":            "encoder",
        "regime":                "encoder",  # Token-Klassifikation
        "dataset":               DATASET_NAME,
        "dataset_language":      DATASET_LANGUAGE,
        "best_validation_f1":    best_validation_f1,
        "train_runtime_seconds": float(train_runtime),
        "best_model_dir":        str(best_model_dir),
        "seed":                  seed,
        "num_train_epochs":      training_args.num_train_epochs,
        "learning_rate":         training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "max_length":            max_length,
        "run_metadata":          collect_run_metadata(cfg),
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
    args = parser.parse_args()
    train_encoder(args.config)
