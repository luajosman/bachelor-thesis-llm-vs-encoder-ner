"""
inference.py — Encoder-Inferenz auf dem WNUT-2017 Test-Set

Lädt das gespeicherte beste Modell, lässt es über alle Test-Sätze laufen,
misst dabei Inferenz-Latenz (mit CUDA-Synchronisierung) und VRAM-Peak.
Alle Vorhersagen werden als JSON gespeichert (für die Fehleranalyse).

Verwendung:
    python -m src.encoder.inference \\
        --model results/deberta-v3-large/best_model \\
        --config configs/deberta_large.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from rich.console import Console
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data.load_wnut17 import ID2LABEL, load_wnut17
from src.data.preprocess_encoder import prepare_encoder_dataset
from src.evaluate.efficiency import count_parameters, reset_vram_tracking, get_vram_peak_mb

console = Console()


# ---------------------------------------------------------------------------
# Vorhersagen dekodieren
# ---------------------------------------------------------------------------

def _decode_predictions(logits: torch.Tensor, label_ids: torch.Tensor) -> Tuple[List[str], List[str]]:
    """Wandelt Logit-Tensor und Label-Tensor in BIO-String-Listen um.

    Positionen mit label_id == -100 (Subword-Tokens, Sondertokens) werden
    übersprungen, da sie kein echtes Label tragen.

    Args:
        logits:    Ausgabe-Logits des Modells, Shape (seq_len, num_labels).
        label_ids: Gold-Label-IDs, Shape (seq_len,).

    Returns:
        Tuple (true_labels, pred_labels) als Listen von BIO-Strings.
    """
    # Wahrscheinlichste Klasse pro Token wählen (argmax über Labels)
    preds  = logits.argmax(dim=-1).cpu().numpy()
    labels = label_ids.cpu().numpy()

    true_labels: List[str] = []
    true_preds:  List[str] = []

    for p, l in zip(preds, labels):
        # -100 = Subword-Fortsetzung oder Sondertoken → überspringen
        if l == -100:
            continue
        true_labels.append(ID2LABEL[int(l)])
        true_preds.append(ID2LABEL[int(p)])

    return true_labels, true_preds


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------

def run_encoder_inference(
    model_path: str,
    config_path: str,
) -> Dict[str, Any]:
    """Führt Inferenz mit einem trainierten Encoder-Modell auf dem Test-Set durch.

    Für jedes Test-Sample wird einzeln inferiert, um eine realistische
    Latenz-Messung zu erhalten (kein Batching bei der Latenz-Messung).
    cuda.synchronize() stellt sicher, dass die GPU tatsächlich fertig ist,
    bevor die Zeit gestoppt wird.

    Args:
        model_path:  Pfad zum gespeicherten best_model/-Verzeichnis.
        config_path: Pfad zur YAML-Config (für Ausgabepfade und Einstellungen).

    Returns:
        Dict mit f1, precision, recall, latency_ms_mean, vram_peak_mb.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold cyan]Inferenz: {cfg['experiment_name']}[/bold cyan]")

    # GPU bevorzugen, falls verfügbar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Gerät: {device}")

    # --- Modell und Tokenizer laden ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Dropout deaktivieren für deterministische Ausgaben

    total_params, trainable_params = count_parameters(model)
    console.print(f"Parameter: {total_params:,} gesamt, {trainable_params:,} trainierbar")

    # --- Tokenisierten Datensatz laden ---
    tokenized_dataset, _, _ = prepare_encoder_dataset(
        model_name=model_path,
        max_length=cfg.get("max_length", 128),
    )
    test_split = tokenized_dataset["test"]

    # Rohe Token-Strings für die Fehleranalyse-Ausgabe aufbewahren
    raw_dataset = load_wnut17()
    raw_test = raw_dataset["test"]

    # --- Warmup-Lauf (CUDA-Caches aufwärmen) ---
    # Der erste Forward-Pass auf der GPU ist typischerweise langsamer
    # (JIT-Kompilierung, Cache-Aufbau). Der Warmup-Lauf verhindert, dass
    # dieser Overhead die Latenz-Messung verzerrt.
    reset_vram_tracking()
    if device.type == "cuda":
        dummy = {k: torch.tensor([v]).to(device) for k, v in test_split[0].items() if k != "labels"}
        with torch.no_grad():
            _ = model(**dummy)
        torch.cuda.synchronize()

    # --- Inferenz-Schleife ---
    all_true:      List[List[str]] = []
    all_preds:     List[List[str]] = []
    latencies_ms:  List[float]     = []
    saved_samples: List[Dict]      = []

    console.print(f"[cyan]Inferenz auf {len(test_split)} Test-Samples...[/cyan]")

    for i, sample in enumerate(test_split):
        # Eingabe-Tensoren aufbauen
        input_ids      = torch.tensor([sample["input_ids"]]).to(device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
        label_ids      = torch.tensor(sample["labels"])

        # BERT hat token_type_ids, DeBERTa nicht → optional übergeben
        extra_kwargs: Dict = {}
        if "token_type_ids" in sample:
            extra_kwargs["token_type_ids"] = torch.tensor([sample["token_type_ids"]]).to(device)

        # Latenz messen: cuda.synchronize() sicherstellen, dass GPU-Operationen
        # abgeschlossen sind, bevor die Zeit gestoppt wird
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **extra_kwargs,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        # Logits und Labels in BIO-Strings dekodieren
        true_labels, pred_labels = _decode_predictions(outputs.logits[0], label_ids)
        all_true.append(true_labels)
        all_preds.append(pred_labels)

        # Sample für spätere Fehleranalyse speichern
        saved_samples.append({
            "tokens": raw_test[i]["tokens"],
            "gold":   true_labels,
            "pred":   pred_labels,
        })

    # VRAM-Peak nach der gesamten Inferenz ablesen
    vram_peak = get_vram_peak_mb()

    # --- Metriken berechnen ---
    f1        = f1_score(all_true, all_preds, zero_division=0)
    precision = precision_score(all_true, all_preds, zero_division=0)
    recall    = recall_score(all_true, all_preds, zero_division=0)

    latency_mean = float(np.mean(latencies_ms))
    latency_p95  = float(np.percentile(latencies_ms, 95))

    console.print(f"\n[bold green]Test F1: {f1:.4f}[/bold green]")
    console.print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")
    console.print(f"Mittlere Latenz: {latency_mean:.2f} ms  (p95: {latency_p95:.2f} ms)")
    console.print(f"VRAM-Peak: {vram_peak:.1f} MB")

    # --- Vorhersagen speichern (für Fehleranalyse) ---
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))
    pred_file = output_dir / "test_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, ensure_ascii=False, indent=2)
    console.print(f"Vorhersagen gespeichert: {pred_file}")

    # --- Inferenz-Metriken als YAML speichern ---
    metrics: Dict[str, Any] = {
        "experiment_name":   cfg["experiment_name"],
        "model_type":        "encoder",
        "test_f1":           float(f1),
        "test_precision":    float(precision),
        "test_recall":       float(recall),
        "latency_ms_mean":   latency_mean,
        "latency_ms_p95":    latency_p95,
        "vram_peak_mb":      vram_peak,
        "total_params":      total_params,
    }

    inf_file = output_dir / "inference_metrics.yaml"
    with open(inf_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    console.print(f"Inferenz-Metriken gespeichert: {inf_file}")

    return metrics


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoder-NER Inferenz auf WNUT-2017 Test-Set")
    parser.add_argument("--model",  required=True, help="Pfad zum gespeicherten Modell-Verzeichnis")
    parser.add_argument("--config", required=True, help="Pfad zur YAML-Config")
    args = parser.parse_args()

    run_encoder_inference(model_path=args.model, config_path=args.config)
