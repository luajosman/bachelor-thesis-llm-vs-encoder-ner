"""
run_all.py — Orchestrierung aller NER-Experimente

Führt alle Pipelinestufen sequenziell aus:
  1. Datenladen und Inspektion (WNUT-2017)
  2. Training der Encoder-Modelle (DeBERTa, BERT)
  3. Training der Decoder-Modelle (Qwen3.5-27B, Qwen3-14B mit LoRA/QLoRA)
  4. Inferenz auf dem Testset
  5. Vergleich und Visualisierung

Verwendung:
    python scripts/run_all.py                  # Alle Experimente
    python scripts/run_all.py --encoder-only   # Nur Encoder-Modelle
    python scripts/run_all.py --decoder-only   # Nur Decoder-Modelle
    python scripts/run_all.py --eval-only      # Nur Auswertung/Vergleich
    python scripts/run_all.py --model deberta_large  # Ein einzelnes Modell
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

# Projektwurzel zum Python-Pfad hinzufügen, damit `src.*`-Importe funktionieren
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()

# Pfade zu allen Encoder-Konfigurationsdateien
ENCODER_CONFIGS = [
    "configs/deberta_large.yaml",
    "configs/deberta_base.yaml",
    "configs/bert_base.yaml",
]

# Pfade zu allen Decoder-Konfigurationsdateien
DECODER_CONFIGS = [
    "configs/qwen35_27b.yaml",
    "configs/qwen3_14b.yaml",
]

# Mapping: Decoder-Config → (LoRA-Adapter-Pfad, Basismodell-Name auf HuggingFace)
DECODER_ADAPTERS = {
    "configs/qwen35_27b.yaml": ("results/qwen35-27b-lora/lora_adapter", "Qwen/Qwen3.5-27B"),
    "configs/qwen3_14b.yaml":  ("results/qwen3-14b-qlora/lora_adapter", "Qwen/Qwen3-14B"),
}

# Mapping: Encoder-Config → Pfad zum gespeicherten besten Modell
ENCODER_MODELS = {
    "configs/deberta_large.yaml": "results/deberta-v3-large/best_model",
    "configs/deberta_base.yaml":  "results/deberta-v3-base/best_model",
    "configs/bert_base.yaml":     "results/bert-base-cased/best_model",
}

# Mapping: CLI-Kurzname → Config-Pfad (für --model Argument)
MODEL_NAME_MAP = {
    "deberta_large": "configs/deberta_large.yaml",
    "deberta_base":  "configs/deberta_base.yaml",
    "bert_base":     "configs/bert_base.yaml",
    "qwen35_27b":    "configs/qwen35_27b.yaml",
    "qwen3_14b":     "configs/qwen3_14b.yaml",
}


# ---------------------------------------------------------------------------
# Schritt-Hilfsfunktion
# ---------------------------------------------------------------------------


def _run_step(label: str, fn, *args, **kwargs):
    """Führt einen Pipeline-Schritt mit Zeitmessung und Fehlerbehandlung aus.

    Gibt bei Erfolg ein grünes Erfolgspanel aus, bei Fehler ein rotes Fehlerpanel.
    Exceptions werden abgefangen und mit Traceback ausgegeben, ohne die gesamte
    Pipeline zu unterbrechen.

    Args:
        label: Bezeichnung des Schritts (wird im Panel angezeigt).
        fn:    Aufzurufende Funktion.
        *args, **kwargs: Weitergabe der Argumente an fn.

    Returns:
        Rückgabewert von fn, oder None bei Fehler.
    """
    console.print(Panel(f"[bold]{label}[/bold]", style="cyan"))
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        console.print(f"[green]✓ {label} abgeschlossen in {elapsed:.1f}s[/green]\n")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        console.print(f"[red]✗ {label} FEHLGESCHLAGEN nach {elapsed:.1f}s: {exc}[/red]\n")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pipeline-Stufen
# ---------------------------------------------------------------------------


def stage_data():
    """Stufe 1: WNUT-2017 laden und Statistiken ausgeben.

    Zeigt Anzahl der Samples pro Split und Entitätsverteilung an.
    Nützlich zur Verifikation, dass der Datensatz korrekt geladen wird.
    """
    from src.data.load_wnut17 import load_wnut17, print_stats, show_examples
    dataset = load_wnut17()
    print_stats(dataset)
    show_examples(dataset, n=3)


def stage_train_encoder(config_path: str):
    """Stufe 2a: Ein Encoder-Modell trainieren.

    Liest die YAML-Config ein und ruft train_encoder() auf.
    Das beste Modell (höchster Validierungs-F1) wird gespeichert.

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei (z.B. configs/deberta_large.yaml).
    """
    from src.encoder.train import train_encoder
    train_encoder(config_path)


def stage_infer_encoder(model_path: str, config_path: str):
    """Stufe 3a: Encoder-Inferenz auf dem Testset ausführen.

    Lädt das gespeicherte Modell und berechnet F1 + Latenz auf dem Testset.
    Speichert Ergebnisse in inference_metrics.yaml und test_predictions.json.

    Args:
        model_path:  Pfad zum gespeicherten Modellordner (best_model/).
        config_path: Pfad zur zugehörigen YAML-Config.
    """
    from src.encoder.inference import run_encoder_inference
    run_encoder_inference(model_path=model_path, config_path=config_path)


def stage_train_decoder(config_path: str):
    """Stufe 2b: Ein Decoder-Modell mit LoRA/QLoRA trainieren.

    Liest die YAML-Config ein und ruft train_decoder() auf.
    Speichert nur den LoRA-Adapter (nicht das gesamte Basismodell).

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei (z.B. configs/qwen3_14b.yaml).
    """
    from src.decoder.train import train_decoder
    train_decoder(config_path)


def stage_infer_decoder(adapter_path: str, base_model: str, config_path: str):
    """Stufe 3b: Decoder-Inferenz auf dem Testset ausführen.

    Lädt Basismodell + LoRA-Adapter und generiert NER-Ausgaben im JSON-Format.
    Speichert Ergebnisse in inference_metrics.yaml und test_predictions.json.

    Args:
        adapter_path: Pfad zum gespeicherten LoRA-Adapter.
        base_model:   HuggingFace-Modellname (z.B. "Qwen/Qwen3-14B").
        config_path:  Pfad zur zugehörigen YAML-Config.
    """
    from src.decoder.inference import run_decoder_inference
    run_decoder_inference(
        adapter_path=adapter_path,
        base_model_name=base_model,
        config_path=config_path,
    )


def stage_compare(results_dir: str = "results"):
    """Stufe 4: Vergleich aller Experimente und Visualisierungen erstellen.

    Liest alle YAML-Ergebnisdateien ein und erstellt:
      - Rich-Vergleichstabelle im Terminal
      - Balkendiagramm (F1-Scores)
      - Heatmap (F1 pro Entitätstyp)
      - LaTeX-Tabelle für die Bachelorarbeit

    Args:
        results_dir: Wurzelverzeichnis mit den Experiment-Unterordnern.
    """
    from src.evaluate.compare_all import (
        create_comparison_plot,
        create_per_entity_heatmap,
        export_latex_table,
        load_all_results,
        print_comparison_table,
    )
    results = load_all_results(results_dir)
    if not results:
        console.print("[yellow]Keine Ergebnisse gefunden. Zuerst Training + Inferenz ausführen.[/yellow]")
        return
    print_comparison_table(results)
    create_comparison_plot(results, output_path=f"{results_dir}/comparison_f1.pdf")
    create_per_entity_heatmap(results_dir=results_dir)
    export_latex_table(results, output_path=f"{results_dir}/comparison_table.tex")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def main():
    """Einstiegspunkt: Argument-Parsing und Pipeline-Steuerung.

    Unterstützte Modi:
      - Standard:        Alle Experimente (Encoder + Decoder + Vergleich)
      - --encoder-only:  Nur Encoder-Modelle
      - --decoder-only:  Nur Decoder-Modelle
      - --eval-only:     Nur Auswertung (setzt fertige Ergebnisse voraus)
      - --model NAME:    Nur ein einzelnes Modell
      - --skip-train:    Training überspringen, direkt Inferenz
      - --skip-inference:Inferenz überspringen, nur Training
    """
    parser = argparse.ArgumentParser(
        description="Alle NER-Experimente ausführen (Encoder + Decoder + Auswertung)"
    )
    parser.add_argument("--encoder-only",    action="store_true", help="Nur Encoder-Experimente")
    parser.add_argument("--decoder-only",    action="store_true", help="Nur Decoder-Experimente")
    parser.add_argument("--eval-only",       action="store_true", help="Nur Auswertung/Vergleich")
    parser.add_argument("--model",           choices=list(MODEL_NAME_MAP.keys()), help="Ein einzelnes Modell ausführen")
    parser.add_argument("--results-dir",     default="results", help="Ergebnisverzeichnis")
    parser.add_argument("--skip-train",      action="store_true", help="Training überspringen, nur Inferenz")
    parser.add_argument("--skip-inference",  action="store_true", help="Inferenz überspringen, nur Training")
    args = parser.parse_args()

    total_start = time.perf_counter()

    # Begrüßungspanel mit Projektübersicht
    console.print(Panel(
        "[bold green]BA-NER Experiment Pipeline[/bold green]\n"
        "WNUT-2017 | Encoder vs. Decoder NER-Vergleich",
        style="green",
    ))

    # Datensatz laden und inspizieren (außer bei --eval-only)
    if not args.eval_only:
        _run_step("WNUT-2017 laden & inspizieren", stage_data)

    # ---- Einzelnes Modell (Shortcut mit --model) ----
    if args.model:
        config_path = MODEL_NAME_MAP[args.model]
        is_encoder  = config_path in ENCODER_CONFIGS

        if not args.skip_train:
            if is_encoder:
                _run_step(f"Training {args.model}", stage_train_encoder, config_path)
            else:
                _run_step(f"Training {args.model}", stage_train_decoder, config_path)

        if not args.skip_inference:
            if is_encoder:
                model_path = ENCODER_MODELS[config_path]
                _run_step(f"Inferenz {args.model}", stage_infer_encoder, model_path, config_path)
            else:
                adapter, base = DECODER_ADAPTERS[config_path]
                _run_step(f"Inferenz {args.model}", stage_infer_decoder, adapter, base, config_path)

        _run_step("Vergleich & Plots", stage_compare, args.results_dir)
        return

    # Welche Gruppen sollen ausgeführt werden?
    run_enc = not args.decoder_only and not args.eval_only
    run_dec = not args.encoder_only and not args.eval_only

    # ---- Encoder-Training ----
    if run_enc and not args.skip_train:
        for cfg in ENCODER_CONFIGS:
            _run_step(f"Training Encoder: {cfg}", stage_train_encoder, cfg)

    # ---- Decoder-Training ----
    if run_dec and not args.skip_train:
        for cfg in DECODER_CONFIGS:
            _run_step(f"Training Decoder: {cfg}", stage_train_decoder, cfg)

    # ---- Encoder-Inferenz ----
    if run_enc and not args.skip_inference:
        for cfg, model_path in ENCODER_MODELS.items():
            _run_step(f"Inferenz Encoder: {cfg}", stage_infer_encoder, model_path, cfg)

    # ---- Decoder-Inferenz ----
    if run_dec and not args.skip_inference:
        for cfg, (adapter, base) in DECODER_ADAPTERS.items():
            _run_step(f"Inferenz Decoder: {cfg}", stage_infer_decoder, adapter, base, cfg)

    # ---- Abschließender Vergleich ----
    _run_step("Vergleich & Plots", stage_compare, args.results_dir)

    total_elapsed = time.perf_counter() - total_start
    console.print(Panel(
        f"[bold green]Pipeline abgeschlossen in {total_elapsed / 60:.1f} Minuten.[/bold green]",
        style="green",
    ))


if __name__ == "__main__":
    main()
