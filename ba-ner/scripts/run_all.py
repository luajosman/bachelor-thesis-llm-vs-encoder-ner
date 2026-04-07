"""
run_all.py — Orchestrierung aller NER-Experimente

Fuehrt alle Pipelinestufen sequenziell aus:
  1. Datenladen und Inspektion
  2. Training der Encoder-Modelle (DeBERTa-v3-base, DeBERTa-v3-large)
  3. Training der Decoder-Modelle (Qwen3.5-4B, Qwen3.5-27B mit QLoRA)
  4. Inferenz auf dem Testset
  5. Vergleich und Visualisierung

Verwendung:
    python scripts/run_all.py                    # Hauptbenchmark (MultiNERD)
    python scripts/run_all.py --encoder-only     # Nur Encoder-Modelle
    python scripts/run_all.py --decoder-only     # Nur Decoder-Modelle
    python scripts/run_all.py --eval-only        # Nur Auswertung/Vergleich
    python scripts/run_all.py --model deberta_base  # Ein einzelnes Modell
    python scripts/run_all.py --wnut17           # Zusatzbenchmark (WNUT-2017)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

# Projektwurzel zum Python-Pfad hinzufuegen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()

# ---------------------------------------------------------------------------
# Finale Modell-Konfigurationen
# ---------------------------------------------------------------------------

# Encoder-Konfigurationen
ENCODER_CONFIGS = [
    "configs/deberta_base.yaml",
    "configs/deberta_large.yaml",
]

# Decoder-Konfigurationen
DECODER_CONFIGS = [
    "configs/qwen35_4b.yaml",
    "configs/qwen35_27b.yaml",
]

# Mapping: CLI-Kurzname → Config-Pfad
MODEL_NAME_MAP = {
    "deberta_base":  "configs/deberta_base.yaml",
    "deberta_large": "configs/deberta_large.yaml",
    "qwen35_4b":     "configs/qwen35_4b.yaml",
    "qwen35_27b":    "configs/qwen35_27b.yaml",
}


def _get_encoder_model_path(config_path: str, dataset: str) -> str:
    """Leitet den Pfad zum gespeicherten Encoder-Modell aus Config und Datensatz ab."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return f"results/{dataset}/{cfg['experiment_name']}/best_model"


def _get_decoder_adapter_path(config_path: str, dataset: str) -> str:
    """Leitet den Pfad zum besten LoRA-Adapter aus Config und Datensatz ab."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return f"results/{dataset}/{cfg['experiment_name']}/best_lora_adapter"


def _get_base_model(config_path: str) -> str:
    """Liest den Basismodell-Namen aus der Config."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model_name"]


# ---------------------------------------------------------------------------
# Schritt-Hilfsfunktion
# ---------------------------------------------------------------------------

def _run_step(label: str, fn, *args, **kwargs):
    """Fuehrt einen Pipeline-Schritt mit Zeitmessung und Fehlerbehandlung aus."""
    console.print(Panel(f"[bold]{label}[/bold]", style="cyan"))
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        console.print(f"[green]OK {label} abgeschlossen in {elapsed:.1f}s[/green]\n")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        console.print(f"[red]FEHLER {label} nach {elapsed:.1f}s: {exc}[/red]\n")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pipeline-Stufen
# ---------------------------------------------------------------------------

def stage_data(dataset: str):
    """Stufe 1: Datensatz laden und Statistiken ausgeben."""
    from src.data.dataset_loader import load_ner_dataset
    ds, info = load_ner_dataset(dataset)
    console.print(f"[green]Datensatz {dataset}: {info.num_labels} Labels, "
                  f"{len(info.entity_types)} Entity-Typen[/green]")
    console.print(f"  Entity-Typen: {', '.join(info.entity_types)}")


def stage_train_encoder(config_path: str, dataset: str | None = None):
    """Stufe 2a: Ein Encoder-Modell trainieren."""
    from src.encoder.train import train_encoder
    train_encoder(config_path, dataset_override=dataset)


def stage_infer_encoder(model_path: str, config_path: str, dataset: str | None = None):
    """Stufe 3a: Encoder-Inferenz auf dem Testset."""
    from src.encoder.inference import run_encoder_inference
    run_encoder_inference(model_path=model_path, config_path=config_path, dataset_override=dataset)


def stage_train_decoder(config_path: str, dataset: str | None = None):
    """Stufe 2b: Ein Decoder-Modell mit LoRA/QLoRA trainieren."""
    from src.decoder.train import train_decoder
    train_decoder(config_path, dataset_override=dataset)


def stage_infer_decoder(adapter_path: str, base_model: str, config_path: str, dataset: str | None = None):
    """Stufe 3b: Decoder-Inferenz auf dem Testset."""
    from src.decoder.inference import run_decoder_inference
    run_decoder_inference(
        adapter_path=adapter_path,
        base_model_name=base_model,
        config_path=config_path,
        dataset_override=dataset,
    )


def stage_compare(results_dir: str = "results"):
    """Stufe 4: Vergleich aller Experimente."""
    from src.evaluate.compare_all import (
        create_comparison_plot,
        create_per_entity_heatmap,
        export_latex_table,
        load_all_results,
        print_comparison_table,
    )
    results = load_all_results(results_dir)
    if not results:
        console.print("[yellow]Keine Ergebnisse gefunden.[/yellow]")
        return
    print_comparison_table(results)
    create_comparison_plot(results, output_path=f"{results_dir}/comparison_f1.pdf")
    create_per_entity_heatmap(results_dir=results_dir)
    export_latex_table(results, output_path=f"{results_dir}/comparison_table.tex")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main():
    """Einstiegspunkt: Argument-Parsing und Pipeline-Steuerung."""
    parser = argparse.ArgumentParser(
        description="Alle NER-Experimente ausfuehren (Encoder + Decoder + Auswertung)"
    )
    parser.add_argument("--encoder-only",   action="store_true", help="Nur Encoder-Experimente")
    parser.add_argument("--decoder-only",   action="store_true", help="Nur Decoder-Experimente")
    parser.add_argument("--eval-only",      action="store_true", help="Nur Auswertung/Vergleich")
    parser.add_argument("--model",          choices=list(MODEL_NAME_MAP.keys()), help="Ein einzelnes Modell")
    parser.add_argument("--results-dir",    default="results", help="Ergebnisverzeichnis")
    parser.add_argument("--skip-train",     action="store_true", help="Training ueberspringen")
    parser.add_argument("--skip-inference", action="store_true", help="Inferenz ueberspringen")
    parser.add_argument(
        "--wnut17",
        action="store_true",
        help="Zusatzbenchmark auf WNUT-2017 ausfuehren (statt MultiNERD)",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()

    # Datensatz bestimmen
    dataset = "wnut_17" if args.wnut17 else None  # None = Config-Default (multinerd)
    dataset_label = "WNUT-2017" if args.wnut17 else "MultiNERD (Englisch)"

    console.print(Panel(
        f"[bold green]BA-NER Experiment Pipeline[/bold green]\n"
        f"Datensatz: {dataset_label} | Encoder vs. Decoder NER-Vergleich",
        style="green",
    ))

    # --- Datensatz inspizieren ---
    if not args.eval_only:
        _run_step(
            f"Datensatz laden: {dataset_label}",
            stage_data,
            dataset or "multinerd",
        )

    # --- Einzelnes Modell (--model Shortcut) ---
    if args.model:
        config_path = MODEL_NAME_MAP[args.model]
        is_encoder = config_path in ENCODER_CONFIGS
        ds = dataset or "multinerd"

        if not args.skip_train:
            if is_encoder:
                _run_step(f"Training {args.model}", stage_train_encoder, config_path, dataset)
            else:
                _run_step(f"Training {args.model}", stage_train_decoder, config_path, dataset)

        if not args.skip_inference:
            if is_encoder:
                model_path = _get_encoder_model_path(config_path, ds)
                _run_step(f"Inferenz {args.model}", stage_infer_encoder, model_path, config_path, dataset)
            else:
                adapter = _get_decoder_adapter_path(config_path, ds)
                base = _get_base_model(config_path)
                _run_step(f"Inferenz {args.model}", stage_infer_decoder, adapter, base, config_path, dataset)

        _run_step("Vergleich & Plots", stage_compare, args.results_dir)
        return

    # --- Welche Gruppen ausfuehren? ---
    run_enc = not args.decoder_only and not args.eval_only
    run_dec = not args.encoder_only and not args.eval_only

    # Fuer WNUT-17 Zusatzbenchmark: nur deberta_base + qwen35_4b
    enc_configs = ENCODER_CONFIGS
    dec_configs = DECODER_CONFIGS
    if args.wnut17:
        enc_configs = ["configs/deberta_base.yaml"]
        dec_configs = ["configs/qwen35_4b.yaml"]
        console.print("[cyan]WNUT-17 Zusatzbenchmark: deberta-v3-base + qwen35-4b[/cyan]")

    ds = dataset or "multinerd"

    # --- Encoder-Training ---
    if run_enc and not args.skip_train:
        for cfg in enc_configs:
            _run_step(f"Training Encoder: {cfg}", stage_train_encoder, cfg, dataset)

    # --- Decoder-Training ---
    if run_dec and not args.skip_train:
        for cfg in dec_configs:
            _run_step(f"Training Decoder: {cfg}", stage_train_decoder, cfg, dataset)

    # --- Encoder-Inferenz ---
    if run_enc and not args.skip_inference:
        for cfg in enc_configs:
            model_path = _get_encoder_model_path(cfg, ds)
            _run_step(f"Inferenz Encoder: {cfg}", stage_infer_encoder, model_path, cfg, dataset)

    # --- Decoder-Inferenz ---
    if run_dec and not args.skip_inference:
        for cfg in dec_configs:
            adapter = _get_decoder_adapter_path(cfg, ds)
            base = _get_base_model(cfg)
            _run_step(f"Inferenz Decoder: {cfg}", stage_infer_decoder, adapter, base, cfg, dataset)

    # --- Vergleich ---
    _run_step("Vergleich & Plots", stage_compare, args.results_dir)

    total_elapsed = time.perf_counter() - total_start
    console.print(Panel(
        f"[bold green]Pipeline abgeschlossen in {total_elapsed / 60:.1f} Minuten.[/bold green]",
        style="green",
    ))


if __name__ == "__main__":
    main()
