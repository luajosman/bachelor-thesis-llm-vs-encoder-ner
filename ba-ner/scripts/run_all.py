"""
run_all.py — Orchestrierung aller NER-Experimente

Fuehrt die finale Experiment-Matrix sequenziell aus:

  Encoder (DeBERTa, fine-tuned):
    - deberta-v3-base
    - deberta-v3-large

  LLM Zero-Shot (Qwen3.5, kein Training):
    - qwen35-08b-zeroshot
    - qwen35-4b-zeroshot
    - qwen35-27b-zeroshot

  LLM LoRA/QLoRA (Qwen3.5, fine-tuned):
    - qwen35-08b-qlora
    - qwen35-4b-qlora
    - qwen35-27b-qlora

Datensatz: ausschliesslich MultiNERD English.

Verwendung:
    python scripts/run_all.py                       # gesamte Matrix
    python scripts/run_all.py --encoder-only        # nur Encoder
    python scripts/run_all.py --decoder-only        # nur LLMs (Zero-Shot + LoRA)
    python scripts/run_all.py --zeroshot-only       # nur LLM Zero-Shot
    python scripts/run_all.py --finetuned-only      # nur DeBERTa + LLM LoRA
    python scripts/run_all.py --eval-only           # nur Auswertung/Vergleich
    python scripts/run_all.py --model deberta_base  # genau ein Modell
    python scripts/run_all.py --skip-train          # Training ueberspringen
    python scripts/run_all.py --skip-inference      # Inferenz ueberspringen
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
# Finale Modell-Konfigurationen (MultiNERD English only)
# ---------------------------------------------------------------------------

# Encoder-Konfigurationen (fine-tuned, Token-Klassifikation)
ENCODER_CONFIGS = [
    "configs/deberta_base.yaml",
    "configs/deberta_large.yaml",
]

# LLM Zero-Shot Konfigurationen (kein Training)
DECODER_ZEROSHOT_CONFIGS = [
    "configs/qwen35_08b_zeroshot.yaml",
    "configs/qwen35_4b_zeroshot.yaml",
    "configs/qwen35_27b_zeroshot.yaml",
]

# LLM LoRA/QLoRA Konfigurationen (fine-tuned)
DECODER_LORA_CONFIGS = [
    "configs/qwen35_08b.yaml",
    "configs/qwen35_4b.yaml",
    "configs/qwen35_27b.yaml",
]

# Mapping: CLI-Kurzname -> Config-Pfad
MODEL_NAME_MAP = {
    # Encoder
    "deberta_base":      "configs/deberta_base.yaml",
    "deberta_large":     "configs/deberta_large.yaml",
    # LLM LoRA
    "qwen35_08b":        "configs/qwen35_08b.yaml",
    "qwen35_4b":         "configs/qwen35_4b.yaml",
    "qwen35_27b":        "configs/qwen35_27b.yaml",
    # LLM Zero-Shot
    "qwen35_08b_zs":     "configs/qwen35_08b_zeroshot.yaml",
    "qwen35_4b_zs":      "configs/qwen35_4b_zeroshot.yaml",
    "qwen35_27b_zs":     "configs/qwen35_27b_zeroshot.yaml",
}


# ---------------------------------------------------------------------------
# Config-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_encoder_model_path(config_path: str) -> str:
    """Pfad zum gespeicherten Encoder-Modell aus der Config ableiten."""
    cfg = _load_cfg(config_path)
    return f"results/multinerd/{cfg['experiment_name']}/best_model"


def _get_decoder_adapter_path(config_path: str) -> str:
    """Pfad zum besten LoRA-Adapter aus der Config ableiten."""
    cfg = _load_cfg(config_path)
    return f"results/multinerd/{cfg['experiment_name']}/best_lora_adapter"


def _get_base_model(config_path: str) -> str:
    """Liest den Basismodell-Namen aus der Config."""
    cfg = _load_cfg(config_path)
    return cfg["model_name"]


def _is_zeroshot(config_path: str) -> bool:
    """Erkennt anhand des mode-Felds, ob es sich um Zero-Shot handelt."""
    cfg = _load_cfg(config_path)
    return str(cfg.get("mode", "lora")).lower() == "zeroshot"


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

def stage_data():
    """Stufe 1: MultiNERD English laden und Statistiken ausgeben."""
    from src.data.dataset_loader import load_ner_dataset
    ds, info = load_ner_dataset("multinerd")
    console.print(
        f"[green]Datensatz multinerd: {info.num_labels} Labels, "
        f"{len(info.entity_types)} Entity-Typen[/green]"
    )
    console.print(f"  Entity-Typen: {', '.join(info.entity_types)}")


def stage_train_encoder(config_path: str):
    """Stufe 2a: Ein Encoder-Modell trainieren."""
    from src.encoder.train import train_encoder
    train_encoder(config_path)


def stage_infer_encoder(model_path: str, config_path: str):
    """Stufe 3a: Encoder-Inferenz auf dem Testset."""
    from src.encoder.inference import run_encoder_inference
    run_encoder_inference(model_path=model_path, config_path=config_path)


def stage_train_decoder(config_path: str):
    """Stufe 2b: Ein Decoder-Modell mit LoRA/QLoRA trainieren."""
    from src.decoder.train import train_decoder
    train_decoder(config_path)


def stage_infer_decoder_lora(adapter_path: str, base_model: str, config_path: str):
    """Stufe 3b: Decoder-Inferenz mit trainiertem LoRA-Adapter."""
    from src.decoder.inference import run_decoder_inference
    run_decoder_inference(
        adapter_path=adapter_path,
        base_model_name=base_model,
        config_path=config_path,
        zeroshot=False,
    )


def stage_infer_decoder_zeroshot(base_model: str, config_path: str):
    """Stufe 3c: Decoder-Inferenz im Zero-Shot Mode (kein Adapter)."""
    from src.decoder.inference import run_decoder_inference
    run_decoder_inference(
        adapter_path=None,
        base_model_name=base_model,
        config_path=config_path,
        zeroshot=True,
    )


def stage_compare(results_dir: str = "results"):
    """Stufe 4: Vergleich aller MultiNERD-Experimente."""
    from src.evaluate.compare_all import (
        create_comparison_plot,
        create_per_entity_heatmap,
        export_latex_table,
        load_all_results,
        print_comparison_table,
    )
    results = load_all_results(results_dir, dataset_filter="multinerd")
    if not results:
        console.print("[yellow]Keine Ergebnisse gefunden.[/yellow]")
        return
    print_comparison_table(results)
    create_comparison_plot(results, output_path=f"{results_dir}/comparison_f1.pdf")
    create_per_entity_heatmap(results_dir=results_dir, dataset_filter="multinerd")
    export_latex_table(results, output_path=f"{results_dir}/comparison_table.tex")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main():
    """Einstiegspunkt: Argument-Parsing und Pipeline-Steuerung."""
    parser = argparse.ArgumentParser(
        description="Alle NER-Experimente ausfuehren (Encoder + LLM Zero-Shot + LLM LoRA)"
    )
    parser.add_argument("--encoder-only",   action="store_true", help="Nur Encoder-Experimente")
    parser.add_argument("--decoder-only",   action="store_true", help="Nur LLM-Experimente (Zero-Shot + LoRA)")
    parser.add_argument("--zeroshot-only",  action="store_true", help="Nur LLM Zero-Shot")
    parser.add_argument("--finetuned-only", action="store_true", help="Nur fine-tuned (DeBERTa + LLM LoRA)")
    parser.add_argument("--eval-only",      action="store_true", help="Nur Auswertung/Vergleich")
    parser.add_argument("--model",          choices=list(MODEL_NAME_MAP.keys()), help="Ein einzelnes Modell")
    parser.add_argument("--results-dir",    default="results", help="Ergebnisverzeichnis")
    parser.add_argument("--skip-train",     action="store_true", help="Training ueberspringen")
    parser.add_argument("--skip-inference", action="store_true", help="Inferenz ueberspringen")
    args = parser.parse_args()

    total_start = time.perf_counter()

    console.print(Panel(
        "[bold green]BA-NER Experiment Pipeline[/bold green]\n"
        "Datensatz: MultiNERD (Englisch) | Encoder vs. LLM (Zero-Shot + LoRA)",
        style="green",
    ))

    # --- Datensatz inspizieren ---
    if not args.eval_only:
        _run_step("Datensatz laden: MultiNERD (Englisch)", stage_data)

    # --- Einzelnes Modell (--model Shortcut) ---
    if args.model:
        config_path = MODEL_NAME_MAP[args.model]
        is_encoder = config_path in ENCODER_CONFIGS
        is_zs = _is_zeroshot(config_path)

        if not args.skip_train:
            if is_encoder:
                _run_step(f"Training {args.model}", stage_train_encoder, config_path)
            elif not is_zs:
                _run_step(f"Training {args.model}", stage_train_decoder, config_path)
            else:
                console.print(f"[dim]Zero-Shot {args.model}: kein Training noetig.[/dim]")

        if not args.skip_inference:
            if is_encoder:
                model_path = _get_encoder_model_path(config_path)
                _run_step(f"Inferenz {args.model}", stage_infer_encoder, model_path, config_path)
            elif is_zs:
                base = _get_base_model(config_path)
                _run_step(f"Inferenz {args.model}", stage_infer_decoder_zeroshot, base, config_path)
            else:
                adapter = _get_decoder_adapter_path(config_path)
                base = _get_base_model(config_path)
                _run_step(
                    f"Inferenz {args.model}",
                    stage_infer_decoder_lora,
                    adapter, base, config_path,
                )

        _run_step("Vergleich & Plots", stage_compare, args.results_dir)
        return

    # --- Welche Gruppen ausfuehren? ---
    # zeroshot-only und finetuned-only sind exklusive Filter ueber alle Gruppen
    run_enc       = not args.decoder_only and not args.eval_only and not args.zeroshot_only
    run_dec_lora  = not args.encoder_only and not args.eval_only and not args.zeroshot_only
    run_dec_zs    = not args.encoder_only and not args.eval_only and not args.finetuned_only

    # --- Encoder-Training ---
    if run_enc and not args.skip_train:
        for cfg in ENCODER_CONFIGS:
            _run_step(f"Training Encoder: {cfg}", stage_train_encoder, cfg)

    # --- LLM LoRA Training ---
    if run_dec_lora and not args.skip_train:
        for cfg in DECODER_LORA_CONFIGS:
            _run_step(f"Training LLM LoRA: {cfg}", stage_train_decoder, cfg)

    # --- Encoder-Inferenz ---
    if run_enc and not args.skip_inference:
        for cfg in ENCODER_CONFIGS:
            model_path = _get_encoder_model_path(cfg)
            _run_step(f"Inferenz Encoder: {cfg}", stage_infer_encoder, model_path, cfg)

    # --- LLM LoRA Inferenz ---
    if run_dec_lora and not args.skip_inference:
        for cfg in DECODER_LORA_CONFIGS:
            adapter = _get_decoder_adapter_path(cfg)
            base = _get_base_model(cfg)
            _run_step(
                f"Inferenz LLM LoRA: {cfg}",
                stage_infer_decoder_lora,
                adapter, base, cfg,
            )

    # --- LLM Zero-Shot Inferenz (kein Training) ---
    if run_dec_zs and not args.skip_inference:
        for cfg in DECODER_ZEROSHOT_CONFIGS:
            base = _get_base_model(cfg)
            _run_step(
                f"Inferenz LLM Zero-Shot: {cfg}",
                stage_infer_decoder_zeroshot,
                base, cfg,
            )

    # --- Vergleich ---
    _run_step("Vergleich & Plots", stage_compare, args.results_dir)

    total_elapsed = time.perf_counter() - total_start
    console.print(Panel(
        f"[bold green]Pipeline abgeschlossen in {total_elapsed / 60:.1f} Minuten.[/bold green]",
        style="green",
    ))


if __name__ == "__main__":
    main()
