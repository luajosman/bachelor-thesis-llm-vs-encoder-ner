"""
Orchestration script: run all NER experiments sequentially.

Usage:
    python scripts/run_all.py                  # Run everything
    python scripts/run_all.py --encoder-only   # Only encoder models
    python scripts/run_all.py --decoder-only   # Only decoder models
    python scripts/run_all.py --eval-only      # Only evaluation/comparison
    python scripts/run_all.py --model deberta_large  # One specific model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()

ENCODER_CONFIGS = [
    "configs/deberta_large.yaml",
    "configs/deberta_base.yaml",
    "configs/bert_base.yaml",
]

DECODER_CONFIGS = [
    "configs/qwen35_27b.yaml",
    "configs/qwen3_14b.yaml",
]

DECODER_ADAPTERS = {
    "configs/qwen35_27b.yaml": ("results/qwen35-27b-lora/lora_adapter", "Qwen/Qwen3.5-27B"),
    "configs/qwen3_14b.yaml": ("results/qwen3-14b-qlora/lora_adapter", "Qwen/Qwen3-14B"),
}

ENCODER_MODELS = {
    "configs/deberta_large.yaml": "results/deberta-v3-large/best_model",
    "configs/deberta_base.yaml": "results/deberta-v3-base/best_model",
    "configs/bert_base.yaml": "results/bert-base-cased/best_model",
}

MODEL_NAME_MAP = {
    "deberta_large": "configs/deberta_large.yaml",
    "deberta_base": "configs/deberta_base.yaml",
    "bert_base": "configs/bert_base.yaml",
    "qwen35_27b": "configs/qwen35_27b.yaml",
    "qwen3_14b": "configs/qwen3_14b.yaml",
}


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _run_step(label: str, fn, *args, **kwargs):
    """Run a step with timing and exception handling."""
    console.print(Panel(f"[bold]{label}[/bold]", style="cyan"))
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        console.print(f"[green]✓ {label} finished in {elapsed:.1f}s[/green]\n")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        console.print(f"[red]✗ {label} FAILED after {elapsed:.1f}s: {exc}[/red]\n")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def stage_data():
    """Load WNUT-2017 and show statistics."""
    from src.data.load_wnut17 import load_wnut17, print_stats, show_examples
    dataset = load_wnut17()
    print_stats(dataset)
    show_examples(dataset, n=3)


def stage_train_encoder(config_path: str):
    """Train a single encoder model."""
    from src.encoder.train import train_encoder
    train_encoder(config_path)


def stage_infer_encoder(model_path: str, config_path: str):
    """Run encoder inference."""
    from src.encoder.inference import run_encoder_inference
    run_encoder_inference(model_path=model_path, config_path=config_path)


def stage_train_decoder(config_path: str):
    """Train a single decoder model."""
    from src.decoder.train import train_decoder
    train_decoder(config_path)


def stage_infer_decoder(adapter_path: str, base_model: str, config_path: str):
    """Run decoder inference."""
    from src.decoder.inference import run_decoder_inference
    run_decoder_inference(
        adapter_path=adapter_path,
        base_model_name=base_model,
        config_path=config_path,
    )


def stage_compare(results_dir: str = "results"):
    """Run comparison and generate plots."""
    from src.evaluate.compare_all import (
        create_comparison_plot,
        create_per_entity_heatmap,
        export_latex_table,
        load_all_results,
        print_comparison_table,
    )
    results = load_all_results(results_dir)
    if not results:
        console.print("[yellow]No results found. Run training + inference first.[/yellow]")
        return
    print_comparison_table(results)
    create_comparison_plot(results, output_path=f"{results_dir}/comparison_f1.pdf")
    create_per_entity_heatmap(results_dir=results_dir)
    export_latex_table(results, output_path=f"{results_dir}/comparison_table.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run all NER experiments (encoder + decoder + evaluation)"
    )
    parser.add_argument("--encoder-only", action="store_true", help="Only run encoder experiments")
    parser.add_argument("--decoder-only", action="store_true", help="Only run decoder experiments")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation / comparison")
    parser.add_argument("--model", choices=list(MODEL_NAME_MAP.keys()), help="Run one specific model")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only run inference")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference, only train")
    args = parser.parse_args()

    total_start = time.perf_counter()

    console.print(Panel(
        "[bold green]BA-NER Experiment Pipeline[/bold green]\n"
        "WNUT-2017 | Encoder vs Decoder NER Comparison",
        style="green",
    ))

    # ---- Data ----
    if not args.eval_only:
        _run_step("Load & Inspect WNUT-2017", stage_data)

    # ---- Single model shortcut ----
    if args.model:
        config_path = MODEL_NAME_MAP[args.model]
        is_encoder = config_path in ENCODER_CONFIGS

        if not args.skip_train:
            if is_encoder:
                _run_step(f"Train {args.model}", stage_train_encoder, config_path)
            else:
                _run_step(f"Train {args.model}", stage_train_decoder, config_path)

        if not args.skip_inference:
            if is_encoder:
                model_path = ENCODER_MODELS[config_path]
                _run_step(f"Inference {args.model}", stage_infer_encoder, model_path, config_path)
            else:
                adapter, base = DECODER_ADAPTERS[config_path]
                _run_step(f"Inference {args.model}", stage_infer_decoder, adapter, base, config_path)

        _run_step("Comparison & Plots", stage_compare, args.results_dir)
        return

    run_enc = not args.decoder_only and not args.eval_only
    run_dec = not args.encoder_only and not args.eval_only

    # ---- Encoder training ----
    if run_enc and not args.skip_train:
        for cfg in ENCODER_CONFIGS:
            _run_step(f"Train encoder: {cfg}", stage_train_encoder, cfg)

    # ---- Decoder training ----
    if run_dec and not args.skip_train:
        for cfg in DECODER_CONFIGS:
            _run_step(f"Train decoder: {cfg}", stage_train_decoder, cfg)

    # ---- Encoder inference ----
    if run_enc and not args.skip_inference:
        for cfg, model_path in ENCODER_MODELS.items():
            _run_step(f"Inference encoder: {cfg}", stage_infer_encoder, model_path, cfg)

    # ---- Decoder inference ----
    if run_dec and not args.skip_inference:
        for cfg, (adapter, base) in DECODER_ADAPTERS.items():
            _run_step(f"Inference decoder: {cfg}", stage_infer_decoder, adapter, base, cfg)

    # ---- Comparison ----
    _run_step("Comparison & Plots", stage_compare, args.results_dir)

    total_elapsed = time.perf_counter() - total_start
    console.print(Panel(
        f"[bold green]Pipeline complete in {total_elapsed / 60:.1f} minutes.[/bold green]",
        style="green",
    ))


if __name__ == "__main__":
    main()
