"""
Aggregate results from all experiments and generate comparison tables,
plots, and LaTeX output.

Usage:
    python -m src.evaluate.compare_all
    python -m src.evaluate.compare_all --results-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

ENCODER_COLOR = "#2ecc71"   # green
DECODER_COLOR = "#9b59b6"   # purple

ENTITY_TYPES = [
    "corporation", "creative-work", "group", "location", "person", "product"
]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------


def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Load all experiment results from YAML files.

    Searches for ``inference_metrics.yaml`` files under ``results_dir``.
    Falls back to ``results.yaml`` if inference metrics don't exist.

    Parameters
    ----------
    results_dir:
        Root directory containing one subdirectory per experiment.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dicts, one per experiment.
    """
    results: List[Dict[str, Any]] = []
    root = Path(results_dir)

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Prefer inference_metrics.yaml (has latency + VRAM)
        inf_file = exp_dir / "inference_metrics.yaml"
        train_file = exp_dir / "results.yaml"

        data: Dict[str, Any] = {}

        if train_file.exists():
            with open(train_file) as f:
                data.update(yaml.safe_load(f) or {})

        if inf_file.exists():
            with open(inf_file) as f:
                data.update(yaml.safe_load(f) or {})

        if data:
            data.setdefault("experiment_name", exp_dir.name)
            results.append(data)

    return results


# ---------------------------------------------------------------------------
# Rich comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a rich table sorted by F1 descending.

    Parameters
    ----------
    results:
        List of result dicts from ``load_all_results()``.
    """
    table = Table(
        title="NER Model Comparison (WNUT-2017 Test Set)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Model", style="bold")
    table.add_column("Type", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Test F1", justify="right", style="bold green")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Train (min)", justify="right")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("Latency (ms)", justify="right")

    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0), reverse=True)

    for rank, r in enumerate(sorted_results, start=1):
        model_type = r.get("model_type", "?")
        type_label = f"[green]Encoder[/green]" if model_type == "encoder" else f"[magenta]Decoder[/magenta]"

        total_params = r.get("total_params", 0)
        params_str = _format_params(total_params)

        train_secs = r.get("train_runtime_seconds", 0.0)
        train_min = f"{train_secs / 60:.1f}" if train_secs else "-"

        vram_mb = r.get("vram_peak_mb", 0.0)
        vram_gb = f"{vram_mb / 1024:.1f}" if vram_mb else "-"

        latency = r.get("latency_ms_mean", 0.0)
        latency_str = f"{latency:.1f}" if latency else "-"

        table.add_row(
            str(rank),
            r.get("experiment_name", r.get("model_name", "?")),
            type_label,
            params_str,
            f"{r.get('test_f1', 0.0):.4f}",
            f"{r.get('test_precision', 0.0):.4f}",
            f"{r.get('test_recall', 0.0):.4f}",
            train_min,
            vram_gb,
            latency_str,
        )

    console.print(table)


def _format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n == 0:
        return "-"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


# ---------------------------------------------------------------------------
# Comparison bar plot
# ---------------------------------------------------------------------------


def create_comparison_plot(
    results: List[Dict[str, Any]],
    output_path: str = "results/comparison_f1.pdf",
) -> None:
    """Create a horizontal bar plot of F1 scores (Encoder=green, Decoder=purple).

    Parameters
    ----------
    results:
        List of result dicts.
    output_path:
        Path to save the PDF/PNG figure.
    """
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0))

    names = [r.get("experiment_name", r.get("model_name", "?")) for r in sorted_results]
    f1s = [r.get("test_f1", 0.0) for r in sorted_results]
    colors = [
        ENCODER_COLOR if r.get("model_type") == "encoder" else DECODER_COLOR
        for r in sorted_results
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.7)))
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", height=0.6)

    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlim(0, min(1.0, max(f1s) + 0.06))
    ax.set_xlabel("Entity-Level F1 Score (seqeval)", fontsize=11)
    ax.set_title("NER Model Comparison — WNUT-2017 Test Set", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    legend_patches = [
        mpatches.Patch(color=ENCODER_COLOR, label="Encoder (Token Classification)"),
        mpatches.Patch(color=DECODER_COLOR, label="Decoder (Generative + LoRA)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Comparison plot saved to {out}[/green]")


# ---------------------------------------------------------------------------
# Per-entity heatmap
# ---------------------------------------------------------------------------


def create_per_entity_heatmap(
    results_dir: str = "results",
    output_path: str = "results/per_entity_heatmap.pdf",
) -> None:
    """Create a heatmap of F1 per entity type and model.

    Loads ``test_predictions.json`` from each experiment directory.

    Parameters
    ----------
    results_dir:
        Root directory of experiment results.
    output_path:
        Output path for the heatmap PDF.
    """
    from src.evaluate.metrics import compute_per_entity_metrics

    root = Path(results_dir)
    model_names: List[str] = []
    per_entity_data: Dict[str, Dict[str, float]] = {}

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        pred_file = exp_dir / "test_predictions.json"
        if not pred_file.exists():
            continue

        with open(pred_file, encoding="utf-8") as f:
            samples = json.load(f)

        name = exp_dir.name
        model_names.append(name)

        gold_tags = [s["gold"] if "gold" in s else s.get("gold_bio", []) for s in samples]
        pred_tags = [s["pred"] if "pred" in s else s.get("pred_bio", []) for s in samples]

        per_entity = compute_per_entity_metrics(gold_tags, pred_tags)
        per_entity_data[name] = {etype: per_entity.get(etype, {}).get("f1", 0.0) for etype in ENTITY_TYPES}

    if not model_names:
        console.print("[yellow]No test_predictions.json files found — skipping heatmap.[/yellow]")
        return

    # Build matrix
    matrix = np.array(
        [[per_entity_data[m][et] for et in ENTITY_TYPES] for m in model_names]
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(model_names) * 0.8)))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("F1 per Entity Type — WNUT-2017 Test Set", fontsize=13, fontweight="bold")

    for i in range(len(model_names)):
        for j in range(len(ENTITY_TYPES)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Heatmap saved to {out}[/green]")


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------


def export_latex_table(
    results: List[Dict[str, Any]],
    output_path: str = "results/comparison_table.tex",
) -> None:
    """Export a LaTeX table of results for the bachelor thesis.

    Parameters
    ----------
    results:
        List of result dicts.
    output_path:
        Path to write the .tex file.
    """
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0), reverse=True)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{NER Model Comparison on WNUT-2017 Test Set}",
        r"  \label{tab:ner-comparison}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Type} & \textbf{Params} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} \\",
        r"    \midrule",
    ]

    for r in sorted_results:
        name = r.get("experiment_name", r.get("model_name", "?"))
        mtype = "Encoder" if r.get("model_type") == "encoder" else "Decoder"
        params = _format_params(r.get("total_params", 0))
        f1 = f"{r.get('test_f1', 0.0):.4f}"
        prec = f"{r.get('test_precision', 0.0):.4f}"
        rec = f"{r.get('test_recall', 0.0):.4f}"
        # Escape underscores for LaTeX
        name_tex = name.replace("_", r"\_").replace("-", r"\text{-}")
        lines.append(f"    {name_tex} & {mtype} & {params} & {f1} & {prec} & {rec} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]LaTeX table saved to {out}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all NER experiment results")
    parser.add_argument(
        "--results-dir", default="results", help="Root results directory (default: results/)"
    )
    args = parser.parse_args()

    results = load_all_results(args.results_dir)

    if not results:
        console.print(f"[yellow]No results found in '{args.results_dir}'. Run training first.[/yellow]")
    else:
        console.print(f"[green]Found {len(results)} experiment(s).[/green]")
        print_comparison_table(results)
        create_comparison_plot(results, output_path=f"{args.results_dir}/comparison_f1.pdf")
        create_per_entity_heatmap(
            results_dir=args.results_dir,
            output_path=f"{args.results_dir}/per_entity_heatmap.pdf",
        )
        export_latex_table(results, output_path=f"{args.results_dir}/comparison_table.tex")
