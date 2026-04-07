"""
compare_all.py — Aggregation und Visualisierung aller Experiment-Ergebnisse

Liest die YAML-Ergebnisdateien aller Experimente ein und erstellt:
  - Eine Rich-Tabelle fuer den Terminal-Vergleich
  - Ein horizontales Balkendiagramm (F1-Scores)
  - Eine Heatmap der F1-Scores pro Entitaetstyp
  - Eine LaTeX-Tabelle fuer die Bachelorarbeit

Erkennt automatisch die Verzeichnisstruktur:
  - Neu: results/<dataset>/<experiment>/...  (Multi-Dataset)
  - Alt: results/<experiment>/...            (Legacy)

Verwendung:
    python -m src.evaluate.compare_all
    python -m src.evaluate.compare_all --results-dir results/
    python -m src.evaluate.compare_all --dataset multinerd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from src.data.dataset_loader import get_dataset_info

console = Console()

# Farben fuer Encoder (gruen) und Decoder (lila)
ENCODER_COLOR = "#2ecc71"
DECODER_COLOR = "#9b59b6"

# Bekannte Datensaetze
KNOWN_DATASETS = {"multinerd", "wnut_17"}


# ---------------------------------------------------------------------------
# Ergebnisse laden
# ---------------------------------------------------------------------------

def _load_experiment_dir(exp_dir: Path) -> Dict[str, Any] | None:
    """Laedt results.yaml + inference_metrics.yaml aus einem Experiment-Ordner.

    Returns:
        Zusammengefuehrtes Dict oder None, wenn keine Ergebnisse vorhanden.
    """
    train_file = exp_dir / "results.yaml"
    inf_file = exp_dir / "inference_metrics.yaml"

    data: Dict[str, Any] = {}

    if train_file.exists():
        with open(train_file) as f:
            data.update(yaml.safe_load(f) or {})
    if inf_file.exists():
        with open(inf_file) as f:
            # Inferenz-Metriken haben Vorrang
            data.update(yaml.safe_load(f) or {})

    if not data:
        return None

    data.setdefault("experiment_name", exp_dir.name)
    return data


def load_all_results(
    results_dir: str = "results",
    dataset_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """Laedt alle Experiment-Ergebnisse rekursiv aus dem Ergebnisverzeichnis.

    Erkennt automatisch zwei Strukturen:
      - results/<dataset>/<experiment>/results.yaml  (Multi-Dataset)
      - results/<experiment>/results.yaml            (Legacy)

    Args:
        results_dir:    Wurzelverzeichnis.
        dataset_filter: Nur Ergebnisse dieses Datensatzes laden.

    Returns:
        Liste von Ergebnis-Dicts.
    """
    results: List[Dict[str, Any]] = []
    root = Path(results_dir)
    if not root.exists():
        return results

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        # Heuristik: ist das ein Datensatz-Ordner?
        if entry.name in KNOWN_DATASETS:
            dataset_name = entry.name
            if dataset_filter and dataset_name != dataset_filter:
                continue
            for exp_dir in sorted(entry.iterdir()):
                if not exp_dir.is_dir():
                    continue
                data = _load_experiment_dir(exp_dir)
                if data:
                    data.setdefault("dataset", dataset_name)
                    results.append(data)
        else:
            # Legacy-Struktur: direkt unter results/
            data = _load_experiment_dir(entry)
            if data:
                if dataset_filter and data.get("dataset") != dataset_filter:
                    continue
                results.append(data)

    return results


# ---------------------------------------------------------------------------
# Rich-Vergleichstabelle
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Gibt eine formatierte Rich-Tabelle mit allen Modellen aus.

    Sortiert absteigend nach Test-F1.
    """
    table = Table(
        title="NER Model Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank",        justify="right",  style="dim")
    table.add_column("Dataset",     justify="left")
    table.add_column("Model",       style="bold")
    table.add_column("Type",        justify="center")
    table.add_column("Params",      justify="right")
    table.add_column("Test F1",     justify="right",  style="bold green")
    table.add_column("Precision",   justify="right")
    table.add_column("Recall",      justify="right")
    table.add_column("Train (min)", justify="right")
    table.add_column("VRAM (GB)",   justify="right")
    table.add_column("Latency (ms)",justify="right")

    sorted_results = sorted(
        results,
        key=lambda x: (x.get("dataset", ""), -x.get("test_f1", 0.0)),
    )

    for rank, r in enumerate(sorted_results, start=1):
        model_type = r.get("model_type", "?")
        type_label = (
            "[green]Encoder[/green]"
            if model_type == "encoder"
            else "[magenta]Decoder[/magenta]"
        )

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
            r.get("dataset", "?"),
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
    """Formatiert eine Parameteranzahl als lesbare Zeichenkette."""
    if n == 0:
        return "-"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


# ---------------------------------------------------------------------------
# Balkendiagramm F1-Scores
# ---------------------------------------------------------------------------

def create_comparison_plot(
    results: List[Dict[str, Any]],
    output_path: str = "results/comparison_f1.pdf",
) -> None:
    """Erstellt ein horizontales Balkendiagramm der F1-Scores.

    Modellnamen enthalten den Datensatz als Praefix, damit beide
    Benchmarks im selben Plot unterscheidbar sind.
    """
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0))

    names = [
        f"[{r.get('dataset', '?')}] {r.get('experiment_name', r.get('model_name', '?'))}"
        for r in sorted_results
    ]
    f1s = [r.get("test_f1", 0.0) for r in sorted_results]
    colors = [
        ENCODER_COLOR if r.get("model_type") == "encoder" else DECODER_COLOR
        for r in sorted_results
    ]

    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.6)))
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", height=0.6)

    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    if f1s:
        ax.set_xlim(0, min(1.0, max(f1s) + 0.06))
    ax.set_xlabel("Entity-Level F1 Score (seqeval)", fontsize=11)
    ax.set_title("NER Model Comparison", fontsize=13, fontweight="bold")
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
    console.print(f"[green]Vergleichsplot gespeichert: {out}[/green]")


# ---------------------------------------------------------------------------
# Heatmap: F1 pro Entitaetstyp (pro Datensatz)
# ---------------------------------------------------------------------------

def create_per_entity_heatmap(
    results_dir: str = "results",
    output_path: str = "results/per_entity_heatmap.pdf",
    dataset_filter: str | None = None,
) -> None:
    """Erstellt eine Heatmap der F1-Scores je Entitaetstyp und Modell.

    Erstellt eine separate Heatmap pro Datensatz, da die Entity-Typen
    sich zwischen MultiNERD (15 Typen) und WNUT-17 (6 Typen) unterscheiden.
    """
    from src.evaluate.metrics import compute_per_entity_metrics

    root = Path(results_dir)
    if not root.exists():
        return

    # Per-Datensatz gruppierte Datenstruktur:
    # {dataset_name: {model_name: {entity_type: f1}}}
    per_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _collect_from_exp(exp_dir: Path, dataset_name: str):
        pred_file = exp_dir / "test_predictions.json"
        if not pred_file.exists():
            return
        with open(pred_file, encoding="utf-8") as f:
            samples = json.load(f)

        # Encoder speichert "gold"/"pred", Decoder "gold_bio"/"pred_bio"
        gold_tags = [
            s["gold"] if "gold" in s else s.get("gold_bio", [])
            for s in samples
        ]
        pred_tags = [
            s["pred"] if "pred" in s else s.get("pred_bio", [])
            for s in samples
        ]

        per_entity = compute_per_entity_metrics(gold_tags, pred_tags)

        try:
            info = get_dataset_info(dataset_name)
            entity_types = info.entity_types
        except ValueError:
            # Fallback: entity types aus per_entity ableiten
            entity_types = sorted(per_entity.keys())

        per_dataset.setdefault(dataset_name, {})[exp_dir.name] = {
            etype: per_entity.get(etype, {}).get("f1", 0.0)
            for etype in entity_types
        }

    # Iteriere ueber Multi-Dataset- und Legacy-Strukturen
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in KNOWN_DATASETS:
            if dataset_filter and entry.name != dataset_filter:
                continue
            for exp_dir in sorted(entry.iterdir()):
                if exp_dir.is_dir():
                    _collect_from_exp(exp_dir, entry.name)
        else:
            # Legacy-Struktur: Datensatz aus results.yaml ableiten
            data = _load_experiment_dir(entry)
            if data:
                ds = data.get("dataset", "unknown")
                if dataset_filter and ds != dataset_filter:
                    continue
                _collect_from_exp(entry, ds)

    if not per_dataset:
        console.print("[yellow]Keine test_predictions.json gefunden.[/yellow]")
        return

    # Eine Heatmap pro Datensatz
    out_base = Path(output_path)
    for dataset_name, models_dict in per_dataset.items():
        if not models_dict:
            continue

        try:
            info = get_dataset_info(dataset_name)
            entity_types = info.entity_types
        except ValueError:
            entity_types = sorted({et for m in models_dict.values() for et in m.keys()})

        model_names = list(models_dict.keys())
        matrix = np.array(
            [[models_dict[m].get(et, 0.0) for et in entity_types] for m in model_names]
        )

        fig, ax = plt.subplots(figsize=(max(8, len(entity_types) * 0.7),
                                         max(3, len(model_names) * 0.8)))
        im = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(entity_types)))
        ax.set_xticklabels(entity_types, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_title(f"F1 per Entity Type — {dataset_name}", fontsize=13, fontweight="bold")

        for i in range(len(model_names)):
            for j in range(len(entity_types)):
                val = matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label="F1 Score")
        plt.tight_layout()

        # Pfad pro Datensatz: heatmap.pdf -> heatmap_multinerd.pdf
        ds_out = out_base.parent / f"{out_base.stem}_{dataset_name}{out_base.suffix}"
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(ds_out), dpi=150, bbox_inches="tight")
        plt.close()
        console.print(f"[green]Heatmap gespeichert: {ds_out}[/green]")


# ---------------------------------------------------------------------------
# LaTeX-Tabelle fuer die Bachelorarbeit
# ---------------------------------------------------------------------------

def export_latex_table(
    results: List[Dict[str, Any]],
    output_path: str = "results/comparison_table.tex",
) -> None:
    """Exportiert eine LaTeX-Tabelle mit allen Modell-Ergebnissen.

    Sortiert nach Datensatz, dann nach F1 absteigend.
    """
    sorted_results = sorted(
        results,
        key=lambda x: (x.get("dataset", ""), -x.get("test_f1", 0.0)),
    )

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{NER Model Comparison (MultiNERD \& WNUT-2017)}",
        r"  \label{tab:ner-comparison}",
        r"  \begin{tabular}{lllrrrr}",
        r"    \toprule",
        r"    \textbf{Dataset} & \textbf{Model} & \textbf{Type} & \textbf{Params} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} \\",
        r"    \midrule",
    ]

    for r in sorted_results:
        dataset = r.get("dataset", "?")
        name   = r.get("experiment_name", r.get("model_name", "?"))
        mtype  = "Encoder" if r.get("model_type") == "encoder" else "Decoder"
        params = _format_params(r.get("total_params", 0))
        f1     = f"{r.get('test_f1', 0.0):.4f}"
        prec   = f"{r.get('test_precision', 0.0):.4f}"
        rec    = f"{r.get('test_recall', 0.0):.4f}"

        # LaTeX-Sonderzeichen escapen
        def _esc(s: str) -> str:
            return s.replace("_", r"\_").replace("-", r"\text{-}")

        lines.append(
            f"    {_esc(dataset)} & {_esc(name)} & {mtype} & {params} & {f1} & {prec} & {rec} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]LaTeX-Tabelle gespeichert: {out}[/green]")


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vergleich aller NER-Experiment-Ergebnisse")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Wurzelverzeichnis (Standard: results/)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional: Nur Ergebnisse fuer einen bestimmten Datensatz",
    )
    args = parser.parse_args()

    results = load_all_results(args.results_dir, dataset_filter=args.dataset)

    if not results:
        console.print(f"[yellow]Keine Ergebnisse in '{args.results_dir}' gefunden.[/yellow]")
    else:
        console.print(f"[green]{len(results)} Experiment(e) gefunden.[/green]")
        print_comparison_table(results)
        create_comparison_plot(results, output_path=f"{args.results_dir}/comparison_f1.pdf")
        create_per_entity_heatmap(
            results_dir=args.results_dir,
            output_path=f"{args.results_dir}/per_entity_heatmap.pdf",
            dataset_filter=args.dataset,
        )
        export_latex_table(results, output_path=f"{args.results_dir}/comparison_table.tex")
