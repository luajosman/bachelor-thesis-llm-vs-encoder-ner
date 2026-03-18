"""
compare_all.py — Aggregation und Visualisierung aller Experiment-Ergebnisse

Liest die YAML-Ergebnisdateien aller Experimente ein und erstellt:
  - Eine Rich-Tabelle für den Terminal-Vergleich
  - Ein horizontales Balkendiagramm (F1-Scores)
  - Eine Heatmap der F1-Scores pro Entitätstyp
  - Eine LaTeX-Tabelle für die Bachelorarbeit

Verwendung:
    python -m src.evaluate.compare_all
    python -m src.evaluate.compare_all --results-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Nicht-interaktives Backend für Cluster-Betrieb (kein Display nötig)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# Farben für Encoder (grün) und Decoder (lila) — konsistent in allen Plots
ENCODER_COLOR = "#2ecc71"
DECODER_COLOR = "#9b59b6"

# Die sechs Entitätstypen des WNUT-2017-Datensatzes
ENTITY_TYPES = [
    "corporation", "creative-work", "group", "location", "person", "product"
]


# ---------------------------------------------------------------------------
# Ergebnisse laden
# ---------------------------------------------------------------------------


def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Lädt alle Experiment-Ergebnisse aus den YAML-Dateien.

    Sucht in jedem Unterordner von results_dir nach:
      - results.yaml         (Training-Metriken: F1, Precision, Recall, Trainingszeit)
      - inference_metrics.yaml (Inferenz-Metriken: Latenz, VRAM)

    Beide Dateien werden zusammengeführt, sodass ein vollständiges
    Ergebnisdict pro Experiment entsteht.

    Args:
        results_dir: Wurzelverzeichnis mit einem Unterordner pro Experiment.

    Returns:
        Liste von Ergebnis-Dicts, eines pro Experiment.
    """
    results: List[Dict[str, Any]] = []
    root = Path(results_dir)

    # Alle Unterordner alphabetisch durchgehen
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Training-Ergebnisse (werden zuerst geladen)
        train_file = exp_dir / "results.yaml"
        # Inferenz-Metriken (überschreiben bei Namenskonflikten)
        inf_file = exp_dir / "inference_metrics.yaml"

        data: Dict[str, Any] = {}

        if train_file.exists():
            with open(train_file) as f:
                data.update(yaml.safe_load(f) or {})

        if inf_file.exists():
            with open(inf_file) as f:
                # Inferenz-Metriken haben Vorrang (aktuellere Daten)
                data.update(yaml.safe_load(f) or {})

        if data:
            # Experiment-Name aus Ordnername ableiten falls nicht vorhanden
            data.setdefault("experiment_name", exp_dir.name)
            results.append(data)

    return results


# ---------------------------------------------------------------------------
# Rich-Vergleichstabelle
# ---------------------------------------------------------------------------


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Gibt eine formatierte Rich-Tabelle mit allen Modellen aus.

    Sortiert absteigend nach Test-F1, damit das beste Modell oben steht.
    Encoder werden grün, Decoder magenta markiert.

    Args:
        results: Liste von Ergebnis-Dicts aus load_all_results().
    """
    table = Table(
        title="NER Model Comparison (WNUT-2017 Test Set)",
        show_header=True,
        header_style="bold cyan",
    )
    # Spalten definieren
    table.add_column("Rank",        justify="right",  style="dim")
    table.add_column("Model",       style="bold")
    table.add_column("Type",        justify="center")
    table.add_column("Params",      justify="right")
    table.add_column("Test F1",     justify="right",  style="bold green")
    table.add_column("Precision",   justify="right")
    table.add_column("Recall",      justify="right")
    table.add_column("Train (min)", justify="right")
    table.add_column("VRAM (GB)",   justify="right")
    table.add_column("Latency (ms)",justify="right")

    # Absteigende Sortierung nach F1
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0), reverse=True)

    for rank, r in enumerate(sorted_results, start=1):
        # Modell-Typ farblich hervorheben
        model_type = r.get("model_type", "?")
        type_label = (
            "[green]Encoder[/green]"
            if model_type == "encoder"
            else "[magenta]Decoder[/magenta]"
        )

        # Parameter-Anzahl menschenlesbar formatieren (z.B. "435M", "27.0B")
        total_params = r.get("total_params", 0)
        params_str = _format_params(total_params)

        # Trainingszeit: Sekunden → Minuten
        train_secs = r.get("train_runtime_seconds", 0.0)
        train_min = f"{train_secs / 60:.1f}" if train_secs else "-"

        # VRAM: MB → GB
        vram_mb = r.get("vram_peak_mb", 0.0)
        vram_gb = f"{vram_mb / 1024:.1f}" if vram_mb else "-"

        # Inferenz-Latenz in ms
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
    """Formatiert eine Parameteranzahl als lesbare Zeichenkette.

    Beispiele: 0 → "-", 435_000_000 → "435M", 27_000_000_000 → "27.0B".
    """
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

    Encoder werden grün, Decoder lila dargestellt. Die Balken sind
    aufsteigend nach F1 sortiert (schlechtestes Modell unten).

    Args:
        results:     Liste von Ergebnis-Dicts.
        output_path: Speicherpfad für die PDF/PNG-Ausgabe.
    """
    # Aufsteigend sortieren: schlechtestes Modell unten im Diagramm
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0))

    names  = [r.get("experiment_name", r.get("model_name", "?")) for r in sorted_results]
    f1s    = [r.get("test_f1", 0.0) for r in sorted_results]
    colors = [
        ENCODER_COLOR if r.get("model_type") == "encoder" else DECODER_COLOR
        for r in sorted_results
    ]

    # Diagrammgröße dynamisch an Anzahl der Modelle anpassen
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.7)))
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", height=0.6)

    # F1-Wert als Text rechts neben jedem Balken einblenden
    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # X-Achse: 0 bis max(F1)+Puffer, maximal 1.0
    ax.set_xlim(0, min(1.0, max(f1s) + 0.06))
    ax.set_xlabel("Entity-Level F1 Score (seqeval)", fontsize=11)
    ax.set_title("NER Model Comparison — WNUT-2017 Test Set", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Legende: Encoder vs. Decoder
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
# Heatmap: F1 pro Entitätstyp
# ---------------------------------------------------------------------------


def create_per_entity_heatmap(
    results_dir: str = "results",
    output_path: str = "results/per_entity_heatmap.pdf",
) -> None:
    """Erstellt eine Heatmap der F1-Scores je Entitätstyp und Modell.

    Liest dazu test_predictions.json aus jedem Experiment-Unterordner
    und berechnet die per-Typ-Metriken mit compute_per_entity_metrics().

    Farbe: YlGn (gelb-grün), 0.0 = gelb, 1.0 = dunkelgrün.
    Zellwerte werden direkt in die Heatmap geschrieben (weiß bei >0.6, sonst schwarz).

    Args:
        results_dir: Wurzelverzeichnis der Experimente.
        output_path: Speicherpfad für die Heatmap.
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

        # Gold- und Vorhersage-Tags aus den Samples extrahieren
        # Encoder speichert "gold"/"pred", Decoder speichert "gold_bio"/"pred_bio"
        gold_tags = [
            s["gold"] if "gold" in s else s.get("gold_bio", [])
            for s in samples
        ]
        pred_tags = [
            s["pred"] if "pred" in s else s.get("pred_bio", [])
            for s in samples
        ]

        # Per-Typ-F1 berechnen; fehlende Typen erhalten 0.0
        per_entity = compute_per_entity_metrics(gold_tags, pred_tags)
        per_entity_data[name] = {
            etype: per_entity.get(etype, {}).get("f1", 0.0)
            for etype in ENTITY_TYPES
        }

    if not model_names:
        console.print("[yellow]Keine test_predictions.json gefunden — Heatmap wird übersprungen.[/yellow]")
        return

    # Matrix aufbauen: Zeilen = Modelle, Spalten = Entitätstypen
    matrix = np.array(
        [[per_entity_data[m][et] for et in ENTITY_TYPES] for m in model_names]
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(model_names) * 0.8)))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Achsenbeschriftungen
    ax.set_xticks(range(len(ENTITY_TYPES)))
    ax.set_xticklabels(ENTITY_TYPES, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("F1 per Entity Type — WNUT-2017 Test Set", fontsize=13, fontweight="bold")

    # F1-Wert in jede Zelle schreiben
    for i in range(len(model_names)):
        for j in range(len(ENTITY_TYPES)):
            val = matrix[i, j]
            # Helle Zellen (hoher F1): weißer Text; dunkle Zellen: schwarzer Text
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Heatmap gespeichert: {out}[/green]")


# ---------------------------------------------------------------------------
# LaTeX-Tabelle für die Bachelorarbeit
# ---------------------------------------------------------------------------


def export_latex_table(
    results: List[Dict[str, Any]],
    output_path: str = "results/comparison_table.tex",
) -> None:
    """Exportiert eine LaTeX-Tabelle mit allen Modell-Ergebnissen.

    Die Tabelle ist nach F1 absteigend sortiert und enthält:
    Modellname, Typ, Parameter, F1, Precision, Recall.

    Sonderzeichen (Unterstriche, Bindestriche) werden für LaTeX escaped.

    Args:
        results:     Liste von Ergebnis-Dicts.
        output_path: Pfad für die .tex-Ausgabedatei.
    """
    sorted_results = sorted(results, key=lambda x: x.get("test_f1", 0.0), reverse=True)

    # LaTeX-Tabellengerüst aufbauen
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
        name   = r.get("experiment_name", r.get("model_name", "?"))
        mtype  = "Encoder" if r.get("model_type") == "encoder" else "Decoder"
        params = _format_params(r.get("total_params", 0))
        f1     = f"{r.get('test_f1', 0.0):.4f}"
        prec   = f"{r.get('test_precision', 0.0):.4f}"
        rec    = f"{r.get('test_recall', 0.0):.4f}"

        # LaTeX-Sonderzeichen escapen: _ → \_ und - → \text{-}
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
    console.print(f"[green]LaTeX-Tabelle gespeichert: {out}[/green]")


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vergleich aller NER-Experiment-Ergebnisse")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Wurzelverzeichnis mit Experiment-Unterordnern (Standard: results/)",
    )
    args = parser.parse_args()

    results = load_all_results(args.results_dir)

    if not results:
        console.print(f"[yellow]Keine Ergebnisse in '{args.results_dir}' gefunden. Zuerst Training ausführen.[/yellow]")
    else:
        console.print(f"[green]{len(results)} Experiment(e) gefunden.[/green]")
        print_comparison_table(results)
        create_comparison_plot(results, output_path=f"{args.results_dir}/comparison_f1.pdf")
        create_per_entity_heatmap(
            results_dir=args.results_dir,
            output_path=f"{args.results_dir}/per_entity_heatmap.pdf",
        )
        export_latex_table(results, output_path=f"{args.results_dir}/comparison_table.tex")
