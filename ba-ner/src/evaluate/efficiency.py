"""
efficiency.py — Effizienz- und Ressourcenmessung für Modell-Experimente

Misst die für die Bachelorarbeit relevanten Effizienzmetriken:
  - Parameteranzahl (total und trainierbar)
  - VRAM-Peak (GPU-Speicher in MB)
  - Inferenz-Latenz pro Sample (in Millisekunden)

Alle Latenz-Messungen verwenden cuda.synchronize() um sicherzustellen,
dass asynchrone GPU-Operationen vollständig abgeschlossen sind, bevor
die Zeit gestoppt wird.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Generator, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Datenklasse für Effizienz-Metriken
# ---------------------------------------------------------------------------

@dataclass
class EfficiencyMetrics:
    """Strukturierter Container für alle Effizienz-Messungen eines Modells.

    Wird von compare_all.py für den Vergleich zwischen Encoder- und
    Decoder-Modellen genutzt.

    Attributes:
        model_name:            Experiment-Name (aus YAML-Config).
        total_params:          Gesamtanzahl aller Parameter.
        trainable_params:      Anzahl der trainierbaren Parameter (bei LoRA: nur Adapter).
        train_time_seconds:    Trainingszeit in Sekunden (Wanduhrzeit).
        vram_peak_mb:          Maximaler GPU-Speicherverbrauch in MB.
        inference_latency_ms:  Mittlere Inferenz-Latenz pro Sample in ms.
        tokens_per_second:     Durchsatz in Tokens/Sekunde (optional).
    """
    model_name:           str   = ""
    total_params:         int   = 0
    trainable_params:     int   = 0
    train_time_seconds:   float = 0.0
    vram_peak_mb:         float = 0.0
    inference_latency_ms: float = 0.0
    tokens_per_second:    float = 0.0

    def to_dict(self):
        """Konvertiert die Dataclass in ein normales Dict (für YAML-Export)."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"EfficiencyMetrics({self.model_name}): "
            f"params={self.total_params:,}, "
            f"train={self.train_time_seconds:.1f}s, "
            f"vram={self.vram_peak_mb:.1f}MB, "
            f"latency={self.inference_latency_ms:.2f}ms"
        )


# ---------------------------------------------------------------------------
# Parameter zählen
# ---------------------------------------------------------------------------

def count_parameters(model) -> Tuple[int, int]:
    """Zählt die Gesamt- und trainierbaren Parameter eines PyTorch-Modells.

    Bei PEFT/LoRA-Modellen ist trainable_params << total_params, da die
    Basisgewichte eingefroren sind und nur der Adapter trainiert wird.

    Args:
        model: Ein torch.nn.Module (auch PEFT-wrapped).

    Returns:
        Tuple (total_params, trainable_params).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# VRAM-Tracking
# ---------------------------------------------------------------------------

def get_vram_peak_mb() -> float:
    """Gibt den maximalen GPU-Speicherverbrauch seit dem letzten Reset zurück.

    Nutzt torch.cuda.max_memory_allocated() — misst allokierten Speicher,
    nicht den vom Betriebssystem reservierten (der ist typischerweise höher).

    Returns:
        VRAM-Peak in MB, oder 0.0 wenn keine GPU verfügbar.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram_tracking() -> None:
    """Setzt den VRAM-Peak-Zähler zurück.

    Muss vor einem Mess-Block aufgerufen werden, damit nur der VRAM
    des gewünschten Blocks gemessen wird (nicht Vorheriges).
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_vram() -> Generator[dict, None, None]:
    """Context-Manager, der den VRAM-Verbrauch innerhalb eines Blocks misst.

    Yields:
        Dict, das nach dem Block die Felder 'vram_peak_mb' und
        'vram_before_mb' enthält.

    Beispiel:
        >>> with track_vram() as info:
        ...     model(inputs)
        >>> print(f"VRAM: {info['vram_peak_mb']:.1f} MB")
    """
    info: dict = {"vram_peak_mb": 0.0, "vram_before_mb": 0.0}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # VRAM vor dem Block festhalten (z.B. bereits geladenes Modell)
        info["vram_before_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)

    try:
        yield info
    finally:
        # VRAM-Peak nach dem Block abfragen
        if torch.cuda.is_available():
            info["vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Latenz-Messung
# ---------------------------------------------------------------------------

def measure_inference_latency(
    fn,
    n_runs:   int = 20,
    n_warmup: int = 3,
) -> Tuple[float, float]:
    """Misst die Inferenz-Latenz einer Funktion mit CUDA-Synchronisierung.

    Warmup-Läufe werden nicht in die Messung einbezogen, da der erste
    CUDA-Aufruf langsamer ist (JIT-Kompilierung, Cache-Aufbau).

    Args:
        fn:       Aufrufbare Funktion ohne Argumente (ein Inferenz-Schritt).
        n_runs:   Anzahl gemessener Läufe.
        n_warmup: Anzahl Warmup-Läufe vor der Messung.

    Returns:
        Tuple (mean_ms, std_ms) — Mittelwert und Standardabweichung in ms.

    Beispiel:
        >>> mean_ms, std_ms = measure_inference_latency(lambda: model(inputs))
    """
    # Warmup-Läufe durchführen (werden nicht gemessen)
    for _ in range(n_warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies: List[float] = []
    for _ in range(n_runs):
        # Vor der Messung: sicherstellen, dass GPU-Warteschlange leer ist
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        fn()

        # Nach der Ausführung: warten bis GPU fertig ist
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # in Millisekunden

    import numpy as np
    mean_ms = float(np.mean(latencies))
    std_ms  = float(np.std(latencies))
    return mean_ms, std_ms
