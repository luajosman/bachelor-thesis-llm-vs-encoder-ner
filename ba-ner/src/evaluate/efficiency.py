"""
Efficiency and resource tracking utilities.

Covers:
- Parameter counting (total + trainable)
- VRAM peak tracking with CUDA
- Inference latency measurement (with CUDA synchronization)
- EfficiencyMetrics dataclass for structured reporting
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Generator, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyMetrics:
    """Structured container for model efficiency measurements.

    Attributes
    ----------
    model_name:
        Short name / experiment name.
    total_params:
        Total number of parameters.
    trainable_params:
        Number of trainable parameters.
    train_time_seconds:
        Wall-clock training time in seconds.
    vram_peak_mb:
        Peak GPU memory usage in megabytes.
    inference_latency_ms:
        Mean per-sample inference latency in milliseconds.
    tokens_per_second:
        Throughput in tokens per second (if applicable).
    """

    model_name: str = ""
    total_params: int = 0
    trainable_params: int = 0
    train_time_seconds: float = 0.0
    vram_peak_mb: float = 0.0
    inference_latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self):
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
# Parameter counting
# ---------------------------------------------------------------------------


def count_parameters(model) -> Tuple[int, int]:
    """Count total and trainable parameters of a PyTorch model.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` (or PEFT wrapped model).

    Returns
    -------
    Tuple[int, int]
        ``(total_params, trainable_params)``
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# VRAM tracking
# ---------------------------------------------------------------------------


def get_vram_peak_mb() -> float:
    """Return the peak GPU memory allocation in megabytes.

    Returns 0.0 if no CUDA device is available.

    Returns
    -------
    float
        Peak VRAM in MB.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram_tracking() -> None:
    """Reset peak VRAM tracking statistics.

    Call this before a measurement block to get an accurate peak for that
    block alone.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_vram() -> Generator[dict, None, None]:
    """Context manager that measures VRAM used within the block.

    Yields
    ------
    dict
        Mutable dict; after the block exits it will contain:
        ``{"vram_peak_mb": float, "vram_before_mb": float}``.

    Example
    -------
    >>> with track_vram() as vram_info:
    ...     model(inputs)
    >>> print(vram_info["vram_peak_mb"])
    """
    info: dict = {"vram_peak_mb": 0.0, "vram_before_mb": 0.0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        info["vram_before_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
    try:
        yield info
    finally:
        if torch.cuda.is_available():
            info["vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_inference_latency(
    fn,
    n_runs: int = 20,
    n_warmup: int = 3,
) -> Tuple[float, float]:
    """Measure inference latency of a callable with CUDA synchronization.

    Parameters
    ----------
    fn:
        Zero-argument callable that performs one inference step.
    n_runs:
        Number of timed runs.
    n_warmup:
        Number of warmup runs (excluded from timing).

    Returns
    -------
    Tuple[float, float]
        ``(mean_ms, std_ms)`` — mean and standard deviation of latency in ms.

    Example
    -------
    >>> mean_ms, std_ms = measure_inference_latency(lambda: model(inputs))
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies: List[float] = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    import numpy as np
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    return mean_ms, std_ms
