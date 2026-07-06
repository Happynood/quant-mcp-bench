# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations


def compute_delta(baseline: float, current: float) -> float:
    """Absolute degradation: baseline - current (positive = degraded)."""
    return baseline - current


def compute_delta_rel(baseline: float, current: float) -> float | None:
    """Relative degradation: (baseline - current) / baseline. None if baseline is 0."""
    if baseline == 0.0:
        return None
    return (baseline - current) / baseline


def compute_eta(fcr: float, peak_vram_gb: float | None) -> float | None:
    """Efficiency score: FCR / peak_vram_gb. None if vram is unavailable or zero."""
    if peak_vram_gb is None or peak_vram_gb == 0.0:
        return None
    return fcr / peak_vram_gb
