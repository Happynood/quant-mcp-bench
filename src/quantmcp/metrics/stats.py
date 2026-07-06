# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import random


def bootstrap_ci(
    data: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of data.

    Returns (lower, upper) bounds.
    """
    if not data:
        raise ValueError("data must not be empty")
    if len(data) == 1:
        return (data[0], data[0])

    rng = random.Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - confidence
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1.0 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, lo_idx)
    hi_idx = min(n_resamples - 1, hi_idx)
    return (means[lo_idx], means[hi_idx])
