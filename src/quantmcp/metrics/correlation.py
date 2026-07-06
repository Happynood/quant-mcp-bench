"""Spearman rank correlation for CBC (spec §4.5) — the only genuinely new
metrics/ module besides metrics/core.py; deltas.py and stats.py are
vendored verbatim and don't need this.
"""

from __future__ import annotations


def _average_ranks(values: list[float]) -> list[float]:
    """Rank `values` ascending, giving tied values the average of the ranks
    they span (the standard tie-handling convention for Spearman's rho)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman's rank correlation coefficient between `xs` and `ys`.

    Implemented as the Pearson correlation of average ranks rather than the
    simplified 1 - 6*sum(d^2)/(n^3-n) formula, since the latter is only
    valid without ties and CBC's (model, quant) sample is small enough that
    ties are common.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 data points to compute a correlation")

    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)

    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom
