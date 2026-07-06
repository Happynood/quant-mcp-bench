# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from typing import Any


def pareto_front(
    points: list[dict[str, Any]],
    x_key: str = "vram_gb",
    y_key: str = "fcr",
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> list[dict[str, Any]]:
    """Return the Pareto-optimal subset (lower x + higher y by default)."""
    if not points:
        return []

    dominated: set[int] = set()
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i == j or i in dominated:
                continue
            xi, yi = float(points[i][x_key]), float(points[i][y_key])
            xj, yj = float(points[j][x_key]), float(points[j][y_key])

            j_better_x = (xj < xi) if minimize_x else (xj > xi)
            j_better_y = (yj > yi) if maximize_y else (yj < yi)
            j_eq_x = xi == xj
            j_eq_y = yi == yj

            if (j_better_x or j_eq_x) and (j_better_y or j_eq_y) and (j_better_x or j_better_y):
                dominated.add(i)

    return [p for i, p in enumerate(points) if i not in dominated]
