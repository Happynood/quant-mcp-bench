"""Per-tool SCI-vs-Δ regression (H2, spec §4.3, Phase 7).

The original H2 write-up (docs/RUN_REAL.md) only had one aggregate point per
tier (4 tiers = 4 points), never enough for a regression coefficient to mean
anything. This module computes the same relationship at per-*tool*
granularity instead: each tool that has at least one task tagged with
`tool:` (tasks/base.py) contributes its own (SCI, Δ) point, using that
tool's own live schema and its own task-level pass-rate delta between two
quant levels. This can turn a 4-point tier-level analysis into one with as
many points as there are distinct covered tools (up to 39 in this project's
current 4-tier corpus).

New module, not vendored: quant-toolcall-bench has no per-tool schema
complexity concept, only BFCL's coarser per-category difficulty tiers.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantmcp.schema.complexity import compute_sci, extract_features


@dataclass(frozen=True)
class ToolDeltaPoint:
    tool: str
    tier: str
    sci: float
    delta_svr: float
    n_baseline: int
    n_quant: int


@dataclass(frozen=True)
class SciRegressionResult:
    points: list[ToolDeltaPoint]
    slope: float
    intercept: float
    slope_ci: tuple[float, float]
    n: int


def _tool_pass_rates(result_files: list[Path], quant: str) -> dict[str, tuple[int, int]]:
    """Return {tool: (n_pass, n_total)} pooled across every result file whose
    config.quant == quant, using each file's per-instance "instances" data."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in result_files:
        data = json.loads(Path(path).read_text())
        if data.get("config", {}).get("quant") != quant:
            continue
        for instance in data.get("instances", []):
            tool = instance.get("tool")
            if not tool:
                continue
            totals[tool][1] += 1
            if instance.get("svr_pass"):
                totals[tool][0] += 1
    return {tool: (n_pass, n_total) for tool, (n_pass, n_total) in totals.items()}


def _ols_slope_intercept(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return 0.0, mean_y
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _bootstrap_slope_ci(
    xs: list[float], ys: list[float], n_resamples: int = 2000, seed: int | None = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI on the OLS slope, resampling (x, y) pairs with
    replacement -- avoids assuming normally-distributed residuals, matching
    this project's existing preference for bootstrap CIs (metrics/stats.py)
    over parametric ones."""
    rng = random.Random(seed)
    n = len(xs)
    slopes: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        rx = [xs[i] for i in idx]
        ry = [ys[i] for i in idx]
        if len(set(rx)) < 2:
            continue
        slope, _ = _ols_slope_intercept(rx, ry)
        slopes.append(slope)
    if not slopes:
        return (0.0, 0.0)
    slopes.sort()
    lo = slopes[int(0.025 * len(slopes))]
    hi = slopes[min(len(slopes) - 1, int(0.975 * len(slopes)))]
    return (lo, hi)


def compute_sci_delta_regression(
    result_files: list[Path],
    schemas: list[dict[str, Any]],
    baseline_quant: str = "fp16",
    degraded_quant: str = "Q4_K_M",
    bootstrap_seed: int | None = 0,
) -> SciRegressionResult:
    """Build one (SCI, Δ SVR-MCP) point per tool that has real pass-rate data
    at both `baseline_quant` and `degraded_quant`, pooling across every model
    present in `result_files`, then fit Δ ~ SCI with a bootstrap CI on the
    slope.

    `schemas` is the live tool-schema dump (see schema/dump.py /
    `quantmcp dump-schemas`) used to compute each tool's SCI in the same
    corpus-relative z-normalization as the rest of this project's SCI
    numbers -- must include every tool referenced by `result_files`'
    per-instance `tool` tags for their SCI to be defined.
    """
    features = [
        extract_features(s["name"], s["input_schema"], s.get("description", "")) for s in schemas
    ]
    sci_by_tool = compute_sci(features)
    tier_by_tool = {s["name"]: s["tier"] for s in schemas}

    baseline = _tool_pass_rates(result_files, baseline_quant)
    degraded = _tool_pass_rates(result_files, degraded_quant)

    points: list[ToolDeltaPoint] = []
    for tool in sorted(set(baseline) & set(degraded)):
        if tool not in sci_by_tool:
            continue
        n_pass_b, n_b = baseline[tool]
        n_pass_d, n_d = degraded[tool]
        if n_b == 0 or n_d == 0:
            continue
        delta = (n_pass_b / n_b) - (n_pass_d / n_d)
        points.append(
            ToolDeltaPoint(
                tool=tool,
                tier=tier_by_tool.get(tool, "?"),
                sci=sci_by_tool[tool],
                delta_svr=delta,
                n_baseline=n_b,
                n_quant=n_d,
            )
        )

    if len(points) < 2:
        return SciRegressionResult(
            points=points, slope=0.0, intercept=0.0, slope_ci=(0.0, 0.0), n=len(points)
        )

    xs = [p.sci for p in points]
    ys = [p.delta_svr for p in points]
    slope, intercept = _ols_slope_intercept(xs, ys)
    slope_ci = _bootstrap_slope_ci(xs, ys, seed=bootstrap_seed)
    return SciRegressionResult(
        points=points, slope=slope, intercept=intercept, slope_ci=slope_ci, n=len(points)
    )
