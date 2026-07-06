"""Repeat-run stability summary: aggregates several independent runs of the
*same* (model, quant, tier) config into a mean + range, to make run-to-run
GPU decode variance an explicit, reported number rather than a one-off
caveat in prose. New, not vendored — quant-toolcall-bench's BFCL runs
sample from a large task pool per seed, so seed-to-seed variance is a
different phenomenon than repeating an identical small fixed task set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepeatStability:
    n_repeats: int
    svr_mcp_mean: float
    svr_mcp_min: float
    svr_mcp_max: float
    tsr_mean: float
    tsr_min: float
    tsr_max: float


def aggregate_repeats(results: list[dict[str, Any]]) -> RepeatStability:
    """Summarize `n_repeats` independent result.json payloads (already
    parsed) for the same config as a mean and a [min, max] range on both
    SVR-MCP and TSR."""
    if not results:
        raise ValueError("need at least 1 result to aggregate")
    svr_values = [float(r["svr_mcp"]) for r in results]
    tsr_values = [float(r["tsr"]) for r in results]
    n = len(results)
    return RepeatStability(
        n_repeats=n,
        svr_mcp_mean=sum(svr_values) / n,
        svr_mcp_min=min(svr_values),
        svr_mcp_max=max(svr_values),
        tsr_mean=sum(tsr_values) / n,
        tsr_min=min(tsr_values),
        tsr_max=max(tsr_values),
    )
