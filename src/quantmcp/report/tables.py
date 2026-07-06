# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from typing import Any

from quantmcp.metrics.deltas import compute_delta, compute_delta_rel


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_delta_table(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    cfg = current.get("config", {})
    rows: list[list[str]] = []
    for metric in ("svr", "tsa", "ac", "abstention", "fcr"):
        base_val = float(baseline.get(metric, 0))
        curr_val = float(current.get(metric, 0))
        delta = compute_delta(base_val, curr_val)
        delta_rel = compute_delta_rel(base_val, curr_val)
        rel_str = f"{delta_rel:.3f}" if delta_rel is not None else "—"
        rows.append(
            [
                f"Δ{metric.upper()}",
                cfg.get("model", "—"),
                cfg.get("quant", "—"),
                cfg.get("backend", "—"),
                f"{delta:+.3f}",
                rel_str,
            ]
        )
    headers = ["Metric", "Model", "Quant", "Backend", "ΔFCR", "Δrel"]
    return _md_table(headers, rows)
