"""MCP-shaped leaderboard + per-server breakdown + reliability-per-VRAM (η,
spec §4.6) + Pareto frontier chart (spec §10 Phase 3). New, not vendored —
quant-toolcall-bench's report/leaderboard.py and report/published.py are
BFCL-tier-shaped (svr/tsa/ac/fcr columns) and have no notion of an MCP
server tier or SVR-MCP/TSR; this module reads the same real result.json
files runner.py already writes and builds an MCP-native view on top,
reusing metrics/deltas.py's compute_eta and report/pareto.py's pareto_front
(both vendored) rather than reimplementing either formula.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from quantmcp.metrics.deltas import compute_eta
from quantmcp.report.pareto import pareto_front
from quantmcp.report.published import sanitize_model_name
from quantmcp.report.tables import _md_table

# SCI per tier, computed once from the live tool schemas of all four real
# servers in a single z-normalized corpus (see docs/RUN_REAL.md's "Schema
# Complexity Index vs. degradation (H2)" section for the exact command and
# discussion). Hardcoded here rather than recomputed on every leaderboard
# build because computing it live would require launching every tier's
# server from this otherwise-synchronous report step; the values are
# deterministic for these fixed fixtures/schemas and are cheap to
# regenerate by re-running that command if a fixture ever changes.
TIER_SCI: dict[str, float] = {
    "filesystem": 0.3329,
    "git": 0.1148,
    "memory": -0.3586,
    "sqlite": -0.5154,
}

_ETA_WEIGHTS = (0.5, 0.5)  # w1, w2 in spec §4.6 — equal weight, no other guidance given


@dataclass(frozen=True)
class LeaderboardRow:
    model: str
    quant: str
    tier: str
    n: int
    svr_mcp: float
    tsr: float
    vram_gb: float | None
    eta: float | None
    pareto_optimal: bool = False


def _reliability(svr_mcp: float, tsr: float) -> float:
    w1, w2 = _ETA_WEIGHTS
    return w1 * svr_mcp + w2 * tsr


def _mark_pareto_optimal(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    """Flag which rows sit on the reliability-vs-VRAM Pareto frontier,
    reusing the vendored pareto_front selector (lower VRAM, higher
    reliability) instead of a bespoke frontier implementation. Rows with no
    VRAM reading can't be placed on the frontier and are left unmarked."""
    points = [
        {"idx": i, "vram_gb": r.vram_gb, "reliability": _reliability(r.svr_mcp, r.tsr)}
        for i, r in enumerate(rows)
        if r.vram_gb is not None
    ]
    front_idx = {p["idx"] for p in pareto_front(points, x_key="vram_gb", y_key="reliability")}
    return [replace(r, pareto_optimal=i in front_idx) for i, r in enumerate(rows)]


def write_pareto_chart(rows: list[LeaderboardRow], output_path: Path) -> bool:
    """Write a self-contained HTML scatter of reliability vs. peak VRAM,
    highlighting the Pareto-optimal (model, quant, tier) configs. Plotly
    lives in the optional `space` extra used by the HF Space (Phase 4), not
    in the offline verify gate, so this degrades to a no-op when it isn't
    installed rather than making `quantmcp leaderboard` depend on it.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
    except ImportError:
        return False

    plotted = [r for r in rows if r.vram_gb is not None]
    if not plotted:
        return False

    fig = go.Figure()
    variants = ((True, "Pareto-optimal", "star"), (False, "dominated", "circle"))
    for on_front, label, symbol in variants:
        subset = [r for r in plotted if r.pareto_optimal is on_front]
        if not subset:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r.vram_gb for r in subset],
                y=[_reliability(r.svr_mcp, r.tsr) for r in subset],
                mode="markers",
                name=label,
                text=[f"{r.model} / {r.quant} / {r.tier}" for r in subset],
                marker={"size": 11, "symbol": symbol},
            )
        )
    fig.update_layout(
        title="Reliability vs. peak VRAM — Pareto frontier (spec §4.6)",
        xaxis_title="Peak VRAM (GB)",
        yaxis_title="Reliability (0.5·SVR-MCP + 0.5·TSR)",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    return True


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in sorted(results_dir.rglob("*.result.json")):
        try:
            data = json.loads(p.read_text())
            if "svr_mcp" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return results


def _row_from_result(data: dict[str, Any]) -> LeaderboardRow:
    cfg = data.get("config", {})
    svr_mcp = float(data.get("svr_mcp", 0.0))
    tsr = float(data.get("tsr", 0.0))
    vram_gb = data.get("vram_gb")
    w1, w2 = _ETA_WEIGHTS
    eta = compute_eta(w1 * svr_mcp + w2 * tsr, vram_gb)
    quant = str(cfg.get("quant", "?"))
    # Local GGUF paths (e.g. "/home/x/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf")
    # would otherwise leak the local username/filesystem layout into any
    # published leaderboard or HF dataset — reuses the vendored sanitizer
    # rather than a bespoke path-scrubbing implementation.
    model = sanitize_model_name(str(cfg.get("model", "?")), quant)
    return LeaderboardRow(
        model=model,
        quant=quant,
        tier=str(cfg.get("server_tier", "?")),
        n=int(data.get("n", 0)),
        svr_mcp=svr_mcp,
        tsr=tsr,
        vram_gb=vram_gb,
        eta=eta,
    )


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_mcp_leaderboard(
    results_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build the MCP-native leaderboard: one row per real (model, quant,
    tier) result, plus a per-tier breakdown with mean SVR-MCP/TSR/η and
    that tier's Schema Complexity Index (H2, spec §4.3)."""
    results_dir = Path(results_dir)
    results = _load_results(results_dir)
    rows = _mark_pareto_optimal([_row_from_result(r) for r in results])

    per_tier: dict[str, list[LeaderboardRow]] = defaultdict(list)
    for row in rows:
        per_tier[row.tier].append(row)

    tier_breakdown = []
    for tier in sorted(per_tier):
        tier_rows = per_tier[tier]
        n_rows = len(tier_rows)
        mean_svr = sum(r.svr_mcp for r in tier_rows) / n_rows
        mean_tsr = sum(r.tsr for r in tier_rows) / n_rows
        etas = [r.eta for r in tier_rows if r.eta is not None]
        mean_eta = sum(etas) / len(etas) if etas else None
        tier_breakdown.append(
            {
                "tier": tier,
                "n_configs": n_rows,
                "mean_svr_mcp": mean_svr,
                "mean_tsr": mean_tsr,
                "mean_eta": mean_eta,
                "sci": TIER_SCI.get(tier),
            }
        )

    leaderboard: dict[str, Any] = {
        "rows": [row.__dict__ for row in rows],
        "tier_breakdown": tier_breakdown,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mcp_leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

        row_cols = [
            "model",
            "quant",
            "tier",
            "n",
            "svr_mcp",
            "tsr",
            "vram_gb",
            "eta",
            "pareto_optimal",
        ]
        with (out / "mcp_runs.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_cols)
            for row in rows:
                writer.writerow([getattr(row, c) for c in row_cols])

        tier_cols = ["tier", "n_configs", "mean_svr_mcp", "mean_tsr", "mean_eta", "sci"]
        with (out / "mcp_tier_breakdown.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(tier_cols)
            for t in tier_breakdown:
                writer.writerow([t[c] for c in tier_cols])

        row_md = _md_table(row_cols, [[_fmt(getattr(row, c)) for c in row_cols] for row in rows])
        tier_md = _md_table(tier_cols, [[_fmt(t[c]) for c in tier_cols] for t in tier_breakdown])
        (out / "mcp_leaderboard.md").write_text(
            f"# MCP Leaderboard\n\n{row_md}\n\n# Per-server breakdown\n\n{tier_md}\n"
        )

        leaderboard["pareto_chart_written"] = write_pareto_chart(rows, out / "pareto.html")

    return leaderboard
