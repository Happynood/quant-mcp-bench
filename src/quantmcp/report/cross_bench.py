"""Cross-Benchmark Consistency (CBC, spec §4.5): correlates this project's
own SVR-MCP quantization deltas against QuantCall's already-published BFCL
SVR deltas for the same (model, quant) pairs. New, not vendored — QuantCall
has no notion of a second benchmark to compare against.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantmcp.metrics.correlation import spearman_correlation

_MODEL_FAMILY_MARKERS = ["Qwen3-0.6B", "Qwen3-1.7B", "Llama-3.2-1B"]

# Every family's baseline is its own fp16 run, except Qwen3-1.7B: its bf16
# weights (~4.07 GB) hit a real CUDA OOM on a 4GB card at any usable context
# length (confirmed at n_ctx=4096 and n_ctx=2048; only n_ctx=512 loads, too
# small for real tool-schema prompts) -- the same hardware limitation
# quant-toolcall-bench already hit and documented for this exact model, so
# its own published BFCL table also uses Q8_0 as this family's baseline.
_FAMILY_BASELINE_QUANT: dict[str, str] = {"Qwen3-1.7B": "Q8_0"}


def _model_family(model_path: str) -> str | None:
    for marker in _MODEL_FAMILY_MARKERS:
        if marker in model_path:
            return marker
    return None


@dataclass(frozen=True)
class CbcResult:
    rho: float
    n_pairs: int
    table: list[dict[str, Any]]


def compute_cbc(result_files: list[Path], bfcl_results_path: Path) -> CbcResult:
    """Pool this project's SVR-MCP across every given result file (weighted
    by each file's task count `n`) per (model family, quant), take the delta
    against that family's own fp16 SVR-MCP, and correlate it (Spearman)
    against the equivalent BFCL SVR delta for the same (model, quant) pairs.
    """
    bfcl_data = json.loads(bfcl_results_path.read_text())
    bfcl_svr: dict[tuple[str, str], float] = {
        (e["model"], e["quant"]): e["svr_bfcl"] for e in bfcl_data["entries"]
    }

    # A result file's server tier isn't distinguished here — every tier
    # provided is pooled into one SVR-MCP per (model, quant) for the CBC
    # computation itself; the per-server breakdown lives separately in
    # report/mcp_leaderboard.py (spec §4.7), not in this pooled number.
    pooled: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for path in result_files:
        data = json.loads(Path(path).read_text())
        cfg = data.get("config", {})
        family = _model_family(str(cfg.get("model", "")))
        quant = cfg.get("quant")
        if family is None or quant is None:
            continue
        pooled[(family, quant)].append((data.get("n", 0), data.get("svr_mcp", 0.0)))

    mcp_svr: dict[tuple[str, str], float] = {}
    for key, samples in pooled.items():
        total_n = sum(n for n, _ in samples)
        if total_n > 0:
            mcp_svr[key] = sum(n * svr for n, svr in samples) / total_n

    families = sorted({family for family, _ in mcp_svr})
    delta_bfcl: list[float] = []
    delta_mcp: list[float] = []
    table: list[dict[str, Any]] = []
    for family in families:
        baseline_quant = _FAMILY_BASELINE_QUANT.get(family, "fp16")
        baseline_bfcl = bfcl_svr.get((family, baseline_quant))
        baseline_mcp = mcp_svr.get((family, baseline_quant))
        if baseline_bfcl is None or baseline_mcp is None:
            continue
        for (fam, quant), svr in sorted(mcp_svr.items()):
            if fam != family or quant == baseline_quant:
                continue
            bfcl = bfcl_svr.get((family, quant))
            if bfcl is None:
                continue
            d_bfcl = bfcl - baseline_bfcl
            d_mcp = svr - baseline_mcp
            delta_bfcl.append(d_bfcl)
            delta_mcp.append(d_mcp)
            table.append(
                {
                    "model": family,
                    "quant": quant,
                    "baseline_quant": baseline_quant,
                    "delta_svr_bfcl": d_bfcl,
                    "delta_svr_mcp": d_mcp,
                }
            )

    if len(delta_bfcl) < 2:
        raise ValueError(
            "need at least 2 (model, quant) pairs with both BFCL and MCP deltas to compute CBC"
        )

    rho = spearman_correlation(delta_bfcl, delta_mcp)
    return CbcResult(rho=rho, n_pairs=len(delta_bfcl), table=table)
