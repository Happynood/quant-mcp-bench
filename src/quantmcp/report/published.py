# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
"""Build the published results dataset (runs.csv + leaderboard.csv).

Single source of truth for the schema shipped to the happynood/quantmcp-results
HF dataset. `docs/RESULTS_SCHEMA.md` documents these same column lists and is
checked against them by tests/test_report.py so the two can never drift apart.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from quantmcp.metrics.deltas import compute_delta_rel, compute_eta
from quantmcp.metrics.stats import bootstrap_ci

RUNS_COLS = [
    "model",
    "quant",
    "backend",
    "decoding",
    "tier",
    "seed",
    "sample_size",
    "svr",
    "tsa",
    "ac",
    "abstention",
    "fcr",
    "vram_gb",
    "git_commit",
    "config_sha256",
    "dataset_sha256",
    "timestamp",
]

LEADERBOARD_COLS = [
    "model",
    "quant",
    "backend",
    "decoding",
    "tier",
    "n_seeds",
    "fcr_mean",
    "fcr_ci_low",
    "fcr_ci_high",
    "svr_mean",
    "tsa_mean",
    "ac_mean",
    "abstention_mean",
    "vram_gb",
    "eta",
    "delta_fcr_rel",
    "delta_ac_rel",
    "baseline_quant",
]

# Higher rank = less lossy precision. Used to pick the Δ baseline quant per
# (model, backend, decoding, tier) scope: the highest-ranked quant actually
# present in that scope. fp16 wins when it fits; otherwise the best available
# quant is used and labeled explicitly via the baseline_quant column.
PRECISION_RANK: dict[str, int] = {
    "fp16": 4,
    "Q8_0": 3,
    "Q5_K_M": 2,
    "Q4_K_M": 1,
    "AWQ": 0,
    "GPTQ": 0,
}

GroupKey = tuple[str, str, str, str, str]


def _tier_str(tiers: list[str]) -> str:
    return "+".join(tiers)


_FP16_ALIASES = ("fp16", "bf16", "f16")


def sanitize_model_name(raw_model: str, quant: str) -> str:
    """Derive a canonical, path-free model name from a local GGUF path.

    Local GGUF filenames encode both the model and the quant, e.g.
    "/home/x/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf" for quant="Q4_K_M". If the
    raw path is published verbatim, every quant of the same model gets a
    different "model" string, which breaks (model,backend,decoding,tier)
    grouping in aggregate_leaderboard() -- every quant lands in its own
    scope and delta/baseline computation degenerates to comparing a quant
    against itself. This strips the directory, extension, and the trailing
    quant suffix (including fp16/bf16/f16 filename aliases) so all quants of
    one model share the same canonical name. Already-clean names (HF repo
    IDs like "Qwen/Qwen3-0.6B", or a name with no matching suffix) pass
    through unchanged.
    """
    name = raw_model.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]

    suffixes = {quant} | (set(_FP16_ALIASES) if quant == "fp16" else set())
    for suf in suffixes:
        for candidate in (f"-{suf}", f"-{suf.upper()}", f"-{suf.lower()}"):
            if name.endswith(candidate):
                name = name[: -len(candidate)]
                break

    # bartowski-style GGUF repos duplicate the org in the filename, e.g.
    # "Qwen_Qwen3-0.6B" for org "Qwen", model "Qwen3-0.6B".
    if "_" in name:
        org, _, rest = name.partition("_")
        if rest.startswith(org):
            name = rest

    return name


def run_row(r: dict[str, Any]) -> dict[str, Any]:
    """Flatten one result.json into a runs.csv row (one row per real run)."""
    cfg = r.get("config", {})
    manifest = r.get("manifest", {})
    return {
        "model": sanitize_model_name(cfg.get("model", ""), cfg.get("quant", "")),
        "quant": cfg.get("quant", ""),
        "backend": cfg.get("backend", ""),
        "decoding": cfg.get("decoding", ""),
        "tier": _tier_str(cfg.get("tiers", [])),
        "seed": cfg.get("seed", ""),
        "sample_size": cfg.get("sample_size", ""),
        "svr": r.get("svr", 0.0),
        "tsa": r.get("tsa", 0.0),
        "ac": r.get("ac", 0.0),
        "abstention": r.get("abstention", 0.0),
        "fcr": r.get("fcr", 0.0),
        "vram_gb": r.get("vram_gb"),
        "git_commit": manifest.get("git_commit", ""),
        "config_sha256": manifest.get("config_sha256", ""),
        "dataset_sha256": manifest.get("dataset_sha256", ""),
        "timestamp": manifest.get("timestamp", ""),
    }


def _group_key(row: dict[str, Any]) -> GroupKey:
    return (row["model"], row["quant"], row["backend"], row["decoding"], row["tier"])


def _scope_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["model"], row["backend"], row["decoding"], row["tier"])


def _pick_baseline_quants(run_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], str]:
    baseline_by_scope: dict[tuple[str, str, str, str], str] = {}
    for row in run_rows:
        scope = _scope_key(row)
        quant = row["quant"]
        rank = PRECISION_RANK.get(quant, -1)
        current = baseline_by_scope.get(scope)
        if current is None or rank > PRECISION_RANK.get(current, -1):
            baseline_by_scope[scope] = quant
    return baseline_by_scope


def aggregate_leaderboard(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed run rows into one row per (model,quant,backend,decoding,tier)."""
    groups: dict[GroupKey, list[dict[str, Any]]] = {}
    for row in run_rows:
        groups.setdefault(_group_key(row), []).append(row)

    baseline_by_scope = _pick_baseline_quants(run_rows)

    agg_rows: dict[GroupKey, dict[str, Any]] = {}
    for key, rows in groups.items():
        model, quant, backend, decoding, tier = key
        n = len(rows)
        fcr_vals = [float(r["fcr"]) for r in rows]
        svr_vals = [float(r["svr"]) for r in rows]
        tsa_vals = [float(r["tsa"]) for r in rows]
        ac_vals = [float(r["ac"]) for r in rows]
        abst_vals = [float(r["abstention"]) for r in rows]
        vram_vals = [float(r["vram_gb"]) for r in rows if isinstance(r["vram_gb"], (int, float))]

        fcr_mean = sum(fcr_vals) / n
        fcr_lo, fcr_hi = bootstrap_ci(fcr_vals, n_resamples=2000, seed=42)
        vram_gb = (sum(vram_vals) / len(vram_vals)) if vram_vals else None

        agg_rows[key] = {
            "model": model,
            "quant": quant,
            "backend": backend,
            "decoding": decoding,
            "tier": tier,
            "n_seeds": n,
            "fcr_mean": fcr_mean,
            "fcr_ci_low": fcr_lo,
            "fcr_ci_high": fcr_hi,
            "svr_mean": sum(svr_vals) / n,
            "tsa_mean": sum(tsa_vals) / n,
            "ac_mean": sum(ac_vals) / n,
            "abstention_mean": sum(abst_vals) / n,
            "vram_gb": vram_gb,
            "eta": compute_eta(fcr_mean, vram_gb),
            "delta_fcr_rel": None,
            "delta_ac_rel": None,
            "baseline_quant": baseline_by_scope[(model, backend, decoding, tier)],
        }

    for key, row in agg_rows.items():
        model, quant, backend, decoding, tier = key
        scope = (model, backend, decoding, tier)
        baseline_quant = baseline_by_scope[scope]
        if quant == baseline_quant:
            continue
        baseline_row = agg_rows.get((model, baseline_quant, backend, decoding, tier))
        if baseline_row is None:
            continue
        row["delta_fcr_rel"] = compute_delta_rel(baseline_row["fcr_mean"], row["fcr_mean"])
        row["delta_ac_rel"] = compute_delta_rel(baseline_row["ac_mean"], row["ac_mean"])

    return sorted(
        agg_rows.values(),
        key=lambda r: (
            r["model"],
            r["backend"],
            r["decoding"],
            r["tier"],
            -PRECISION_RANK.get(r["quant"], -1),
        ),
    )


def write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in cols})
