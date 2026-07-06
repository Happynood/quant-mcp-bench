from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantmcp.report.mcp_leaderboard import build_mcp_leaderboard


def _write_result(
    path: Path,
    model: str,
    quant: str,
    tier: str,
    n: int,
    svr_mcp: float,
    tsr: float,
    vram_gb: float | None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "n": n,
                "svr_mcp": svr_mcp,
                "tsr": tsr,
                "vram_gb": vram_gb,
                "config": {"model": model, "quant": quant, "server_tier": tier},
            }
        )
    )


def test_build_mcp_leaderboard_computes_eta_and_tier_breakdown(tmp_path: Path):
    results_dir = tmp_path / "results" / "modelA-filesystem"
    results_dir.mkdir(parents=True)
    _write_result(
        results_dir / "fp16.result.json", "modelA", "fp16", "filesystem", 10, 0.8, 0.6, 2.0
    )
    other_dir = tmp_path / "results" / "modelA-git"
    other_dir.mkdir(parents=True)
    _write_result(other_dir / "fp16.result.json", "modelA", "fp16", "git", 10, 1.0, 1.0, 2.0)

    leaderboard = build_mcp_leaderboard(tmp_path / "results")

    assert len(leaderboard["rows"]) == 2
    fs_row = next(r for r in leaderboard["rows"] if r["tier"] == "filesystem")
    # eta = (0.5*0.8 + 0.5*0.6) / 2.0 = 0.35
    assert fs_row["eta"] == pytest.approx(0.35)

    tiers = {t["tier"]: t for t in leaderboard["tier_breakdown"]}
    assert tiers["filesystem"]["mean_svr_mcp"] == pytest.approx(0.8)
    assert tiers["git"]["mean_svr_mcp"] == pytest.approx(1.0)
    assert tiers["filesystem"]["sci"] == pytest.approx(0.2063)


def test_build_mcp_leaderboard_writes_output_files(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "fp16.result.json", "modelA", "fp16", "sqlite", 10, 0.5, 0.4, 1.5)

    output_dir = tmp_path / "leaderboard"
    build_mcp_leaderboard(results_dir, output_dir)

    assert (output_dir / "mcp_leaderboard.json").exists()
    assert (output_dir / "mcp_runs.csv").exists()
    assert (output_dir / "mcp_tier_breakdown.csv").exists()
    assert (output_dir / "mcp_leaderboard.md").exists()


def test_build_mcp_leaderboard_handles_missing_vram(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "fp16.result.json", "modelA", "fp16", "sqlite", 10, 0.5, 0.4, None)

    leaderboard = build_mcp_leaderboard(results_dir)
    assert leaderboard["rows"][0]["eta"] is None
