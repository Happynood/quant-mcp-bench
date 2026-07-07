from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from quantmcp.report.mcp_leaderboard import build_mcp_leaderboard, write_pareto_chart


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


def test_row_from_result_sanitizes_local_model_path():
    from quantmcp.report.mcp_leaderboard import _row_from_result

    row = _row_from_result(
        {
            "n": 10,
            "svr_mcp": 0.8,
            "tsr": 0.6,
            "vram_gb": 2.0,
            "config": {
                "model": "/home/someuser/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
                "quant": "Q4_K_M",
                "server_tier": "filesystem",
            },
        }
    )
    assert row.model == "Qwen3-0.6B"
    assert "/home/" not in row.model


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
    assert tiers["filesystem"]["sci"] == pytest.approx(0.2286)


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
    # a lone row with no VRAM reading can't be placed on the frontier
    assert leaderboard["rows"][0]["pareto_optimal"] is False


def test_build_mcp_leaderboard_marks_pareto_optimal(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # same reliability, more VRAM -> dominated
    _write_result(results_dir / "q1.result.json", "modelA", "q1", "filesystem", 10, 0.8, 0.8, 4.0)
    # same reliability, less VRAM -> dominates q1
    _write_result(results_dir / "q2.result.json", "modelA", "q2", "filesystem", 10, 0.8, 0.8, 2.0)

    leaderboard = build_mcp_leaderboard(results_dir)
    rows = {r["quant"]: r for r in leaderboard["rows"]}
    assert rows["q2"]["pareto_optimal"] is True
    assert rows["q1"]["pareto_optimal"] is False


def test_build_mcp_leaderboard_writes_pareto_chart_when_plotly_available(tmp_path: Path):
    pytest.importorskip("plotly")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "fp16.result.json", "modelA", "fp16", "sqlite", 10, 0.5, 0.4, 1.5)

    output_dir = tmp_path / "leaderboard"
    leaderboard = build_mcp_leaderboard(results_dir, output_dir)

    assert leaderboard["pareto_chart_written"] is True
    assert (output_dir / "pareto.html").exists()


def test_write_pareto_chart_skips_gracefully_without_plotly(tmp_path: Path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("plotly", "plotly.graph_objects"):
            raise ImportError("plotly not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    row = _row_from_result_for_test()
    written = write_pareto_chart([row], tmp_path / "pareto.html")

    assert written is False
    assert not (tmp_path / "pareto.html").exists()


def _row_from_result_for_test():
    from quantmcp.report.mcp_leaderboard import _row_from_result

    return _row_from_result(
        {
            "n": 10,
            "svr_mcp": 0.8,
            "tsr": 0.6,
            "vram_gb": 2.0,
            "config": {"model": "m", "quant": "q", "server_tier": "t"},
        }
    )


def test_write_pareto_chart_no_op_when_no_vram_data(tmp_path: Path):
    pytest.importorskip("plotly")
    from quantmcp.report.mcp_leaderboard import _row_from_result

    row = _row_from_result(
        {
            "n": 10,
            "svr_mcp": 0.8,
            "tsr": 0.6,
            "vram_gb": None,
            "config": {"model": "m", "quant": "q", "server_tier": "t"},
        }
    )
    written = write_pareto_chart([row], tmp_path / "pareto.html")
    assert written is False
