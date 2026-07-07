from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantmcp.report.cross_bench import compute_cbc


@pytest.fixture
def bfcl_results_path(tmp_path: Path) -> Path:
    path = tmp_path / "bfcl.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"model": "FamilyA", "quant": "fp16", "svr_bfcl": 0.8},
                    {"model": "FamilyA", "quant": "Q4_K_M", "svr_bfcl": 0.6},
                    {"model": "FamilyB", "quant": "fp16", "svr_bfcl": 0.3},
                    {"model": "FamilyB", "quant": "Q4_K_M", "svr_bfcl": 0.35},
                ]
            }
        )
    )
    return path


def _write_result(
    tmp_path: Path, name: str, model: str, quant: str, n: int, svr_mcp: float
) -> Path:
    path = tmp_path / name
    config = {"model": model, "quant": quant}
    path.write_text(json.dumps({"n": n, "svr_mcp": svr_mcp, "config": config}))
    return path


def test_compute_cbc_perfect_positive_agreement(
    tmp_path: Path, bfcl_results_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Both families decline under Q4_K_M in this project's own numbers too,
    # in the same direction and rough relative size as the fixture's BFCL
    # deltas above -> should correlate strongly.
    import quantmcp.report.cross_bench as cb

    monkeypatch.setattr(cb, "_MODEL_FAMILY_MARKERS", ["FamilyA", "FamilyB"])

    files = [
        _write_result(tmp_path, "a-fp16.json", "FamilyA-x.gguf", "fp16", 10, 0.9),
        _write_result(tmp_path, "a-q4.json", "FamilyA-x.gguf", "Q4_K_M", 10, 0.5),
        _write_result(tmp_path, "b-fp16.json", "FamilyB-x.gguf", "fp16", 10, 0.2),
        _write_result(tmp_path, "b-q4.json", "FamilyB-x.gguf", "Q4_K_M", 10, 0.3),
    ]

    result = compute_cbc(files, bfcl_results_path)
    assert result.n_pairs == 2
    assert result.rho == pytest.approx(1.0)


def test_compute_cbc_pools_multiple_files_per_pair_weighted_by_n(
    tmp_path: Path, bfcl_results_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import quantmcp.report.cross_bench as cb

    monkeypatch.setattr(cb, "_MODEL_FAMILY_MARKERS", ["FamilyA", "FamilyB"])

    files = [
        _write_result(tmp_path, "a-fp16-t1.json", "FamilyA-x.gguf", "fp16", 10, 1.0),
        _write_result(tmp_path, "a-fp16-t2.json", "FamilyA-x.gguf", "fp16", 10, 0.8),
        _write_result(tmp_path, "a-q4.json", "FamilyA-x.gguf", "Q4_K_M", 10, 0.5),
        _write_result(tmp_path, "b-fp16.json", "FamilyB-x.gguf", "fp16", 10, 0.2),
        _write_result(tmp_path, "b-q4.json", "FamilyB-x.gguf", "Q4_K_M", 10, 0.3),
    ]

    result = compute_cbc(files, bfcl_results_path)
    a_row = next(r for r in result.table if r["model"] == "FamilyA")
    # pooled fp16 = (10*1.0 + 10*0.8) / 20 = 0.9, so delta_svr_mcp = 0.5 - 0.9 = -0.4
    assert a_row["delta_svr_mcp"] == pytest.approx(-0.4)


def test_compute_cbc_raises_with_fewer_than_two_pairs(
    tmp_path: Path, bfcl_results_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import quantmcp.report.cross_bench as cb

    monkeypatch.setattr(cb, "_MODEL_FAMILY_MARKERS", ["FamilyA", "FamilyB"])

    files = [
        _write_result(tmp_path, "a-fp16.json", "FamilyA-x.gguf", "fp16", 10, 0.9),
        _write_result(tmp_path, "a-q4.json", "FamilyA-x.gguf", "Q4_K_M", 10, 0.5),
    ]

    with pytest.raises(ValueError):
        compute_cbc(files, bfcl_results_path)


def test_compute_cbc_uses_per_family_baseline_quant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Qwen3-1.7B has no fp16 entry on either side (its bf16 weights OOM on
    a 4GB card) -- compute_cbc must use its configured Q8_0 baseline instead
    of silently dropping the family via the hardcoded "fp16" lookup."""
    import quantmcp.report.cross_bench as cb

    monkeypatch.setattr(cb, "_MODEL_FAMILY_MARKERS", ["FamilyA", "FamilyC"])
    monkeypatch.setattr(cb, "_FAMILY_BASELINE_QUANT", {"FamilyC": "Q8_0"})

    bfcl_path = tmp_path / "bfcl.json"
    bfcl_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"model": "FamilyA", "quant": "fp16", "svr_bfcl": 0.8},
                    {"model": "FamilyA", "quant": "Q4_K_M", "svr_bfcl": 0.6},
                    {"model": "FamilyC", "quant": "Q8_0", "svr_bfcl": 0.9},
                    {"model": "FamilyC", "quant": "Q4_K_M", "svr_bfcl": 0.85},
                ]
            }
        )
    )

    files = [
        _write_result(tmp_path, "a-fp16.json", "FamilyA-x.gguf", "fp16", 10, 0.9),
        _write_result(tmp_path, "a-q4.json", "FamilyA-x.gguf", "Q4_K_M", 10, 0.5),
        _write_result(tmp_path, "c-q8.json", "FamilyC-x.gguf", "Q8_0", 10, 0.7),
        _write_result(tmp_path, "c-q4.json", "FamilyC-x.gguf", "Q4_K_M", 10, 0.6),
    ]

    result = compute_cbc(files, bfcl_path)
    c_row = next(r for r in result.table if r["model"] == "FamilyC")
    assert c_row["baseline_quant"] == "Q8_0"
    assert c_row["delta_svr_bfcl"] == pytest.approx(0.85 - 0.9)
    assert c_row["delta_svr_mcp"] == pytest.approx(0.6 - 0.7)


def test_cross_bench_cli_writes_output_json(
    tmp_path: Path, bfcl_results_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import quantmcp.report.cross_bench as cb
    from quantmcp.cli import main

    monkeypatch.setattr(cb, "_MODEL_FAMILY_MARKERS", ["FamilyA", "FamilyB"])

    files = [
        _write_result(tmp_path, "a-fp16.json", "FamilyA-x.gguf", "fp16", 10, 0.9),
        _write_result(tmp_path, "a-q4.json", "FamilyA-x.gguf", "Q4_K_M", 10, 0.5),
        _write_result(tmp_path, "b-fp16.json", "FamilyB-x.gguf", "fp16", 10, 0.2),
        _write_result(tmp_path, "b-q4.json", "FamilyB-x.gguf", "Q4_K_M", 10, 0.3),
    ]
    output_path = tmp_path / "cbc.json"

    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "cross-bench",
            *[str(f) for f in files],
            "--bfcl-results",
            str(bfcl_results_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["n_pairs"] == 2
    assert data["rho"] == pytest.approx(1.0)
    assert len(data["table"]) == 2
