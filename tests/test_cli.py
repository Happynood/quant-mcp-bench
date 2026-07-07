from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from quantmcp.cli import _build_backend, main
from quantmcp.config import QuantMCPConfig


def _write_result_file(
    path: Path, model: str, quant: str, tier: str, svr_mcp: float, tsr: float
) -> None:
    path.write_text(
        json.dumps(
            {
                "n": 10,
                "svr_mcp": svr_mcp,
                "tsr": tsr,
                "config": {"model": model, "quant": quant, "server_tier": tier},
            }
        )
    )


def test_compare_cmd_table_format(tmp_path: Path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    _write_result_file(f1, "modelA", "fp16", "filesystem", 0.9, 0.8)
    _write_result_file(f2, "modelA", "Q4_K_M", "filesystem", 0.6, 0.5)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(f1), str(f2)])

    assert result.exit_code == 0, result.output
    assert "modelA fp16 filesystem" in result.output
    assert "0.900" in result.output
    assert "0.600" in result.output


def test_compare_cmd_json_format_and_output_file(tmp_path: Path):
    f1 = tmp_path / "a.json"
    _write_result_file(f1, "modelA", "fp16", "filesystem", 0.9, 0.8)
    out = tmp_path / "compare.json"

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(f1), "--format", "json", "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data[0]["svr_mcp"] == pytest.approx(0.9)


def test_validate_config_cmd_valid_config(tmp_path: Path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: mock\nmodel: mock\nquant: fp16\n")

    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "backend: mock" in result.output


def test_validate_config_cmd_invalid_config_exits_nonzero(tmp_path: Path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: not-a-real-backend\n")

    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "OK" not in result.output


def test_leaderboard_cmd_writes_output_dir(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result_file(results_dir / "fp16.result.json", "modelA", "fp16", "sqlite", 0.5, 0.4)
    output_dir = tmp_path / "leaderboard"

    runner = CliRunner()
    result = runner.invoke(main, ["leaderboard", str(results_dir), "--output-dir", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert "1 run(s) across 1 tier(s)" in result.output
    assert (output_dir / "mcp_leaderboard.json").exists()


def test_sweep_cmd_prints_not_implemented_pointer():
    runner = CliRunner()
    result = runner.invoke(
        main, ["sweep", "--model", "m", "--quants", "fp16,Q4_K_M", "--servers", "u0"]
    )

    assert result.exit_code == 0, result.output
    assert "not yet implemented" in result.output
    assert "RUN_REAL.md" in result.output


def test_run_cmd_exits_nonzero_when_no_tasks_loaded(tmp_path: Path, smoke_config_path: Path):
    empty_tasks = tmp_path / "empty_tasks.yaml"
    empty_tasks.write_text("[]\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--config", str(smoke_config_path), "--tasks-file", str(empty_tasks)],
    )

    assert result.exit_code == 1
    assert "No tasks loaded" in result.output


def test_build_backend_mock():
    from quantmcp.backends.mock import MockBackend

    cfg = QuantMCPConfig(backend="mock", model="mock")
    backend = _build_backend(cfg)
    assert isinstance(backend, MockBackend)


def test_build_backend_llama_cpp_forwards_verbose_and_n_threads(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _FakeLlamaCppBackend:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("quantmcp.backends.llama_cpp.LlamaCppBackend", _FakeLlamaCppBackend)

    cfg = QuantMCPConfig(backend="llama-cpp", model="~/models/x.gguf")
    cfg.llama_cpp.verbose = True
    cfg.llama_cpp.n_threads = 4

    backend = _build_backend(cfg)

    assert isinstance(backend, _FakeLlamaCppBackend)
    assert captured["verbose"] is True
    assert captured["n_threads"] == 4
    assert captured["chat_format"] == cfg.llama_cpp.chat_format
    assert captured["decoding"] == cfg.decoding


def test_build_backend_unknown_raises():
    cfg = QuantMCPConfig(backend="mock", model="mock")
    # bypasses Literal validation (no validate_assignment) to exercise the
    # defensive branch that would otherwise never be reachable through config
    # loading, since Pydantic rejects an unknown backend before this runs
    cast(Any, cfg).backend = "not-a-backend"
    with pytest.raises(ValueError, match="Unknown backend"):
        _build_backend(cfg)
