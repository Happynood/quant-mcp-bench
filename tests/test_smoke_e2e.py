from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

from quantmcp.backends.mock import MockBackend
from quantmcp.config import load_config
from quantmcp.manifest import write_manifest
from quantmcp.runner import run_eval, write_result
from quantmcp.tasks.loader import load_tasks

SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "smoke.yaml"
U0_TASKS = (
    Path(__file__).parent.parent / "src" / "quantmcp" / "tasks" / "fixtures" / "u0_tasks.yaml"
)
TOY_COMMAND = sys.executable
TOY_ARGS = ["-m", "quantmcp.servers.toy"]


def test_smoke_e2e_writes_result_json():
    cfg = load_config(SMOKE_CONFIG)
    tasks = load_tasks(U0_TASKS)
    backend = MockBackend(latency_ms=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        manifest_path = Path(tmpdir) / "manifest.json"

        result = run_eval(
            cfg, tasks, backend, TOY_COMMAND, TOY_ARGS, uuid.uuid4().hex, config_path=SMOKE_CONFIG
        )
        write_result(result, result_path)
        write_manifest(result.manifest, manifest_path)

        assert result_path.exists()
        assert manifest_path.exists()

        data = json.loads(result_path.read_text())
        manifest_data = json.loads(manifest_path.read_text())

        assert "svr_mcp" in data
        assert "tsr" in data
        assert "n" in data
        assert data["n"] == len(tasks)
        assert "manifest" in data
        assert "config" in data

        assert "timestamp" in manifest_data
        assert "model" in manifest_data
        assert "backend" in manifest_data
        assert "config_sha256" in manifest_data
        assert "fixture_sha256" in manifest_data
        assert "server_tier" in manifest_data


def test_smoke_e2e_metrics_in_range():
    cfg = load_config(SMOKE_CONFIG)
    tasks = load_tasks(U0_TASKS)
    backend = MockBackend(latency_ms=0)
    result = run_eval(cfg, tasks, backend, TOY_COMMAND, TOY_ARGS, uuid.uuid4().hex)

    assert 0.0 <= result.metrics.svr_mcp <= 1.0
    assert 0.0 <= result.metrics.tsr <= 1.0


def test_smoke_e2e_all_instances_evaluated():
    cfg = load_config(SMOKE_CONFIG)
    tasks = load_tasks(U0_TASKS)
    backend = MockBackend(latency_ms=0)
    result = run_eval(cfg, tasks, backend, TOY_COMMAND, TOY_ARGS, uuid.uuid4().hex)
    assert result.metrics.n == len(tasks) == 2


def test_smoke_e2e_cli(smoke_config_path, tmp_path):
    from click.testing import CliRunner

    from quantmcp.cli import main

    runner = CliRunner()
    out = tmp_path / "result.json"
    mf = tmp_path / "manifest.json"
    r = runner.invoke(
        main,
        [
            "run",
            "--config",
            str(smoke_config_path),
            "--output",
            str(out),
            "--manifest",
            str(mf),
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    assert mf.exists()
    data = json.loads(out.read_text())
    assert "svr_mcp" in data


def test_dump_schemas_cli(tmp_path):
    from click.testing import CliRunner

    from quantmcp.cli import main

    runner = CliRunner()
    out = tmp_path / "schemas.json"
    r = runner.invoke(main, ["dump-schemas", "--tiers", "u0", "--output", str(out)])

    assert r.exit_code == 0, r.output
    assert out.exists()
    schemas = json.loads(out.read_text())
    assert {s["name"] for s in schemas} == {"add", "write_note"}
    assert all(s["tier"] == "u0" for s in schemas)
