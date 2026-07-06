"""CLI skeleton (spec §6.3), modeled on quant-toolcall-bench's `quantcall` CLI
shape but not vendored verbatim: quantmcp's `run` command launches a sandboxed
MCP server + task fixture instead of sampling a BFCL tier. `sweep`,
`cross-bench`, and the full `leaderboard`/`pareto` wiring are stubs until
Phase 1+ (see spec §10 roadmap) — this mirrors quant-toolcall-bench's own
Phase 0 CLI, which stubs `pareto`/`sweep` the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from quantmcp import __version__
from quantmcp.config import QuantMCPConfig, load_config

_PACKAGE_ROOT = Path(__file__).parent


@click.group()
@click.version_option(version=__version__, prog_name="quantmcp")
def main() -> None:
    """QuantMCP — does quantization survive real MCP tool schemas?"""


def _server_command_for_tier(tier: str) -> tuple[str, list[str], Path | None, Path]:
    """Return (command, args, fixture_dir, default_tasks_file) for a server tier.

    `args` may contain the literal placeholder "{root}", resolved by
    runner.py once a task instance's sandbox directory actually exists.
    """
    if tier == "u0":
        return (
            sys.executable,
            ["-m", "quantmcp.servers.toy"],
            None,
            _PACKAGE_ROOT / "tasks" / "fixtures" / "u0_tasks.yaml",
        )
    if tier == "filesystem":
        from quantmcp.servers.filesystem import ARGS_TEMPLATE, COMMAND

        return (
            COMMAND,
            list(ARGS_TEMPLATE),
            _PACKAGE_ROOT / "tasks" / "fixtures" / "u1_filesystem",
            _PACKAGE_ROOT / "tasks" / "fixtures" / "u1_filesystem_tasks.yaml",
        )
    raise click.ClickException(f"Server tier {tier!r} is not implemented yet (spec §10 roadmap)")


def _build_backend(cfg: QuantMCPConfig) -> Any:
    from quantmcp.backends.mock import MockBackend

    if cfg.backend == "mock":
        return MockBackend(model=cfg.model, latency_ms=cfg.mock.latency_ms)
    if cfg.backend == "llama-cpp":
        from quantmcp.backends.llama_cpp import LlamaCppBackend  # type: ignore[import]

        return LlamaCppBackend(
            model_path=cfg.model,
            n_ctx=cfg.llama_cpp.n_ctx,
            n_gpu_layers=cfg.llama_cpp.n_gpu_layers,
            max_tokens=cfg.llama_cpp.max_tokens,
            temperature=cfg.temperature,
            chat_format=cfg.llama_cpp.chat_format,
            decoding=cfg.decoding,
        )
    if cfg.backend == "transformers":
        from quantmcp.backends.hf import HFBackend  # type: ignore[import]

        return HFBackend(
            model_id=cfg.model,
            device=cfg.hf.device,
            torch_dtype=cfg.hf.torch_dtype,
            max_new_tokens=cfg.hf.max_new_tokens,
            temperature=cfg.temperature,
            load_in_4bit=cfg.hf.load_in_4bit,
            load_in_8bit=cfg.hf.load_in_8bit,
        )
    if cfg.backend == "openai":
        from quantmcp.backends.openai_endpoint import OpenAIEndpointBackend

        return OpenAIEndpointBackend(
            base_url=cfg.openai.base_url,
            model=cfg.model,
            max_tokens=cfg.openai.max_tokens,
            temperature=cfg.temperature,
            timeout_s=cfg.openai.timeout_s,
            api_key_env=cfg.openai.api_key_env,
        )
    if cfg.backend == "vllm":
        from quantmcp.backends.vllm_backend import VLLMBackend

        return VLLMBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.vllm.max_new_tokens,
            temperature=cfg.temperature,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")


@main.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
@click.option("--manifest", "manifest_path", default=None, type=click.Path())
@click.option("--tasks-file", "tasks_file_override", default=None, type=click.Path(exists=True))
def run_cmd(
    config_path: str,
    output_path: str | None,
    manifest_path: str | None,
    tasks_file_override: str | None,
) -> None:
    """Run the MCP tool-calling benchmark against one server tier + backend."""
    import uuid

    from quantmcp.manifest import write_manifest
    from quantmcp.runner import run_eval, write_result
    from quantmcp.tasks.loader import load_tasks

    cfg = load_config(config_path)
    command, args, fixture_dir, default_tasks_file = _server_command_for_tier(cfg.server.tier)
    tasks_file = Path(tasks_file_override) if tasks_file_override else default_tasks_file
    tasks = load_tasks(tasks_file)

    if not tasks:
        click.echo(f"No tasks loaded from {tasks_file}", err=True)
        sys.exit(1)

    backend = _build_backend(cfg)
    run_id = uuid.uuid4().hex

    click.echo(
        f"Running {len(tasks)} task(s) | server={cfg.server.tier} backend={cfg.backend} "
        f"model={cfg.model} quant={cfg.quant}"
    )
    result = run_eval(
        cfg,
        tasks,
        backend,
        command,
        args,
        run_id,
        fixture_dir=fixture_dir,
        config_path=config_path,
    )

    out = output_path or "result.json"
    write_result(result, out)
    click.echo(f"Result written to {out}")
    click.echo(f"  SVR-MCP={result.metrics.svr_mcp:.3f}  TSR={result.metrics.tsr:.3f}")

    if manifest_path:
        write_manifest(result.manifest, manifest_path)
        click.echo(f"Manifest written to {manifest_path}")


@main.command("sweep")
@click.option("--model", required=True)
@click.option("--quants", required=True, help="Comma-separated quant levels")
@click.option("--servers", default="u0", help="Comma-separated server tiers")
def sweep_cmd(model: str, quants: str, servers: str) -> None:
    """Sweep model x quant x server combinations."""
    click.echo(f"[sweep stub] model={model} quants={quants} servers={servers}")
    click.echo("Full sweep implementation lands in Phase 1 (spec §10).")


@main.command("compare")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
@click.option("--output", "output_path", default=None, type=click.Path())
def compare_cmd(result_files: tuple[str, ...], fmt: str, output_path: str | None) -> None:
    """Compare multiple result.json files and show SVR-MCP/TSR side by side."""
    results: list[dict[str, Any]] = [json.loads(Path(p).read_text()) for p in result_files]

    if fmt == "json":
        text = json.dumps(results, indent=2)
    else:
        lines = [f"{'Config':<40}  {'SVR-MCP':>8}  {'TSR':>8}"]
        lines.append("-" * 60)
        for r in results:
            cfg = r.get("config", {})
            label = f"{cfg.get('model', '?')} {cfg.get('quant', '?')} {cfg.get('server_tier', '?')}"
            lines.append(f"{label:<40}  {r.get('svr_mcp', 0):.3f}  {r.get('tsr', 0):.3f}")
        text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(text + "\n")
        click.echo(f"Written to {output_path}")
    else:
        click.echo(text)


@main.command("cross-bench")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--bfcl-results", required=True, type=click.Path(exists=True))
def cross_bench_cmd(result_files: tuple[str, ...], bfcl_results: str) -> None:
    """Compute Cross-Benchmark Consistency (CBC, spec §4.5) vs QuantCall results."""
    click.echo("[cross-bench stub] Implementation lands in Phase 2 (spec §10).")


@main.command("leaderboard")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output-dir", default="leaderboard", show_default=True, type=click.Path())
def leaderboard_cmd(results_dir: str, output_dir: str) -> None:
    """Build runs.csv + leaderboard.{json,csv,md} from a directory of results."""
    click.echo("[leaderboard stub] Extended report layer lands in Phase 3 (spec §10).")


@main.command("validate-config")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def validate_config_cmd(config_path: str) -> None:
    """Validate a QuantMCP YAML config file."""
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Config: {config_path}")
    click.echo(f"  backend: {cfg.backend}")
    click.echo(f"  model  : {cfg.model}")
    click.echo(f"  quant  : {cfg.quant}")
    click.echo(f"  server : {cfg.server.tier}")
    click.echo("OK")
