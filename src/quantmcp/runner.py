"""Single-run pipeline orchestration (spec §6.4): launch sandboxed server ->
discover tools -> build prompt -> infer -> parse -> validate (SVR-MCP) ->
execute (TSR) -> teardown -> compute metrics + manifest.

Not vendored: quant-toolcall-bench's runner.py is BFCL-tier-shaped (T0-T6,
FCR/TSA/AC weights) and has no notion of a live MCP server or sandbox, so
this is genuinely new orchestration code built on top of the vendored
Backend/parsing/validation layers.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quantmcp.backends.base import Backend, tools_to_openai_spec
from quantmcp.config import QuantMCPConfig
from quantmcp.execution.dispatcher import execute_calls
from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.manifest import RunManifest, collect_manifest, compute_fixture_sha256
from quantmcp.metrics.core import (
    InstanceOutcome,
    MetricsResult,
    compute_metrics,
    evaluate_svr_mcp,
    evaluate_tsr,
)
from quantmcp.parsing.base import CallParser
from quantmcp.parsing.hermes_xml import HermesXmlParser
from quantmcp.parsing.raw_json import RawJsonParser
from quantmcp.servers.base import MCPServerHandle
from quantmcp.tasks.base import MCPTaskInstance, SandboxState


@dataclass
class RunResult:
    config: dict[str, Any]
    metrics: MetricsResult
    manifest: RunManifest
    total_latency_ms: float
    peak_vram_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.metrics.n,
            "svr_mcp": self.metrics.svr_mcp,
            "tsr": self.metrics.tsr,
            # "svr" alias kept for report/leaderboard.py (vendored, BFCL-shaped
            # column reader) until it's extended for MCP metrics in Phase 3.
            "svr": self.metrics.svr_mcp,
            "total_latency_ms": self.total_latency_ms,
            "vram_gb": (self.peak_vram_mb / 1024.0) if self.peak_vram_mb is not None else None,
            "config": self.config,
            "manifest": asdict(self.manifest),
        }


def _get_parser(chat_variant: str) -> CallParser:
    if chat_variant == "qwen3_nothink":
        return HermesXmlParser()
    return RawJsonParser()


async def _run_one_instance(
    task: MCPTaskInstance,
    cfg: QuantMCPConfig,
    backend: Backend,
    parser: CallParser,
    server_command: str,
    server_args: list[str],
    server_env: dict[str, str] | None,
    run_id: str,
    fixture_dir: Path | None,
) -> tuple[InstanceOutcome, float, float | None, str | None]:
    with sandbox_instance(
        fixture_dir, run_id, keep_on_failure=cfg.sandbox.keep_on_failure
    ) as instance_root:
        env = dict(server_env or {})
        env.setdefault("QUANTMCP_U0_ROOT", str(instance_root))
        # Some reference servers (e.g. filesystem) take their allowed root as a
        # positional CLI arg rather than an env var/cwd, and that root is only
        # known once the sandbox instance exists — resolve the "{root}"
        # placeholder here rather than in the static per-tier command table.
        resolved_args = [a.replace("{root}", str(instance_root)) for a in server_args]
        async with MCPServerHandle(
            server_command, resolved_args, env=env, cwd=instance_root
        ) as handle:
            tools = await handle.list_tools()
            tool_schemas = {t.name: t.inputSchema for t in tools}
            openai_tools = tools_to_openai_spec(tools)
            # Task instructions may reference "{root}" so a task can tell the
            # model the absolute path it's allowed to operate in, mirroring how
            # a real client would surface the server's allowed directories.
            instruction = task.instruction.format(root=str(instance_root))
            if cfg.chat_variant == "qwen3_nothink":
                instruction = f"{instruction} /no_think"
            messages = [{"role": "user", "content": instruction}]

            latency_ms = 0.0
            peak_vram_mb: float | None = None
            try:
                result = backend.generate_toolcall(messages, openai_tools)
                latency_ms = result.latency_ms
                peak_vram_mb = result.peak_vram_mb
                parsed_calls = parser.parse(result.raw_output)
                parse_succeeded = True
            except Exception:
                parsed_calls = []
                parse_succeeded = False

            svr_pass = evaluate_svr_mcp(parsed_calls, parse_succeeded, tool_schemas)
            exec_results = await execute_calls(handle, parsed_calls) if svr_pass else []
            sandbox_state = SandboxState(root=instance_root, results=exec_results)
            tsr_pass = evaluate_tsr(task, sandbox_state) if svr_pass else False

            outcome = InstanceOutcome(
                task_id=task.id,
                parse_succeeded=parse_succeeded,
                svr_pass=svr_pass,
                tsr_pass=tsr_pass,
            )
            return outcome, latency_ms, peak_vram_mb, handle.server_version


async def run_eval_async(
    cfg: QuantMCPConfig,
    tasks: list[MCPTaskInstance],
    backend: Backend,
    server_command: str,
    server_args: list[str],
    run_id: str,
    server_env: dict[str, str] | None = None,
    fixture_dir: Path | None = None,
    config_path: str | Path = "",
) -> RunResult:
    parser = _get_parser(cfg.chat_variant)
    outcomes: list[InstanceOutcome] = []
    total_latency_ms = 0.0
    peak_vram_mb: float | None = None
    mcp_server_version: str | None = None

    for task in tasks:
        outcome, latency_ms, vram_mb, server_version = await _run_one_instance(
            task, cfg, backend, parser, server_command, server_args, server_env, run_id, fixture_dir
        )
        outcomes.append(outcome)
        total_latency_ms += latency_ms
        if vram_mb is not None:
            peak_vram_mb = max(peak_vram_mb or 0.0, vram_mb)
        mcp_server_version = mcp_server_version or server_version

    metrics = compute_metrics(outcomes)

    fixture_files = (
        [p for p in fixture_dir.rglob("*") if p.is_file()]
        if fixture_dir is not None and fixture_dir.exists()
        else []
    )
    fixture_sha = compute_fixture_sha256(fixture_files)
    manifest = collect_manifest(
        config_path, cfg, fixture_sha256=fixture_sha, mcp_server_version=mcp_server_version
    )

    config_dict = {
        "model": cfg.model,
        "backend": cfg.backend,
        "quant": cfg.quant,
        "decoding": cfg.decoding,
        "chat_variant": cfg.chat_variant,
        "server_tier": cfg.server.tier,
        "sample_size": cfg.sample_size,
        "seed": cfg.seed,
        "temperature": cfg.temperature,
    }

    return RunResult(
        config=config_dict,
        metrics=metrics,
        manifest=manifest,
        total_latency_ms=total_latency_ms,
        peak_vram_mb=peak_vram_mb,
    )


def run_eval(
    cfg: QuantMCPConfig,
    tasks: list[MCPTaskInstance],
    backend: Backend,
    server_command: str,
    server_args: list[str],
    run_id: str,
    server_env: dict[str, str] | None = None,
    fixture_dir: Path | None = None,
    config_path: str | Path = "",
) -> RunResult:
    return asyncio.run(
        run_eval_async(
            cfg,
            tasks,
            backend,
            server_command,
            server_args,
            run_id,
            server_env=server_env,
            fixture_dir=fixture_dir,
            config_path=config_path,
        )
    )


def write_result(result: RunResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
