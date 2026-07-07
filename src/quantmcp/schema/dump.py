"""Dump a server tier's live tool schemas — the source of truth for both the
Schema Complexity Index (H2, spec §4.3) and the frozen-schema artifact
published in the `quantmcp-suite` HF dataset (spec §10 Phase 4). New, not
vendored — quant-toolcall-bench has no notion of a live MCP server to
introspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.servers.base import MCPServerHandle


async def dump_tier_schemas(
    tier: str,
    command: str,
    args: list[str],
    fixture_dir: Path | None,
    run_id: str = "schema-dump",
) -> list[dict[str, Any]]:
    """Launch the given tier's real server in a sandboxed fixture instance
    and return one entry per tool: tier, name, description, input_schema."""
    with sandbox_instance(fixture_dir, run_id) as instance_root:
        resolved_args = [a.replace("{root}", str(instance_root)) for a in args]
        env = {
            "QUANTMCP_U0_ROOT": str(instance_root),
            "QUANTMCP_U3_ROOT": str(instance_root),
        }
        async with MCPServerHandle(command, resolved_args, env=env, cwd=instance_root) as handle:
            tools = await handle.list_tools()
            return [
                {
                    "tier": tier,
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in tools
            ]
