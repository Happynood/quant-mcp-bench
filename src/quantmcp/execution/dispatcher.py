"""Executes parsed calls against a live, sandboxed MCPServerHandle (spec §6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantmcp.parsing.base import ParsedCall
from quantmcp.servers.base import MCPServerHandle


@dataclass(frozen=True)
class ExecutionResult:
    call: ParsedCall
    ok: bool
    error: str | None = None
    raw_result: Any = None


def _extract_error_text(raw: Any) -> str:
    content = getattr(raw, "content", None) or []
    texts = [text for c in content if (text := getattr(c, "text", None))]
    return "; ".join(texts) or "tool call reported isError=True"


async def execute_calls(
    handle: MCPServerHandle,
    calls: list[ParsedCall],
) -> list[ExecutionResult]:
    """Execute every call in order against `handle`, never raising: a failed
    call becomes an ExecutionResult(ok=False), so one bad call doesn't abort
    the rest of the instance's structurally-valid calls.
    """
    results: list[ExecutionResult] = []
    for call in calls:
        try:
            raw = await handle.call_tool(call.name, call.arguments)
            is_error = bool(getattr(raw, "isError", False))
            results.append(
                ExecutionResult(
                    call=call,
                    ok=not is_error,
                    error=_extract_error_text(raw) if is_error else None,
                    raw_result=raw,
                )
            )
        except Exception as exc:
            results.append(ExecutionResult(call=call, ok=False, error=str(exc)))
    return results
