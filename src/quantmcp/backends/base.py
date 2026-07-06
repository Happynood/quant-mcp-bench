# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallResult:
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    peak_vram_mb: float | None = None
    tokens_per_second: float | None = None


class Backend(ABC):
    @abstractmethod
    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


def tools_to_openai_spec(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP `tools/list` results to OpenAI-compatible tool spec format.

    Diff from quant-toolcall-bench: the upstream ToolSpec dataclass (BFCL-shaped)
    doesn't exist here. MCP's `mcp.types.Tool` has `.name`/`.description`/
    `.inputSchema` instead of `.json_schema` — duck-type on that shape and fall
    back to passing dicts through unchanged.
    """
    result: list[dict[str, Any]] = []
    for t in tools:
        schema = getattr(t, "inputSchema", None)
        if schema is not None:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": schema,
                    },
                }
            )
        else:
            result.append(t)
    return result
