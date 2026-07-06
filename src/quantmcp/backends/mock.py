# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json
import time
from typing import Any

from quantmcp.backends.base import Backend, ToolCallResult


def _mock_value(schema: dict[str, Any]) -> Any:
    """Generate a minimal valid value for a JSON Schema type."""
    t = schema.get("type", "string")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "string":
        return "mock_value"
    if t in ("number", "float"):
        return 0.0
    if t == "integer":
        return 0
    if t == "boolean":
        return True
    if t == "array":
        return []
    if t == "object":
        props = schema.get("properties", {})
        return {k: _mock_value(v) for k, v in props.items()}
    return "mock_value"


def _mock_arguments(schema: dict[str, Any]) -> dict[str, Any]:
    """Generate mock arguments matching a JSON Schema."""
    if not schema:
        return {}
    props = schema.get("properties", {})
    return {k: _mock_value(v) for k, v in props.items()}


class MockBackend(Backend):
    """Deterministic mock backend for CI and plumbing validation."""

    def __init__(
        self,
        model: str = "mock",
        latency_ms: float = 5.0,
        seed: int | None = None,
        emit_no_call: bool = False,
    ) -> None:
        self._model = model
        self._latency_s = latency_ms / 1000.0
        self._emit_no_call = emit_no_call

    @property
    def name(self) -> str:
        return "mock"

    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        start = time.perf_counter()
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if self._emit_no_call or not tools:
            raw_output = "I don't need to call any tool to answer this."
        else:
            tool = tools[0]
            fn = tool.get("function", tool)
            name = fn.get("name", "unknown")
            params = fn.get("parameters", fn.get("json_schema", {}))
            args = _mock_arguments(params)
            raw_output = json.dumps(
                {"tool_call": {"name": name, "arguments": args}},
                ensure_ascii=False,
            )

        input_tokens = sum(len(m.get("content", "").split()) for m in messages)
        output_tokens = len(raw_output.split())
        return ToolCallResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            tokens_per_second=output_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
        )
