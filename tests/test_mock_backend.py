from __future__ import annotations

import json

from quantmcp.backends.base import ToolCallResult
from quantmcp.backends.mock import MockBackend


def _sample_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                },
            },
        }
    ]


def test_mock_backend_returns_tool_call_result():
    backend = MockBackend(model="mock", latency_ms=0)
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    result = backend.generate_toolcall(messages, _sample_tools())
    assert isinstance(result, ToolCallResult)
    assert result.raw_output
    assert result.latency_ms >= 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0


def test_mock_backend_raw_output_is_parseable_json():
    backend = MockBackend(model="mock", latency_ms=0)
    messages = [{"role": "user", "content": "Translate hello to Spanish"}]
    tools = [
        {
            "function": {
                "name": "translate_text",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            }
        }
    ]
    result = backend.generate_toolcall(messages, tools)
    obj = json.loads(result.raw_output)
    assert "tool_call" in obj or "name" in obj


def test_mock_backend_no_call_when_emit_no_call():
    backend = MockBackend(model="mock", latency_ms=0, emit_no_call=True)
    messages = [{"role": "user", "content": "Tell me a joke"}]
    result = backend.generate_toolcall(messages, [])
    assert "tool_call" not in result.raw_output.lower() or len(result.raw_output) < 30


def test_mock_backend_name():
    backend = MockBackend()
    assert backend.name == "mock"


def test_mock_backend_enum_value_used():
    backend = MockBackend(latency_ms=0)
    tools = [
        {
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["fast", "thorough"]}},
                },
            }
        }
    ]
    result = backend.generate_toolcall([{"role": "user", "content": "Search"}], tools)
    obj = json.loads(result.raw_output)
    tc = obj.get("tool_call", {})
    args = tc.get("arguments", {})
    if "mode" in args:
        assert args["mode"] in ("fast", "thorough")
