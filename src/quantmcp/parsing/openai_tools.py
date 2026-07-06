# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json

from quantmcp.parsing.base import CallParser, ParsedCall


class OpenAIToolsParser(CallParser):
    """Parse the OpenAI tool_calls response field from a JSON completion object."""

    @property
    def name(self) -> str:
        return "openai_tools"

    def parse(self, raw_output: str) -> list[ParsedCall]:
        raw_output = raw_output.strip()
        if not raw_output:
            return []
        try:
            obj = json.loads(raw_output)
        except json.JSONDecodeError:
            return []

        choices = obj.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return []

        calls: list[ParsedCall] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            if not fn_name:
                continue
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            calls.append(ParsedCall(name=fn_name, arguments=args if isinstance(args, dict) else {}))

        return calls
