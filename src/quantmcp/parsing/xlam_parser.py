# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json

from quantmcp.parsing.base import CallParser, ParsedCall
from quantmcp.parsing.raw_json import RawJsonParser


class XlamParser(CallParser):
    """Parser for xLAM-style JSON array or wrapped tool call output."""

    def __init__(self) -> None:
        self._fallback = RawJsonParser()

    @property
    def name(self) -> str:
        return "xlam"

    def parse(self, raw_output: str) -> list[ParsedCall]:
        raw_output = raw_output.strip()
        if not raw_output:
            return []

        try:
            obj = json.loads(raw_output)
        except json.JSONDecodeError:
            return self._fallback.parse(raw_output)

        if isinstance(obj, list):
            return self._extract_from_list(obj)

        if isinstance(obj, dict):
            tool_calls = obj.get("tool_calls") or obj.get("tools") or []
            if isinstance(tool_calls, list):
                return self._extract_from_list(tool_calls)
            return self._fallback.parse(raw_output)

        return []

    def _extract_from_list(self, items: list[object]) -> list[ParsedCall]:
        calls: list[ParsedCall] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if not isinstance(name, str) or not name:
                continue
            args = item.get("arguments", item.get("args", item.get("parameters", {})))
            calls.append(ParsedCall(name=name, arguments=args if isinstance(args, dict) else {}))
        return calls
