# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json
import re
from typing import Any

from quantmcp.parsing.base import CallParser, ParsedCall

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract all JSON objects from a string, trying multiple strategies."""
    candidates: list[str] = []

    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1).strip())

    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass
    return results


def _obj_to_parsed_call(obj: dict[str, Any]) -> ParsedCall | None:
    """Convert a raw JSON object to a ParsedCall if it looks like a tool call."""
    if "tool_call" in obj and isinstance(obj["tool_call"], dict):
        tc = obj["tool_call"]
        name = tc.get("name") or tc.get("function") or tc.get("tool")
        if isinstance(name, str):
            args = tc.get("arguments") or tc.get("args") or tc.get("parameters") or {}
            return ParsedCall(name=name, arguments=args if isinstance(args, dict) else {})

    if "name" in obj and isinstance(obj.get("name"), str):
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        if isinstance(args, dict):
            return ParsedCall(name=obj["name"], arguments=args)

    if "function" in obj and isinstance(obj["function"], str):
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        return ParsedCall(name=obj["function"], arguments=args if isinstance(args, dict) else {})

    return None


class RawJsonParser(CallParser):
    """Fallback parser: extract JSON objects from free-form text and interpret as tool calls."""

    @property
    def name(self) -> str:
        return "raw_json"

    def parse(self, raw_output: str) -> list[ParsedCall]:
        objects = _extract_json_objects(raw_output)
        calls: list[ParsedCall] = []
        for obj in objects:
            pc = _obj_to_parsed_call(obj)
            if pc is not None:
                calls.append(pc)
        return calls
