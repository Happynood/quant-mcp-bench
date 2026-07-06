# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json
import re

from quantmcp.parsing.base import CallParser, ParsedCall
from quantmcp.parsing.raw_json import RawJsonParser

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_PYTHON_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", re.DOTALL)

_RESERVED = frozenset(
    [
        "if",
        "for",
        "while",
        "def",
        "class",
        "return",
        "import",
        "from",
        "with",
        "print",
        "assert",
        "raise",
        "lambda",
        "yield",
    ]
)


def _parse_python_kwargs(args_str: str) -> dict[str, object]:
    """Parse key=value pairs from a Python-style function call argument string."""
    result: dict[str, object] = {}
    for m in re.finditer(r'([a-zA-Z_]\w*)\s*=\s*(["\']?)([^,]*?)\2(?:,|$)', args_str):
        key = m.group(1)
        raw_val = m.group(3).strip()
        try:
            result[key] = json.loads(raw_val)
        except json.JSONDecodeError:
            result[key] = raw_val
    return result


class GGUFTemplateParser(CallParser):
    """Parser for llama.cpp / chat-template tool-call output formats.

    Handles:
    - <tool_call>...</tool_call> XML-like tags
    - Python-style function calls: func_name(arg=value, ...)
    - Falls back to RawJsonParser
    """

    def __init__(self) -> None:
        self._fallback = RawJsonParser()

    @property
    def name(self) -> str:
        return "gguf_template"

    def parse(self, raw_output: str) -> list[ParsedCall]:
        calls: list[ParsedCall] = []

        for m in _TOOL_CALL_TAG_RE.finditer(raw_output):
            content = m.group(1).strip()
            try:
                obj = json.loads(content)
                fn_name = obj.get("name") or obj.get("function")
                if isinstance(fn_name, str):
                    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
                    calls.append(ParsedCall(name=fn_name, arguments=args))
                    continue
            except json.JSONDecodeError:
                pass
            sub = self._fallback.parse(content)
            calls.extend(sub)

        if calls:
            return calls

        fallback = self._fallback.parse(raw_output)
        if fallback:
            return fallback

        for m in _PYTHON_CALL_RE.finditer(raw_output):
            fn_name = m.group(1)
            if fn_name in _RESERVED:
                continue
            args_str = m.group(2)
            args = _parse_python_kwargs(args_str)
            calls.append(ParsedCall(name=fn_name, arguments=args))
            break

        return calls
