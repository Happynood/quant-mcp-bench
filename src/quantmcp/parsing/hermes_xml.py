# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json
import re

from quantmcp.parsing.base import CallParser, ParsedCall

_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


class HermesXmlParser(CallParser):
    """Parser for Hermes-style <tool_call>...</tool_call> XML output."""

    @property
    def name(self) -> str:
        return "hermes_xml"

    def parse(self, raw_output: str) -> list[ParsedCall]:
        calls: list[ParsedCall] = []
        for m in _TAG_RE.finditer(raw_output):
            content = m.group(1).strip()
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                continue
            fn_name = obj.get("name") or obj.get("function")
            if not isinstance(fn_name, str):
                continue
            args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
            calls.append(ParsedCall(name=fn_name, arguments=args if isinstance(args, dict) else {}))
        return calls
