# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from typing import Any

import jsonschema

from quantmcp.parsing.base import ParsedCall


def validate_call(call: ParsedCall, json_schema: dict[str, Any]) -> bool:
    """Return True if call.arguments satisfies the given JSON Schema."""
    try:
        jsonschema.validate(instance=call.arguments, schema=json_schema)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        return False


def validate_calls(
    calls: list[ParsedCall],
    tool_schemas: dict[str, dict[str, Any]],
) -> bool:
    """Return True if all calls have a known tool name AND valid arguments.

    tool_schemas: mapping from tool name → json_schema
    """
    for call in calls:
        schema = tool_schemas.get(call.name)
        if schema is None:
            return False
        if not validate_call(call, schema):
            return False
    return True
