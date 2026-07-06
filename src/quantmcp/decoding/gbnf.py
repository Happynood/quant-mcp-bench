# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from typing import Any

_PREAMBLE = """\
# JSON grammar for constrained decoding
ws ::= [ \\t\\n]*
true ::= "true"
false ::= "false"
null ::= "null"
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
integer ::= "-"? ([0-9] | [1-9] [0-9]*)
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
value ::= string | number | object-any | array | true | false | null
object-any ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
"""


def _schema_rule_name(path: str) -> str:
    """Turn a schema path into a GBNF rule name using a single separator char.

    llama.cpp's GBNF parser segfaults on rule names that mix "_" and "-"
    (e.g. "product_list-array" crashes; "product-list-array" does not --
    reproduced directly against llama-cpp-python). Property names from JSON
    Schema commonly contain underscores (e.g. "product_list"), and this
    function's own path-joining used "-", so every nested/array property
    name produced a mixed-separator rule name. Normalizing everything to
    "-" avoids the crash.
    """
    name = path.replace(".", "-").replace("[", "").replace("]", "").replace("/", "-")
    name = name.replace("_", "-")
    return name or "root"


def _compile_type(schema: dict[str, Any], path: str, rules: dict[str, str]) -> str:
    """Recursively compile a JSON Schema node into GBNF rule names. Returns the rule name."""
    schema_type = schema.get("type", "any")
    enum_vals = schema.get("enum")

    if enum_vals is not None:
        rule_name = _schema_rule_name(path)
        choices = " | ".join(f'"{v}"' for v in enum_vals)
        rules[rule_name] = f"{choices}"
        return rule_name

    if schema_type == "string":
        return "string"
    if schema_type in ("number", "float"):
        return "number"
    if schema_type == "integer":
        return "integer"
    if schema_type == "boolean":
        return '( "true" | "false" )'
    if schema_type == "null":
        return "null"
    if schema_type == "array":
        items = schema.get("items", {})
        item_rule = _compile_type(items, path + ".item", rules)
        rule_name = _schema_rule_name(path) + "-array"
        rules[rule_name] = f'"[" ws ({item_rule} (ws "," ws {item_rule})*)? ws "]"'
        return rule_name
    if schema_type == "object" or "properties" in schema:
        return _compile_object(schema, path, rules)

    return "value"


def _compile_object(schema: dict[str, Any], path: str, rules: dict[str, str]) -> str:
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    rule_name = _schema_rule_name(path)
    if not properties:
        rules[rule_name] = '"{" ws "}"'
        return rule_name

    prop_rules: list[tuple[bool, str]] = []
    for prop_name, prop_schema in properties.items():
        child_path = f"{path}.{prop_name}"
        value_rule = _compile_type(prop_schema, child_path, rules)
        pair = f'"\\"" "{prop_name}" "\\"" ws ":" ws {value_rule}'
        prop_rules.append((prop_name in required, pair))

    required_parts = [pair for is_req, pair in prop_rules if is_req]
    optional_parts = [pair for is_req, pair in prop_rules if not is_req]

    if required_parts and optional_parts:
        required_seq = ' ws "," ws '.join(required_parts)
        optional_seq = " ".join(f'(ws "," ws {p})?' for p in optional_parts)
        body = f"{required_seq} {optional_seq}"
    elif required_parts:
        body = ' ws "," ws '.join(required_parts)
    else:
        all_parts = [pair for _, pair in prop_rules]
        body = (
            f"({all_parts[0]}" + "".join(f' (ws "," ws {p})?' for p in all_parts[1:]) + ")?"
            if all_parts
            else ""
        )

    rules[rule_name] = '"{"' + f" ws {body} ws " + '"}"'
    return rule_name


def build_tool_call_grammar(tools: list[Any], allow_no_call: bool = True) -> str:
    """Build a GBNF grammar for Hermes-style tool-call output.

    The grammar's root is a union: ``tool-call-path | no-call`` (when
    ``allow_no_call`` is True, the default). ``tool-call-path`` is a single
    ``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` block whose
    ``name`` is one of the given tools and whose ``arguments`` conform
    exactly to that tool's JSON Schema (required/optional properties,
    types, enums -- reusing the same schema compiler as gbnf_from_schema).
    ``no-call`` matches arbitrary free text, so the model can correctly
    abstain (BFCL T6 irrelevance) instead of being forced to always emit a
    tool call. Pass ``allow_no_call=False`` to force a call unconditionally
    (kept only for tests / callers that specifically want the old behavior).
    """
    rules: dict[str, str] = {}
    alternatives: list[str] = []
    for i, tool in enumerate(tools):
        fn = tool.get("function", tool) if isinstance(tool, dict) else tool
        name = getattr(fn, "name", None)
        if name is None and isinstance(fn, dict):
            name = fn.get("name", f"tool{i}")
        schema = getattr(fn, "json_schema", None)
        if schema is None and isinstance(fn, dict):
            schema = fn.get("json_schema") or fn.get("parameters") or {}
        schema = schema or {}
        args_rule = _compile_object(schema, f"tool{i}", rules)
        name_key = '"\\"" "name" "\\""'
        name_val = f'"\\"" "{name}" "\\""'
        args_key = '"\\"" "arguments" "\\""'
        alt = f'{name_key} ws ":" ws {name_val} ws "," ws {args_key} ws ":" ws {args_rule}'
        alternatives.append(alt)

    if not alternatives:
        rules["tool-call-body"] = "value"
    else:
        rules["tool-call-body"] = " | ".join(f"({alt})" for alt in alternatives)

    rules["tool-call-path"] = '"<tool_call>" ws "{" ws tool-call-body ws "}" ws "</tool_call>"'

    if allow_no_call:
        rules["no-call"] = "plain-text"
        rules["plain-text"] = ".*"
        rules["root"] = "tool-call-path | no-call"
    else:
        rules["root"] = "tool-call-path"

    lines = [_PREAMBLE]
    for rule_name, body in rules.items():
        lines.append(f"{rule_name} ::= {body}")
    return "\n".join(lines) + "\n"


def gbnf_from_schema(schema: dict[str, Any]) -> str:
    """Generate a GBNF grammar string constraining output to a given JSON Schema."""
    rules: dict[str, str] = {}
    root_type = _compile_type(schema, "root", rules)

    if "root" not in rules:
        if root_type == "root":
            rules["root"] = '"{" ws "}"'
        else:
            rules["root"] = root_type

    lines = [_PREAMBLE]
    for name, body in rules.items():
        lines.append(f"{name} ::= {body}")

    return "\n".join(lines) + "\n"
