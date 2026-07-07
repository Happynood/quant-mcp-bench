# First 9 tests vendored verbatim from Happynood/quant-toolcall-bench @6b6e29e5c83a
# (quantcall->quantmcp) — generic JSON-Schema-to-GBNF compiler tests, unrelated to
# BFCL vs MCP. Never previously wired into this project's test suite (Phase 0 vendored
# decoding/gbnf.py itself but not its tests, since constrained decoding was still a
# Phase 6 stretch goal at that point). The rest are new: build_tool_call_grammar
# against real MCP tool schemas, which gbnf.py had never actually been exercised
# against until Phase 6.
from __future__ import annotations

from quantmcp.decoding.gbnf import build_tool_call_grammar, gbnf_from_schema


def test_empty_object_schema():
    grammar = gbnf_from_schema({"type": "object", "properties": {}})
    assert "root" in grammar
    assert "{" in grammar


def test_string_property():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    grammar = gbnf_from_schema(schema)
    assert "string" in grammar
    assert "city" in grammar


def test_integer_property():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    grammar = gbnf_from_schema(schema)
    assert "integer" in grammar or "number" in grammar


def test_boolean_property():
    schema = {"type": "object", "properties": {"active": {"type": "boolean"}}}
    grammar = gbnf_from_schema(schema)
    assert "true" in grammar or "boolean" in grammar


def test_enum_property():
    schema = {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    }
    grammar = gbnf_from_schema(schema)
    assert "celsius" in grammar
    assert "fahrenheit" in grammar


def test_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    grammar = gbnf_from_schema(schema)
    assert "location" in grammar
    assert "string" in grammar


def test_array_property():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    grammar = gbnf_from_schema(schema)
    assert "array" in grammar or "[" in grammar


def test_grammar_is_string():
    grammar = gbnf_from_schema({"type": "object", "properties": {}})
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def _openai_tool(name: str, schema: dict) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": schema}}


def test_build_tool_call_grammar_against_real_memory_schema():
    """The memory tier's create_entities schema is an array of objects, the
    exact shape that previously crashed llama.cpp's GBNF parser on mixed
    "_"/"-" rule names (see _schema_rule_name's docstring) — regression-tests
    that fix against a real MCP schema, not just a synthetic one."""
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entityType": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "entityType", "observations"],
                },
            }
        },
        "required": ["entities"],
    }
    grammar = build_tool_call_grammar([_openai_tool("create_entities", schema)])
    assert "create_entities" in grammar
    assert "entityType" in grammar
    # rule names must not mix "_" and "-" (the documented llama.cpp segfault trigger)
    for line in grammar.splitlines():
        rule_name = line.split("::=")[0].strip()
        assert not ("_" in rule_name and "-" in rule_name), f"mixed separators: {rule_name!r}"


def test_build_tool_call_grammar_allows_no_call_by_default():
    tool = _openai_tool("read_graph", {"type": "object", "properties": {}})
    grammar = build_tool_call_grammar([tool])
    assert "no-call" in grammar
    assert "root ::= tool-call-path | no-call" in grammar


def test_build_tool_call_grammar_can_force_a_call():
    grammar = build_tool_call_grammar(
        [_openai_tool("read_graph", {"type": "object", "properties": {}})], allow_no_call=False
    )
    assert "root ::= tool-call-path" in grammar
    assert "no-call" not in grammar


def test_build_tool_call_grammar_multiple_tools_all_selectable():
    tools = [
        _openai_tool("read_graph", {"type": "object", "properties": {}}),
        _openai_tool(
            "search_nodes",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
    ]
    grammar = build_tool_call_grammar(tools)
    assert '"read_graph"' in grammar
    assert '"search_nodes"' in grammar
