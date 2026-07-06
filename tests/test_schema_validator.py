from __future__ import annotations

from quantmcp.parsing.base import ParsedCall
from quantmcp.validation.schema_validator import validate_call, validate_calls


def _schema_with_required() -> dict:
    return {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    }


def _schema_no_required() -> dict:
    return {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
    }


def test_valid_call_passes():
    call = ParsedCall(name="get_weather", arguments={"city": "Paris", "unit": "celsius"})
    assert validate_call(call, _schema_with_required()) is True


def test_valid_call_minimal_required_only():
    call = ParsedCall(name="get_weather", arguments={"city": "Berlin"})
    assert validate_call(call, _schema_with_required()) is True


def test_invalid_call_missing_required():
    call = ParsedCall(name="get_weather", arguments={})
    assert validate_call(call, _schema_with_required()) is False


def test_invalid_call_wrong_enum():
    call = ParsedCall(name="get_weather", arguments={"city": "Paris", "unit": "kelvin"})
    assert validate_call(call, _schema_with_required()) is False


def test_invalid_call_wrong_type():
    call = ParsedCall(name="get_weather", arguments={"city": 42})
    assert validate_call(call, _schema_with_required()) is False


def test_valid_call_no_required_empty_args():
    call = ParsedCall(name="web_search", arguments={})
    assert validate_call(call, _schema_no_required()) is True


def test_validate_calls_all_valid():
    tool_schemas = {"get_weather": _schema_no_required()}
    calls = [ParsedCall(name="get_weather", arguments={"query": "Paris"})]
    assert validate_calls(calls, tool_schemas) is True


def test_validate_calls_unknown_tool():
    tool_schemas = {"get_weather": _schema_no_required()}
    calls = [ParsedCall(name="unknown_tool", arguments={})]
    assert validate_calls(calls, tool_schemas) is False


def test_validate_calls_empty():
    assert validate_calls([], {}) is True
