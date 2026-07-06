from __future__ import annotations

from quantmcp.parsing.base import ParsedCall
from quantmcp.parsing.gguf_template import GGUFTemplateParser
from quantmcp.parsing.openai_tools import OpenAIToolsParser


def test_openai_tools_parser_basic():
    p = OpenAIToolsParser()
    raw = """{
  "choices": [{
    "message": {
      "tool_calls": [{
        "function": {
          "name": "get_weather",
          "arguments": "{\\"city\\": \\"Paris\\"}"
        }
      }]
    }
  }]
}"""
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}


def test_openai_tools_parser_multiple_calls():
    p = OpenAIToolsParser()
    raw = """{
  "choices": [{
    "message": {
      "tool_calls": [
        {"function": {"name": "tool_a", "arguments": "{\\"x\\": 1}"}},
        {"function": {"name": "tool_b", "arguments": "{\\"y\\": 2}"}}
      ]
    }
  }]
}"""
    calls = p.parse(raw)
    assert len(calls) == 2
    assert {c.name for c in calls} == {"tool_a", "tool_b"}


def test_openai_tools_parser_no_tool_call():
    p = OpenAIToolsParser()
    raw = '{"choices": [{"message": {"content": "Hello!", "tool_calls": null}}]}'
    calls = p.parse(raw)
    assert calls == []


def test_openai_tools_parser_plain_text():
    p = OpenAIToolsParser()
    calls = p.parse("Hello, how can I help you?")
    assert calls == []


def test_openai_tools_parser_name():
    assert OpenAIToolsParser().name == "openai_tools"


def test_gguf_template_parser_json_block():
    p = GGUFTemplateParser()
    raw = '<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "test"}


def test_gguf_template_parser_python_style():
    p = GGUFTemplateParser()
    raw = "get_weather(city='Paris', unit='celsius')"
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"


def test_gguf_template_parser_no_call():
    p = GGUFTemplateParser()
    assert p.parse("I cannot help with that.") == []


def test_gguf_template_parser_name():
    assert GGUFTemplateParser().name == "gguf_template"


def test_parsed_call_hashable():
    c = ParsedCall(name="foo", arguments={"a": 1})
    s = {c}
    assert c in s
