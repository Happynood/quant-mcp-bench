from __future__ import annotations

import sys

import pytest

from quantmcp.execution.dispatcher import execute_calls
from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.parsing.base import ParsedCall
from quantmcp.servers.base import MCPServerHandle

TOY_COMMAND = sys.executable
TOY_ARGS = ["-m", "quantmcp.servers.toy"]


@pytest.mark.asyncio
async def test_toy_server_lists_two_tools():
    with sandbox_instance(fixture_dir=None, run_id="toy-list-tools") as root:
        async with MCPServerHandle(TOY_COMMAND, TOY_ARGS, cwd=root) as handle:
            tools = await handle.list_tools()
            names = {t.name for t in tools}
            assert names == {"add", "write_note"}
            assert handle.server_name == "quantmcp-u0-toy"


@pytest.mark.asyncio
async def test_toy_server_add_tool_call():
    with sandbox_instance(fixture_dir=None, run_id="toy-add") as root:
        async with MCPServerHandle(TOY_COMMAND, TOY_ARGS, cwd=root) as handle:
            result = await handle.call_tool("add", {"a": 2, "b": 3})
            assert result.isError is not True
            text = result.content[0].text
            assert float(text) == 5.0


@pytest.mark.asyncio
async def test_toy_server_write_note_scoped_to_sandbox():
    with sandbox_instance(fixture_dir=None, run_id="toy-write") as root:
        env = {"QUANTMCP_U0_ROOT": str(root)}
        async with MCPServerHandle(TOY_COMMAND, TOY_ARGS, env=env, cwd=root) as handle:
            result = await handle.call_tool("write_note", {"filename": "note.txt", "content": "hi"})
            assert result.isError is not True
            assert (root / "note.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_toy_server_write_note_path_traversal_stays_in_sandbox():
    with sandbox_instance(fixture_dir=None, run_id="toy-traversal") as root:
        env = {"QUANTMCP_U0_ROOT": str(root)}
        async with MCPServerHandle(TOY_COMMAND, TOY_ARGS, env=env, cwd=root) as handle:
            result = await handle.call_tool(
                "write_note", {"filename": "../../../etc/evil.txt", "content": "hi"}
            )
            assert result.isError is not True
            # only the basename is honored — nothing escapes the sandbox root
            assert (root / "evil.txt").read_text() == "hi"
            assert not (root.parent.parent / "etc" / "evil.txt").exists()


@pytest.mark.asyncio
async def test_dispatcher_executes_calls_against_live_server():
    with sandbox_instance(fixture_dir=None, run_id="toy-dispatch") as root:
        env = {"QUANTMCP_U0_ROOT": str(root)}
        async with MCPServerHandle(TOY_COMMAND, TOY_ARGS, env=env, cwd=root) as handle:
            calls = [
                ParsedCall(name="add", arguments={"a": 10, "b": 5}),
                ParsedCall(name="does_not_exist", arguments={}),
            ]
            results = await execute_calls(handle, calls)
            assert len(results) == 2
            assert results[0].ok is True
            assert results[1].ok is False
            assert results[1].error is not None
