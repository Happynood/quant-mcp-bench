"""MCPServerHandle: launch/sandbox/teardown a stdio MCP server (spec §6.2).

This is the one chokepoint every server tier (U0 toy, U1 filesystem, U2 git,
U3 sqlite, U4 memory) launches through. It never decides *what* command to
run — callers (servers/filesystem.py etc., or tests) pass the command/args —
but it does enforce that the subprocess's working directory is inside the
sandbox root: no MCP server may ever be pointed at the real project tree or
the user's home directory.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.types import Tool

from quantmcp.execution.sandbox import assert_within_sandbox


class MCPServerHandle:
    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if cwd is not None:
            assert_within_sandbox(Path(cwd))
        merged_env = get_default_environment()
        if env:
            merged_env.update(env)
        self._params = StdioServerParameters(
            command=command,
            args=args or [],
            env=merged_env,
            cwd=cwd,
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.server_name: str | None = None
        self.server_version: str | None = None

    async def __aenter__(self) -> MCPServerHandle:
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        init_result = await self._session.initialize()
        self.server_name = init_result.serverInfo.name
        self.server_version = init_result.serverInfo.version
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def list_tools(self) -> list[Tool]:
        if self._session is None:
            raise RuntimeError("MCPServerHandle used outside an `async with` block")
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("MCPServerHandle used outside an `async with` block")
        return await self._session.call_tool(name, arguments)
