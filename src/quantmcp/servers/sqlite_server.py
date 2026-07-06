"""U3 sqlite tier (spec §5): the official reference `sqlite` MCP server was
checked against github.com/modelcontextprotocol/servers (commit d31124c,
2026-07-06) and is not present in the current server list — only
`everything, fetch, filesystem, git, memory, sequentialthinking, time` are
maintained there. Per the disclosed scope-note convention this project
follows for exactly this situation, U3 uses a minimal self-written FastMCP
wrapper instead, over a committed fixture `.db` file, with a tool surface
(`list_tables`/`describe_table`/`read_query`/`write_query`) that matches the
tool names the reference server historically exposed before removal, so the
schema shape stays representative of a real SQL-over-MCP tool rather than
inventing a different interface.

Every tool is scoped to $QUANTMCP_U3_ROOT/fixture.db — the sandbox instance
directory created by execution.sandbox, never the real working tree.
`read_query`/`write_query` reject statements outside their own statement
class (SELECT-only / mutating-only) and reject `ATTACH`/`PRAGMA`, since
attaching an external database file would be a real sandbox-escape vector
for a tool whose entire purpose is running arbitrary SQL.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("quantmcp-u3-sqlite")

_FORBIDDEN_KEYWORDS = ("attach", "pragma", "detach")


def _db_path() -> Path:
    root = os.environ.get("QUANTMCP_U3_ROOT")
    if not root:
        raise RuntimeError("QUANTMCP_U3_ROOT must be set before the sqlite tools can be used")
    return Path(root) / "fixture.db"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_db_path())


def _reject_forbidden(query: str) -> None:
    lowered = query.strip().lower()
    if any(keyword in lowered for keyword in _FORBIDDEN_KEYWORDS):
        raise ValueError("query contains a forbidden keyword (attach/detach/pragma)")


@mcp.tool()
def list_tables() -> str:
    """List every table name in the sandboxed sqlite database."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return "\n".join(r[0] for r in rows)


@mcp.tool()
def describe_table(table: str) -> str:
    """Describe the columns of `table` (name, type, nullable, primary key)."""
    conn = _connect()
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    finally:
        conn.close()
    lines = [
        f"{r[1]} {r[2]} {'NOT NULL' if r[3] else ''} {'PK' if r[5] else ''}".strip() for r in rows
    ]
    return "\n".join(lines)


@mcp.tool()
def read_query(query: str) -> str:
    """Run a read-only SELECT query against the sandboxed sqlite database."""
    if not query.strip().lower().startswith("select"):
        raise ValueError("read_query only accepts SELECT statements")
    _reject_forbidden(query)
    conn = _connect()
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
    finally:
        conn.close()
    header = ", ".join(columns)
    body = "\n".join(", ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{body}" if header else body


@mcp.tool()
def write_query(query: str) -> str:
    """Run an INSERT/UPDATE/DELETE statement against the sandboxed sqlite database."""
    lowered = query.strip().lower()
    if lowered.startswith("select"):
        raise ValueError("write_query does not accept SELECT statements; use read_query")
    _reject_forbidden(query)
    conn = _connect()
    try:
        cursor = conn.execute(query)
        conn.commit()
        affected = cursor.rowcount
    finally:
        conn.close()
    return f"{affected} row(s) affected"


if __name__ == "__main__":
    mcp.run(transport="stdio")
