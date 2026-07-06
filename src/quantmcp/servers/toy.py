"""U0 smoke-tier toy MCP server (spec §5): a 2-tool FastMCP server, in-repo,
stdio transport, no download. Exists purely to smoke-test the parser/
validator/dispatcher plumbing in CI without needing Node/npx or a real
reference server.

Every filesystem-touching tool is scoped to $QUANTMCP_U0_ROOT — the sandbox
instance directory created by execution.sandbox — never the real working
tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("quantmcp-u0-toy")


def _sandbox_root() -> Path:
    root = os.environ.get("QUANTMCP_U0_ROOT")
    if not root:
        raise RuntimeError("QUANTMCP_U0_ROOT must be set before write_note can be used")
    return Path(root)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@mcp.tool()
def write_note(filename: str, content: str) -> str:
    """Write content to a file named `filename` inside the sandboxed root dir.

    Only the basename of `filename` is used, so a call cannot escape the
    sandbox root via a path-traversal argument (e.g. "../../etc/passwd").
    """
    root = _sandbox_root()
    safe_name = Path(filename).name
    target = root / safe_name
    target.write_text(content)
    return f"wrote {len(content)} bytes to {safe_name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
