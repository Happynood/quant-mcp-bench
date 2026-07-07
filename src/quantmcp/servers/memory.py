"""U4 memory tier (spec §5, stretch): the official reference `memory`
(knowledge-graph) MCP server.

Launched via `npx -y @modelcontextprotocol/server-memory`. Verified against
the npm registry on 2026-07-07: the package is
`@modelcontextprotocol/server-memory` (still present, version 2026.7.4) —
the reference server list has changed before, so this was checked rather
than assumed. Unlike the filesystem/git servers, this one takes no
positional path argument; it persists its knowledge graph as newline-
delimited JSON at the path given by the `MEMORY_FILE_PATH` env var, which
runner.py points at the per-instance sandbox root (so no "{root}"
substitution is needed in ARGS_TEMPLATE here).
"""

from __future__ import annotations

COMMAND = "npx"
ARGS_TEMPLATE = ["-y", "@modelcontextprotocol/server-memory"]
