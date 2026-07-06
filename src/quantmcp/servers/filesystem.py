"""U1 filesystem tier (spec §5): the official reference `filesystem` MCP server.

Launched via `npx -y @modelcontextprotocol/server-filesystem <sandbox_root>`.
Verified against github.com/modelcontextprotocol/servers on 2026-07-07: the
package is `@modelcontextprotocol/server-filesystem` (still present, version
0.6.3 at that commit) — the reference server list has changed before, so
this was checked rather than assumed. The server takes its allowed directory
as a positional CLI argument; "{root}" is substituted by runner.py with the
actual per-instance sandbox path, since that path only exists once the
sandbox has been created for a given task instance.
"""

from __future__ import annotations

COMMAND = "npx"
ARGS_TEMPLATE = ["-y", "@modelcontextprotocol/server-filesystem", "{root}"]
