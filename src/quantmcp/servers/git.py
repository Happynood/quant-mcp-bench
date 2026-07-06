"""U2 git tier (spec §5): the official reference `git` MCP server.

Launched via `uvx mcp-server-git --repository <sandbox_root>`. Verified
against github.com/modelcontextprotocol/servers at commit d31124c
(2026-07-06): the package is `mcp-server-git` (still present, version 0.6.2
at that commit) — the reference server list has changed before, so this was
checked rather than assumed. Unlike the filesystem server, every git tool
call also takes `repo_path` as an explicit per-call argument (the
`--repository` launch flag only sets a default); "{root}" is substituted by
runner.py with the actual per-instance sandbox path, since that path only
exists once the sandbox has been created for a given task instance.
"""

from __future__ import annotations

COMMAND = "uvx"
ARGS_TEMPLATE = ["mcp-server-git", "--repository", "{root}"]
