# Security Policy

## Scope

QuantMCP is a local benchmarking tool that launches real Model Context
Protocol servers (filesystem, git, sqlite, memory) against sandboxed
fixtures. Security concerns are limited to:

- MCP server sandboxing (see below) — this is the main security-relevant
  design decision in this project, unlike a purely offline benchmark.
- Local file handling (result files, fixture bundles, manifests).
- Optional remote model endpoints (OpenAI-compatible backend).

## MCP Server Sandboxing

Every MCP server instance is launched with its working directory pointed at
a freshly created, ephemeral copy of its fixture, rooted under
`/tmp/quantmcp-sandbox/<run_id>/<instance_id>/`, and destroyed once the
instance completes (`src/quantmcp/execution/sandbox.py`). No server may ever
be pointed at the real repository checkout or a user's home directory —
`assert_within_sandbox()` is the single enforcement point every server
launch (`src/quantmcp/servers/base.py`) routes through, and it raises rather
than silently permitting an escape. This matters specifically because a
quantized, degraded model is expected to occasionally emit a path-traversal
or out-of-scope argument (a real failure mode this project measures, not
just guards against) — the sandbox boundary must hold even when the model
being tested is actively misbehaving.

## Reporting a Vulnerability

To report a security issue, open a GitHub issue with the label `security`.
For sensitive reports (e.g. a sandbox-escape path), use GitHub's private
security advisory feature instead of a public issue.

## Dependencies

Keep dependencies up to date. Run `uv sync` to install pinned versions.
Review `uv.lock` before deploying in shared environments. The `mcp` SDK is
pinned to `>=1.27,<2` deliberately (see `docs/RUN_REAL.md`) — do not bump
past the stable v1.x line without re-auditing the sandboxing code against
the new protocol version.

## Model Endpoints

When using the `openai` backend with a remote endpoint (including a local
server such as LM Studio), keep API keys in environment variables — never in
config YAML files committed to version control.

```bash
export QUANTMCP_API_KEY=sk-...
quantmcp run --config configs/openai.yaml
```
