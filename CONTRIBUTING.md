# Contributing

## Submitting Benchmark Results

To add a model/quant/server combination to the leaderboard:

1. Run the evaluation on your own hardware:
   ```bash
   quantmcp run \
     --config configs/your_config.yaml \
     --output results/your_model_quant_server.result.json \
     --manifest results/your_model_quant_server.manifest.json
   ```
2. Verify the result file includes a manifest (git SHA, config hash, fixture
   bundle hash, MCP server package version, hardware fingerprint). The
   manifest is embedded in `result.json` automatically; `--manifest` also
   writes it standalone for easy diffing across runs.
3. Open a PR adding only the `results/*.result.json` and
   `results/*.manifest.json` files. Do not edit the leaderboard table
   manually — run `quantmcp leaderboard results/` to regenerate it from
   result files.

## Submitting a New MCP Server Tier

1. Add `src/quantmcp/servers/your_server.py` exposing a launch command
   compatible with `MCPServerHandle` (`src/quantmcp/servers/base.py`).
2. Add a committed, pinned fixture under
   `src/quantmcp/tasks/fixtures/your_server_fixture/` (directory snapshot,
   git tarball, or `.db` file — whatever the server operates on).
3. Hand-write 10-20 deterministic tasks in
   `src/quantmcp/tasks/fixtures/your_server_tasks.yaml`, each with a checker
   registered in `src/quantmcp/tasks/checkers.py`.
4. Register the tier in `src/quantmcp/config.py` (`ServerConfig.tier`) and
   `src/quantmcp/cli.py` (`_server_command_for_tier`).
5. Add tests in `tests/test_servers_your_server.py`, marked
   `pytestmark = pytest.mark.integration` if launching the real tool
   requires network access or an external runtime (Node/npx, uvx) — run
   them locally with `make test-integration`. `make verify`/the main CI
   job never depend on network access, so a fresh clone with no external
   tool available still passes.
6. Once the structural checks pass, attach a real `result.json` + manifest
   from your own hardware, then run `quantmcp leaderboard results/` to
   confirm it appears in the regenerated leaderboard before opening a PR.

## Code Contributions

### Setup

```bash
git clone https://github.com/Happynood/quant-mcp-bench
cd quant-mcp-bench
pip install uv
uv sync --dev
```

### Workflow

1. Create a feature branch.
2. Write failing tests first (TDD).
3. Implement until tests pass.
4. Run `make verify` — must be green before any PR.
5. Open a PR with a clear description.

### Verification Gate

```bash
make verify
```

This runs: `ruff check`, `ruff format --check`, `pyright`, the non-
integration test suite, and the smoke end-to-end test (mock backend + the
in-repo U0 toy server) — no network, no download, no GPU required. Run
`make test-integration` separately if you have Node.js/`npx` and `uvx`
available and want to exercise the real filesystem/git/sqlite/memory
servers.

## Hard Rules

- Never fabricate, hardcode, or guess metric values. Real numbers come only
  from real model runs against a real, sandboxed MCP server.
- The leaderboard ships empty and is populated only from verified
  `result.json` files.
- Every MCP server launch must go through `MCPServerHandle` with a `cwd`
  inside the sandbox root — see `SECURITY.md`.
- Every result file must include a complete manifest for reproducibility.

## Adding a Backend

1. Create `src/quantmcp/backends/your_backend.py` inheriting from `Backend`.
2. Implement `generate_toolcall(messages, tools) -> ToolCallResult`.
3. Register in `src/quantmcp/cli.py` (`_build_backend`).
4. Add an optional dependency group in `pyproject.toml`.
