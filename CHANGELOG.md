# Changelog

## [Unreleased]

### Added
- Phase 0: Project skeleton — vendored Backend/parsing/validation/metrics
  deltas+stats/report layers from `quant-toolcall-bench`, new MCP-specific
  layers (servers, execution/sandbox+dispatcher, tasks, schema/complexity,
  metrics/core), U0 in-repo toy FastMCP server, mock-backend smoke test,
  Click CLI skeleton, Makefile verify gate, GitHub Actions CI.
- Phase 1: U1 `filesystem` MCP server tier (official reference server via
  `npx`), 12 hand-verified naturalistic tasks, first real GPU sweep
  (Qwen3-0.6B, 4 quants) on an RTX 3050 Laptop 4GB.
- Phase 2: Llama-3.2-1B family-contrast sweep on U1; U2 `git` tier
  (official `mcp-server-git` via `uvx`, 10 tasks, deterministic tarball
  fixture); Spearman rank correlation helper and the `cross-bench` CLI
  command computing Cross-Benchmark Consistency (CBC) against
  `quant-toolcall-bench`'s published BFCL numbers; SVR-vs-TSR execution
  gap write-up.
- Phase 3: U3 `sqlite` tier (self-written FastMCP wrapper, since the
  official reference server is not currently maintained upstream); a new
  `sqlite_query_scalar` checker for verifying database state directly;
  bootstrap 95% CI on SVR-MCP/TSR (resampled from each run's own per-task
  outcomes); the `leaderboard` CLI command (per-server breakdown,
  reliability-per-VRAM η, and each tier's real Schema Complexity Index);
  a repeat-stability summary comparing independent runs of the same
  config.

### Fixed
- `hardware.py`'s GPU fingerprint collection failed silently on drivers
  that reject `--query-gpu=...,cuda_version,...` as a CSV field, leaving
  every manifest's `gpu` field `null` despite real GPU execution. Now
  queries the fields the driver does support and parses `cuda_version`
  from the plain-text banner separately.
- U1 `filesystem` task instructions originally named the tool to call
  explicitly, collapsing SVR-MCP to an argument-construction-only metric
  incomparable to BFCL's SVR (which also measures tool selection).
  Rewritten to be naturalistic; every checker re-verified against the
  real server's intended call before any dependent number was trusted.
