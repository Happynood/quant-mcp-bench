# Changelog

## [Unreleased]

### Added
- Phase 7 (SCI statistical power + family coverage), part 1: fixed
  `schema/complexity.py`'s `_max_depth`/`_prop_count` to recurse through
  array `items`, not just object `properties` — the array-recursion gap
  already disclosed in the README. Recomputed SCI across the live 39-tool
  corpus; memory's mean SCI moved from -0.359 (2nd-lowest of 4 tiers) to
  +0.194 (2nd-highest), confirming it had been undercounted.
- Phase 7, part 2: added per-tool tracking end to end (`tool:` on every
  task fixture, a new `"instances"` field on `result.json`, and
  `report/sci_regression.py` + `quantmcp sci-regression`) so the SCI-vs-Δ
  regression (H2) uses one point per tool instead of one per tier. Added
  task coverage for the 3 previously-untested tools this made worth
  closing (`read_media_file`, `git_diff`, `git_checkout`), taking 38 of 39
  live tools to at least one task (`read_file` deliberately excluded: it's
  marked deprecated in favor of the schema-identical `read_text_file`, so
  no naturalistic instruction can fairly force its selection). Re-ran
  real GPU sweeps for both existing model families across all 4 tiers to
  populate the new per-instance data. Result: n=38, slope=+0.140, 95%
  bootstrap CI=[-0.007, +0.315] — a positive (H2-predicted-direction)
  relationship that doesn't reach significance, a materially different
  and better-powered null result than the original 4-point analysis.
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
  config; a Pareto-frontier chart (reliability vs. peak VRAM, `plotly`,
  optional `space` extra) marking which (model, quant, tier) configs are
  Pareto-optimal.
- Phase 6 (stretch): U4 `memory` tier (official reference knowledge-graph
  server via `npx`, 10 tasks covering all 9 real tools, a new `file_lacks`
  checker for verifying delete-style tools against the server's real
  post-execution state); real GPU sweep for both model families; SCI/CBC
  recomputed across all 4 tiers; first real exercise of the vendored GBNF
  constrained-decoding module against real MCP schemas (previously
  untested), with a real comparative sweep for Llama-3.2-1B on U1 and a
  root-caused explanation for why it gave no benefit.
- Phase 4: `dump-schemas` CLI command (frozen live tool schemas per tier)
  and `cross-bench --output` (machine-readable CBC result); published the
  `quantmcp-suite` and `quantmcp-results` HF datasets and a Gradio Space
  leaderboard (Leaderboard, Pareto Front, Cross-Benchmark, Schema
  Complexity, and About tabs) at
  [happynood/quantmcp-leaderboard](https://huggingface.co/spaces/happynood/quantmcp-leaderboard).

### Fixed
- Every committed result/manifest file and sweep config recorded the exact
  local absolute path this benchmark was run from, leaking the local
  username into public GitHub/HF artifacts. Replaced with a portable
  `~/models/...` path throughout; `QuantMCPConfig` now expands `~` at load
  time so the committed sweep configs keep working unchanged.
- The MCP leaderboard published local GGUF paths verbatim in its `model`
  column; now reuses the already-vendored `sanitize_model_name` (the same
  fix `quant-toolcall-bench` already applied to the same problem).
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
- `llama_cpp.verbose` and `llama_cpp.n_threads` were accepted by the config
  schema but silently had no effect: `verbose` was never forwarded from the
  CLI's backend builder, and `n_threads` wasn't even accepted by
  `LlamaCppBackend`'s constructor. Wired both through; added CLI-layer
  tests covering `compare`, `validate-config`, `leaderboard`, `sweep`, the
  `run` command's empty-task-list error path, and the backend-selection
  branch for every backend kind.
