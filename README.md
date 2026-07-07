# QuantMCP

Does quantization survive real, unmodified MCP tool schemas — not just
curated benchmark schemas? QuantMCP re-runs the `quant-toolcall-bench`
(QuantCall) measurement methodology (schema-validity, execution success,
quantization delta, bootstrap CI) against live Model Context Protocol
servers, executed end-to-end against sandboxed fixtures rather than only
structurally validated.

## Status

Real GPU results across three MCP server tiers (`filesystem`, `git`,
`sqlite`) and two model families (Qwen3-0.6B, Llama-3.2-1B), each at four
quantization levels (fp16, Q8_0, Q5_K_M, Q4_K_M). Full methodology,
honest scope limitations, and every real number are in
[`docs/RUN_REAL.md`](docs/RUN_REAL.md) — read that before citing anything
here.

## Headline findings so far

- **Cross-Benchmark Consistency (CBC) is negative.** QuantCall's
  BFCL-measured quantization degradation pattern does not carry over
  cleanly to real MCP tool schemas for either model family tested. The
  exact magnitude is not yet stable at this sample size (a re-run of the
  identical config moved the point estimate from -0.824 to -0.265 — sign
  stable, magnitude not), so treat the direction, not the number, as the
  finding.
- **Real MCP schemas surface failure modes BFCL's curated schemas don't.**
  Llama-3.2-1B shifts between echoing back a tool's JSON *schema* instead
  of calling it (at fp16/Q8_0) and a flatter, sometimes-correct call shape
  (at Q5_K_M/Q4_K_M) — a qualitative format shift, not a smooth accuracy
  decline. On the sqlite tier, both models sometimes hallucinate a
  fictitious database schema or refuse tasks a real query could answer.
- **Schema complexity (SCI) does not obviously predict degradation** in
  the 3 tiers measured so far — if anything, the simplest-schema tier
  (sqlite) shows the largest degradation swing. Only 3 tiers exist so far,
  which is too few for a real correlation; this is flagged as a
  methodological question (SCI measures schema *shape*, not argument
  *content* difficulty) worth carrying into later tiers, not a settled
  result.
- **SVR-MCP (schema-valid call) and TSR (actually correct outcome) diverge**
  — passing schema validation never implies task success, and the gap is
  largest on the sqlite tier.

## Quickstart

```bash
uv sync
uv run quantmcp run --config configs/smoke.yaml   # mock backend, no GPU, no download
```

Real GPU run (requires a downloaded GGUF model — see
[`docs/RUN_REAL.md`](docs/RUN_REAL.md) for the full setup and the CUDA
offload verification step):

```bash
uv sync --extra llama-cpp
uv run quantmcp run --config configs/qwen3-0.6b-u1-sweep/Q4_K_M.yaml
```

Build the leaderboard + per-server breakdown from any directory of real
results. `uv sync --extra space` first also produces a Pareto/η chart
(`leaderboard/pareto.html`, plotting reliability against peak VRAM):

```bash
uv sync --extra space
uv run quantmcp leaderboard results/ --output-dir leaderboard
```

Compute Cross-Benchmark Consistency against QuantCall's published BFCL
numbers:

```bash
uv run quantmcp cross-bench results/*/*.result.json --bfcl-results docs/bfcl_reference_svr.json
```

## Repository layout

- `src/quantmcp/servers/` — MCP server launch wrappers (`filesystem`,
  `git`, `sqlite`) and the sandbox-scoped `MCPServerHandle`.
- `src/quantmcp/tasks/` — hand-written, hand-verified task fixtures per
  tier, plain-Python checkers (no model-as-judge scoring).
- `src/quantmcp/execution/` — sandboxing (`/tmp/quantmcp-sandbox/<run_id>/`,
  fresh per instance, destroyed after) and call dispatch.
- `src/quantmcp/metrics/` — SVR-MCP/TSR (new) plus vendored deltas/
  bootstrap-CI/Spearman-correlation helpers.
- `src/quantmcp/report/` — leaderboard, per-server breakdown, reliability-
  per-VRAM (η), and cross-benchmark (CBC) computation.
- `results/` — real `result.json` + manifest per (model, quant, tier); see
  [`CONTRIBUTING.md`](CONTRIBUTING.md) to add your own hardware's numbers.
- `docs/RUN_REAL.md` — the actual, current source of truth for every
  number this project has produced, including what didn't work.

## Related project

QuantCall ([`quant-toolcall-bench`](https://github.com/Happynood/quant-toolcall-bench))
established this measurement methodology against curated BFCL/ToolACE
schemas. QuantMCP reuses its Backend/parsing/validation/stats layers
verbatim and applies the same discipline to real MCP servers.

## License

MIT — see `LICENSE`. Citation: see `CITATION.cff`.
