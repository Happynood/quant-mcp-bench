# QuantMCP

[![Leaderboard](https://img.shields.io/badge/🤗%20Space-Leaderboard-yellow)](https://huggingface.co/spaces/happynood/quantmcp-leaderboard)
[![Results dataset](https://img.shields.io/badge/🤗%20Dataset-Results-blue)](https://huggingface.co/datasets/happynood/quantmcp-results)
[![Suite dataset](https://img.shields.io/badge/🤗%20Dataset-Suite-blue)](https://huggingface.co/datasets/happynood/quantmcp-suite)

Does quantization survive real, unmodified MCP tool schemas — not just
curated benchmark schemas? QuantMCP re-runs the `quant-toolcall-bench`
(QuantCall) measurement methodology (schema-validity, execution success,
quantization delta, bootstrap CI) against live Model Context Protocol
servers, executed end-to-end against sandboxed fixtures rather than only
structurally validated.

## Status

Real GPU results across four MCP server tiers (`filesystem`, `git`,
`sqlite`, `memory`) and three model families (Qwen3-0.6B, Qwen3-1.7B,
Llama-3.2-1B) — Qwen3-0.6B and Llama-3.2-1B at four quantization levels
(fp16, Q8_0, Q5_K_M, Q4_K_M each), Qwen3-1.7B at three (Q8_0, Q5_K_M,
Q4_K_M — its bf16 weights don't fit this project's 4GB card at a usable
context length, a real, confirmed limitation, not a gap; see
[`docs/RUN_REAL.md`](docs/RUN_REAL.md)). Full methodology, honest scope
limitations, and every real number are in
[`docs/RUN_REAL.md`](docs/RUN_REAL.md) — read that before citing anything
here.

## Headline findings so far

- **Cross-Benchmark Consistency (CBC) is negative, and got more negative
  with a 3rd model family, not less.** QuantCall's BFCL-measured
  quantization degradation pattern does not carry over cleanly to real
  MCP tool schemas. With 2 families the estimate was -0.551 (n=6 pairs,
  itself the product of 3 increasingly-averaged computations — see
  [`docs/RUN_REAL.md`](docs/RUN_REAL.md) for that convergence story).
  Adding **Qwen3-1.7B** as a 3rd family — a real within-family size
  contrast against Qwen3-0.6B, not just another family — moved CBC to
  **-0.755 (n=8 pairs)**: the sign didn't change, but the magnitude
  strengthened. Qwen3-1.7B itself shows the same flat, quantization-robust
  pattern as its smaller 0.6B sibling, just at a uniformly higher absolute
  level — support for H3 (model family predicts sensitivity, not size)
  from an actual size contrast, not only a family-vs-family one. Still
  n=8, still too few for a rigorous p-value on a Spearman correlation.
- **Real MCP schemas surface failure modes BFCL's curated schemas don't.**
  Llama-3.2-1B shifts between echoing back a tool's JSON *schema* instead
  of calling it (at fp16/Q8_0) and a flatter, sometimes-correct call shape
  (at Q5_K_M/Q4_K_M) — a qualitative format shift, not a smooth accuracy
  decline. On the sqlite tier, both models sometimes hallucinate a
  fictitious database schema or refuse tasks a real query could answer.
- **A real SCI bug was found and fixed, and it changed which tier looked
  simplest.** `_max_depth`/`_prop_count` didn't recurse into array `items`,
  so the `memory` tier's array-of-objects tools (`create_entities` etc.)
  scored as flat as a single string argument. Fixed: memory's mean SCI
  moved from -0.359 (2nd-lowest of 4 tiers) to **+0.194 (2nd-highest)** —
  it had been undercounted exactly as suspected.
- **With the bug fixed and the sample size increased from 4 tier-level
  points to 38 tool-level points, H2 is a genuinely better-powered null,
  not just a bigger version of the same one.** Per-tool SCI-vs-Δ
  regression: slope=+0.045, 95% CI=[-0.064, +0.170] (n=38, computed after
  Qwen3-1.7B was added — see below; +0.140 before it was added, sign
  unchanged either way). The *sign* actually flipped from the tier-level
  view (which ran opposite to H2) to positive — the direction H2 predicts
  — but the interval spans zero, so this isn't statistically significant.
  Full writeup and the
  reproducible `quantmcp sci-regression` command in
  [`docs/RUN_REAL.md`](docs/RUN_REAL.md).
- **SVR-MCP (schema-valid call) and TSR (actually correct outcome) diverge**
  — passing schema validation never implies task success, and the gap is
  largest on the sqlite tier.
- **Constrained decoding (GBNF) doesn't rescue Llama-3.2-1B, and is
  71-112% slower.** Extends QuantCall's own Qwen3-only negative result to
  a genuinely unreliable model, with a different, directly-verified cause:
  forcing the grammar's tool-call envelope (removing its abstention escape
  hatch) makes the model get stuck emitting whitespace rather than
  producing a correct call — constraining *which* tokens are legal can't
  manufacture a continuation the model has no real probability mass on.

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

Compute the per-tool SCI-vs-Δ regression (H2) from real result data and a
live/frozen tool-schema dump:

```bash
uv run quantmcp sci-regression results/*/*.result.json --schemas docs/live_schemas_phase7.json
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
  per-VRAM (η), cross-benchmark (CBC), and per-tool SCI-vs-Δ regression
  (H2) computation.
- `results/` — real `result.json` + manifest per (model, quant, tier); see
  [`CONTRIBUTING.md`](CONTRIBUTING.md) to add your own hardware's numbers.
- `docs/RUN_REAL.md` — the actual, current source of truth for every
  number this project has produced, including what didn't work.

## HuggingFace

| Artifact | URL |
|----------|-----|
| Eval suite (task fixtures + frozen schemas) | [happynood/quantmcp-suite](https://huggingface.co/datasets/happynood/quantmcp-suite) |
| Results dataset (submit your runs) | [happynood/quantmcp-results](https://huggingface.co/datasets/happynood/quantmcp-results) |
| Live leaderboard | [happynood/quantmcp-leaderboard](https://huggingface.co/spaces/happynood/quantmcp-leaderboard) |

## Related project

QuantCall ([`quant-toolcall-bench`](https://github.com/Happynood/quant-toolcall-bench))
established this measurement methodology against curated BFCL/ToolACE
schemas. QuantMCP reuses its Backend/parsing/validation/stats layers
verbatim and applies the same discipline to real MCP servers.

## License

MIT — see `LICENSE`. Citation: see `CITATION.cff`.
