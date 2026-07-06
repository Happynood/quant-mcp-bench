# Running Real GPU Evaluations

This guide covers running QuantMCP against actual quantized models on a GPU,
and documents exactly what has been run so far to produce the numbers in
this repository.

## Prerequisites

- Python 3.11+, `uv` installed
- Node.js (`npx`) on PATH — needed for the `filesystem`/`memory` reference
  MCP servers
- A CUDA-capable NVIDIA GPU. Only the driver is required in principle — the
  CUDA *toolkit* does not need to be installed for a prebuilt wheel — but
  see the caveat below.
- A downloaded GGUF model file

## Install with the llama.cpp backend

```bash
git clone https://github.com/Happynood/quant-mcp-bench
cd quant-mcp-bench
uv sync --extra llama-cpp
```

### GPU offload caveat

`pyproject.toml`'s `llama-cpp` extra pulls in `nvidia-cuda-runtime-cu12` and
`nvidia-cublas-cu12`, and `LlamaCppBackend` preloads them with
`ctypes.RTLD_GLOBAL` before importing `llama_cpp` — this is enough to fix
the common `libcudart.so.12: cannot open shared object file` error on
driver-only machines, **provided the installed `llama-cpp-python` wheel was
actually built with CUDA support in the first place**.

Importing `llama_cpp` successfully does **not** mean CUDA is available: a
plain `uv sync --extra llama-cpp` may resolve to a source build (no prebuilt
wheel available for your platform/Python/CUDA combination), and a source
build without `CMAKE_ARGS="-DGGML_CUDA=on"` is silently CPU-only. Always
verify before trusting a timing/VRAM number:

```bash
uv run python3 -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

If this prints `False`, rebuild with the CUDA toolkit (`nvcc`) installed:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp
```

or obtain a CUDA-enabled build through another verified channel, and
re-check with the command above. For extra certainty, run with
`llama_cpp.verbose=True` and look for `load_tensors: layer N assigned to
device CUDA0` — a CPU-only build reports `device CPU` for every layer.

## Download a GGUF model

Qwen3-0.6B, chosen for the same reason quant-toolcall-bench chose it: its
fp16 (bf16) weights (~1.5 GB) fit a 4 GB card with room to spare.

```bash
pip install huggingface-hub
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
for QUANT_FILE in Q4_K_M Q5_K_M Q8_0 bf16; do
    hf download \
        bartowski/Qwen_Qwen3-0.6B-GGUF \
        --include "Qwen_Qwen3-0.6B-${QUANT_FILE}.gguf" \
        --local-dir ~/models/
done
```

## What has actually been run (Phase 1, superseded by Phase 2 below)

Hardware: RTX 3050 Laptop GPU, 4 GB VRAM (3772 MiB reported free by
`ggml_cuda_init`), driver 595.71.05, CUDA 13.2 (driver-reported; the
`llama-cpp-python` build actually used links against the CUDA 12.x runtime
via the preloaded `nvidia-cuda-runtime-cu12`/`nvidia-cublas-cu12` packages).

Config: `configs/qwen3-0.6b-u1-sweep/{fp16,Q8_0,Q5_K_M,Q4_K_M}.yaml` —
Qwen3-0.6B (bf16 GGUF used as the fp16 baseline, matching
quant-toolcall-bench's own convention), `llama-cpp` backend,
`chat_variant: qwen3_nothink` (suppresses the `<think>` block via a
`/no_think` suffix and parses with the Hermes XML tool-call parser), U1
`filesystem` MCP server tier, 12 hand-written tasks, greedy decoding
(temperature 0), single seed (0), single repeat.

The first Phase 1 run of this sweep used task instructions that explicitly
named the tool to call for every task (e.g. "Use the `read_text_file` tool
to read..."). That run's numbers (SVR-MCP 0.917-1.000 across all four
quants) are superseded by the Phase 2 re-run below and are no longer
reported here — see "Task design: tool selection must stay in scope" for
why.

## Task design: tool selection must stay in scope (Phase 2 correction)

QuantCall's BFCL-based SVR for Qwen3-0.6B is fp16=0.877, Q8_0=0.878,
Q5_K_M=0.878, Q4_K_M=0.873 (free decoding, T1+T6). The first Phase 1 U1
sweep came back noticeably higher (0.917-1.000) and, per this project's own
cross-check requirement, that gap was investigated before being trusted: it
was **not a harness bug**, it was a task-design difference. BFCL's
natural-language queries never name the function to call, so its SVR also
captures *tool-selection* difficulty. Every one of the original 12 U1 tasks
explicitly said "use the `<tool_name>` tool," which removed tool selection
and left only argument-construction — a substantially easier structural-
validity bar than BFCL's.

Fix: `src/quantmcp/tasks/fixtures/u1_filesystem_tasks.yaml` was rewritten so
every instruction is naturalistic and never names the tool (e.g. "There's a
to-do list at {root}/notes/todo.txt. What does it say?" instead of "Use the
`read_text_file` tool..."). All 12 checkers were re-verified against the
exact intended call before re-running anything
(`tests/test_servers_filesystem.py::test_all_u1_tasks_pass_with_the_intended_call`),
then the full sweep was re-run.

While investigating this, a second, unrelated bug was found and fixed:
`hardware.py`'s GPU fingerprint collection queried `nvidia-smi
--query-gpu=...,cuda_version,...`, but this driver's `nvidia-smi` (595.71.05)
rejects `cuda_version` as a queryable CSV field (it only appears in the
plain-text banner on this release) — the query failed outright and a bare
`except Exception: pass` silently discarded it, so every manifest produced
so far (including the original Phase 1 files) recorded `"gpu": null` despite
genuinely running on the GPU. Fixed by querying only the fields this driver
does support via CSV and parsing `cuda_version` separately from the
plain-text banner. Regression-tested in `tests/test_hardware.py`. The
numbers below are the first ones with a real, non-null GPU fingerprint.

## What has actually been run (Phase 2 re-run, current)

Same config/hardware/hyperparameters as above, only the task instructions
and the GPU-fingerprint fix changed.

| Quant | SVR-MCP | TSR | Peak VRAM (GB) |
|---|---|---|---|
| fp16 (bf16.gguf) | 0.833 | 0.750 | 1.995 |
| Q8_0 | 0.833 | 0.667 | 1.474 |
| Q5_K_M | 0.833 | 0.750 | 1.292 |
| Q4_K_M | 0.833 | 0.667 | 1.247 |

These SVR-MCP numbers (0.833 flat across all four quants) now sit close to
QuantCall's published BFCL SVR range (0.873-0.878) rather than well above
it — consistent with the task-design fix actually closing the gap, not just
moving it. TSR (execution success) is lower than or equal to SVR-MCP at
every quant, as expected: TSR additionally requires the executed call to
produce the *correct* outcome, not just a schema-valid one. TSR varies by
one task (0.667 vs 0.750, i.e. 8/12 vs 9/12) between quants in a way that
does not track monotonically with precision — at n=12, single-seed,
single-repeat, this is consistent with the GPU decode non-determinism
already documented below, not a precision-ordered effect.

Raw results + manifests: `results/qwen3-0.6b-u1/*.result.json` (each embeds
its own manifest: git commit, config hash, fixture hash, MCP server package
version `0.2.0` for `@modelcontextprotocol/server-filesystem`, hardware
fingerprint including GPU name/driver/CUDA version — all captured
automatically, not hand-entered).

### Honest scope limitations of this result

1. **n=12, single seed, single repeat.** This is a plumbing-proving MVP
   result (spec's Phase 1 goal), not a statistically powered sweep. No
   bootstrap CI is reported here for that reason — at n=12 a CI would be
   too wide to say anything the raw numbers don't already say. Phase 3 adds
   bootstrap CI across a larger task set and multiple seeds/repeats.
2. **GPU greedy decoding is not perfectly bit-deterministic run-to-run** —
   a repeated Q4_K_M run in Phase 1 produced 12/12 passing where the
   recorded sweep run got 11/12 (non-associative floating-point reduction
   order in parallel GPU kernels). Expected, not a bug, and exactly why
   multiple seeds/repeats and bootstrap CI matter for any claim stronger
   than "plumbing works."
3. **TSR is currently execution-success only for this tool corpus** — all
   12 tasks are single-call, no multi-step chaining yet (matches spec's
   "single/few-step, cheap-to-execute tasks" scope for tractability on a
   4 GB GPU).

## Family contrast: Llama-3.2-1B (Phase 2)

QuantCall's own BFCL results flag Llama-3.2-1B as its most quantization-
sensitive family: SVR (T1+T6, n=200) fp16=0.327, Q8_0=0.305, Q5_K_M=0.313,
Q4_K_M=0.280 — a real, if modest, monotonic-ish decline. To see whether the
same family shows a comparable pattern on real MCP schemas, the identical
U1 filesystem sweep (naturalistic tasks, same 12 tasks, greedy, seed 0,
single repeat) was run for Llama-3.2-1B-Instruct across the same 4 quants,
same `llama-cpp` backend, default (`raw_json`) parser (Llama does not need
the `qwen3_nothink` chat variant).

| Quant | SVR-MCP | TSR | Peak VRAM (GB) |
|---|---|---|---|
| fp16 | 0.000 | 0.000 | 2.896 |
| Q8_0 | 0.000 | 0.000 | 1.823 |
| Q5_K_M | 0.250 | 0.250 | 1.440 |
| Q4_K_M | 0.250 | 0.250 | 1.345 |

Raw results + manifests: `results/llama3.2-1b-u1/*.result.json`.

**This is the opposite direction from BFCL's monotonic decline, and was
investigated before being written here** (per the cross-check requirement):
it is a genuine, reproducible model behavior difference, not a harness bug.
Manually driving the backend with verbose raw-output logging for all 12
tasks at each quant showed a clear, consistent failure-mode shift:

- At **fp16 and Q8_0**, the model responds to every task by echoing back a
  tool-*definition*-shaped JSON object — `{"type": "function", "function":
  {"name": ..., "parameters": {<the tool's own JSON Schema>}}}` — instead of
  a tool *call* with filled-in argument values. This is not a valid call
  under any of the parser's recognized shapes (it correctly does not treat
  a schema echo as a call), so SVR-MCP is genuinely 0 at both quants: the
  model never attempts an actual call.
- At **Q5_K_M and Q4_K_M**, the model partially shifts to a flatter
  `{"name": "<tool>", "parameters": {<actual values>}}` shape for several
  (not all) tasks — a shape the parser does accept as a call. Of the 3
  tasks that validate (`list_directory`, `write_file`, `get_file_info`),
  the tool names are genuine and the arguments are correct; on other tasks
  the model still hallucinates a plausible-sounding but nonexistent tool
  name (`read_file`, `rename_file` — neither exists in the real
  `server-filesystem` tool list, whose actual names are `read_text_file`
  and `move_file`), which correctly fails SVR-MCP.

In other words, quantization did not make this model uniformly worse here —
it shifted which *output format* the model defaults to, and the flatter
format Q5_K_M/Q4_K_M happen to produce is easier for the parser to accept
as a real (if sometimes wrong) call attempt than the schema-echo fp16/Q8_0
consistently produce. BFCL's simpler, terser function signatures apparently
never trigger the schema-echo failure mode in QuantCall's own results, so
this qualifies as a genuinely new observation from testing on real,
unmodified MCP tool schemas rather than a replication of the BFCL finding.

**This result should not be over-read**: n=12, single seed, single repeat,
and the entire effect is 0/12 vs 3/12 — a 3-task swing. At this sample size
that is the finest resolution the harness has; it is reported honestly
because the underlying raw-output failure-mode shift was directly observed
and is qualitatively real, not because the point estimate itself is
precise. Phase 3's larger task set, multiple seeds, and bootstrap CI are
what would be needed to state a confidence interval on this effect.

## U2 git tier (Phase 2)

The same two models were also run against the U2 `git` tier (official
`mcp-server-git`, launched via `uvx`), 10 naturalistic tasks (status, log,
diff unstaged/staged, branch list, show HEAD, add, reset, create-branch,
commit) against a small fixed fixture repository.

| Model | Quant | SVR-MCP | TSR |
|---|---|---|---|
| Qwen3-0.6B | fp16 | 1.000 | 1.000 |
| Qwen3-0.6B | Q8_0 | 0.900 | 0.900 |
| Qwen3-0.6B | Q5_K_M | 1.000 | 1.000 |
| Qwen3-0.6B | Q4_K_M | 1.000 | 0.800 |
| Llama-3.2-1B | fp16 | 0.400 | 0.200 |
| Llama-3.2-1B | Q8_0 | 0.400 | 0.200 |
| Llama-3.2-1B | Q5_K_M | 0.600 | 0.300 |
| Llama-3.2-1B | Q4_K_M | 0.600 | 0.200 |

Raw results + manifests: `results/{qwen3-0.6b,llama3.2-1b}-u2/*.result.json`.
Llama-3.2-1B shows the same qualitative direction here as on U1 (higher
SVR-MCP at lower precision), consistent with the same schema-echo-vs-flat-
call failure-mode shift described above rather than a filesystem-tier-
specific artifact.

## U3 sqlite tier (Phase 3)

The official reference `sqlite` MCP server is not present in the current
`github.com/modelcontextprotocol/servers` list (checked again at commit
`d31124c`, 2026-07-06 — only `everything, fetch, filesystem, git, memory,
sequentialthinking, time` are maintained there). Per this project's
disclosed-fallback convention, U3 uses a minimal self-written FastMCP
wrapper (`servers/sqlite_server.py`) over a committed fixture `.db` file
instead, exposing the same tool names (`list_tables`/`describe_table`/
`read_query`/`write_query`) the reference server historically had. 10
naturalistic tasks against a 2-table (`employees`, `inventory`) fixture.

| Model | Quant | SVR-MCP | TSR |
|---|---|---|---|
| Qwen3-0.6B | fp16 | 0.500 | 0.400 |
| Qwen3-0.6B | Q8_0 | 0.500 | 0.300 |
| Qwen3-0.6B | Q5_K_M | 0.400 | 0.400 |
| Qwen3-0.6B | Q4_K_M | 0.800 | 0.500 |
| Llama-3.2-1B | fp16 | 0.200 | 0.100 |
| Llama-3.2-1B | Q8_0 | 0.200 | 0.100 |
| Llama-3.2-1B | Q5_K_M | 0.100 | 0.100 |
| Llama-3.2-1B | Q4_K_M | 0.300 | 0.100 |

Both models score markedly worse here than on U1/U2, and the fp16 raw
output was inspected directly to confirm this is genuine model behavior on
this specific tool surface, not a harness bug. Two new failure modes appear
that filesystem/git tasks didn't trigger:

- **Hallucinated schema instead of a tool call.** Asked "What tables exist
  in this database?", Qwen3-0.6B fp16 answered directly with a fabricated,
  plausible-sounding schema ("users", "orders", "products") — none of which
  exist in the real fixture — instead of calling the zero-argument
  `list_tables` tool it had just been given.
- **Outright refusal on tasks that need a query.** Asked "What is Carla
  Diaz's salary?" or "How many widgets are currently in inventory?", the
  same model responded that it could not answer without more information,
  rather than recognizing that `read_query` could resolve the question.

Both are genuine capability gaps at this scale on this particular
ambiguous-natural-language-to-SQL task shape, not a parser or schema
mismatch: the model *did* call tools correctly and validly for the more
directly-worded tasks (`describe_table`, `write_query` with syntactically
valid SQL) — it just sometimes hallucinated non-existent table/column
names in the generated SQL (e.g. `UPDATE salaries SET salary = ... WHERE
employee_id = 1` against a schema with no `salaries` table and no
`employee_id` column), which is exactly the SVR-MCP/TSR split doing its
job: the call is schema-valid (a `query: string` argument, structurally
fine) but execution-wrong.

Raw results + manifests: `results/{qwen3-0.6b,llama3.2-1b}-u3/*.result.json`.

## Cross-Benchmark Consistency (CBC, spec §4.5) and the SVR-vs-TSR gap (H4)

CBC asks: does QuantCall's BFCL-measured quantization degradation pattern
predict what happens on real MCP schemas? It is computed as the Spearman
correlation between each (model, quant) pair's BFCL SVR delta (vs. that
model's own fp16 baseline) and its SVR-MCP delta, where SVR-MCP is pooled
across all three tiers' result files (weighted by task count):

```
quantmcp cross-bench results/*/*.result.json --bfcl-results docs/bfcl_reference_svr.json
```

| Model | Quant | Δ SVR bfcl | Δ SVR-MCP (pooled U1+U2+U3) |
|---|---|---|---|
| Llama-3.2-1B | Q4_K_M | -0.047 | +0.188 |
| Llama-3.2-1B | Q5_K_M | -0.014 | +0.125 |
| Llama-3.2-1B | Q8_0 | -0.022 | 0.000 |
| Qwen3-0.6B | Q4_K_M | -0.004 | +0.094 |
| Qwen3-0.6B | Q5_K_M | +0.001 | -0.031 |
| Qwen3-0.6B | Q8_0 | +0.001 | -0.031 |

**CBC = -0.824 (n=6 pairs)**, strengthening (more negative) from Phase 2's
-0.720 now that U3 is included. This remains the opposite of a high,
significant positive rho — QuantCall's BFCL-measured degradation pattern
does **not** carry over to real MCP tool schemas for these two families at
this sample size. The correlation is driven almost entirely by
Llama-3.2-1B: BFCL shows its largest quantization drop at Q4_K_M, while our
MCP measurement shows its largest *increase* at Q4_K_M/Q5_K_M — which, per
the raw-output investigation in the family-contrast section above, is not
really "the model getting better at tool use under quantization" but a
shift between two different failure modes (unparseable schema-echo vs. a
flatter, sometimes-correct call shape).

**This should be read as a preliminary, directional finding, not a
statistically established one**: n=6 (model, quant) pairs is far too few
for a meaningful p-value or CI on a Spearman correlation, and each
family's MCP-side deltas are now pooled across 3 tiers × ~10-12 tasks —
more data than Phase 2, but still nowhere near enough for a CI on the
correlation itself. What can be stated honestly: the *direction and
magnitude* of degradation this project measured on real, unmodified MCP
schemas does not resemble QuantCall's BFCL-measured pattern for either
family, and the mechanism behind that mismatch (qualitative output-format
failure modes, not a smooth accuracy decline) was directly observed, not
inferred from the correlation number alone. Bootstrap CI on Δ itself
(spec §4.7, not yet on CBC) is still outstanding and would need multiple
seeds/repeats per (model, quant, tier), not just the single-seed runs done
so far.

### Schema Complexity Index vs. degradation (H2) — preliminary, 3 tiers only

H2 asks whether a tier's Schema Complexity Index (SCI, spec §4.3, computed
from each tier's live tool schemas) predicts how much quantization
degradation that tier shows. Real SCI was computed across all 30 tools
from the three live tiers (filesystem, git, sqlite) in a single
z-normalized corpus:

| Tier | Tools | Mean SCI | Mean \|Δ SVR-MCP\| across quants (both models) |
|---|---|---|---|
| filesystem (U1) | 14 | +0.206 | 0.083 |
| git (U2) | 12 | +0.027 | 0.083 |
| sqlite (U3) | 4 | -0.615 | 0.100 |

At face value this runs **opposite** to H2's hypothesis: sqlite has the
*lowest* schema complexity (its tools take one or two flat string
arguments each, no nesting, no unions) but the *largest* degradation
swing, while filesystem has the *highest* schema complexity but a
middling swing. **This is not treated as evidence against H2** — with only
3 tiers, a correlation coefficient over 3 points is not meaningful in
either direction, and is not reported as a rho for that reason (spec §4.3
calls for a regression once "enough tool schemas across tiers" exist; 3 is
not enough).

What is worth stating plainly, though, is a real methodological
observation surfaced by this attempt: SCI as defined (spec §4.3) measures
*schema shape* complexity (nesting depth, property count, unions,
description length) — it does not, and by construction cannot, measure the
difficulty of composing a correct *argument value* for a schema-simple
field. sqlite's `query: string` parameter is trivially simple by every SCI
component, yet writing a correct SQL query for a natural-language question
is exactly where both models struggled (hallucinated table/column names,
outright refusals). If a future tier's degradation really is driven by
argument-content difficulty rather than schema shape, SCI alone won't
capture it — worth flagging before Phase 6's U4 `memory` tier (spec's
designated "best stress test of H2") is added, since a 4th tier still
won't fully resolve this with only 4 points, but the qualitative caveat
should be carried forward regardless of sample size.

### The SVR-vs-TSR gap (H4)

Across every (model, quant, tier) combination measured so far, TSR is never
higher than SVR-MCP, and is often meaningfully lower (e.g. Llama-3.2-1B
Q4_K_M on U2: SVR-MCP=0.600 but TSR=0.200 — 3 of the 6 schema-valid calls
still failed to produce the *correct* outcome; Qwen3-0.6B Q4_K_M on U3:
SVR-MCP=0.800 but TSR=0.500 — 3 of 8 schema-valid calls used a
syntactically fine but semantically wrong query). This confirms the
expected execution gap: passing schema validation is necessary but not
sufficient for a tool call to actually do what was asked, and that gap is
not uniform across quants, model families, or tiers, so a leaderboard
built on SVR-MCP alone would overstate real-world reliability — the sqlite
tier makes this most visible of the three.

## Reference server versions used

- `@modelcontextprotocol/server-filesystem` — version `0.6.3` per its
  `package.json` at the pinned check (2026-07-07) against
  `github.com/modelcontextprotocol/servers`; the live server reports its own
  MCP `serverInfo.version` as `0.2.0` (the MCP protocol/server version, not
  the npm package version — both are legitimate, distinct version strings,
  and the manifest records the one the live server actually reports).
- `mcp-server-git` — version `0.6.2` per its `pyproject.toml` at commit
  `d31124c` (2026-07-06) against `github.com/modelcontextprotocol/servers`;
  launched via `uvx mcp-server-git --repository <sandbox_root>`.
- `sqlite` is **not** present in the current reference servers list (checked
  the same day) — Phase 3's U3 tier will need the documented fallback (a
  minimal self-written FastMCP wrapper), per the scope note this project's
  spec already anticipated.
