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

**Note on the committed sweep configs** (`configs/*-sweep/*.yaml`): their
`model:` field uses a `~/models/...`-relative path rather than the specific
machine's absolute home directory — `QuantMCPConfig` expands `~` at load
time (`config.py`'s `_expand_home_tilde` validator), so these configs work
unchanged as long as you download the GGUF files to `~/models/` on your own
machine. If you use a different location, edit the path accordingly — the
configs are real, working examples of the harness, not a claim that this
exact layout exists on your machine.

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
   result (spec's Phase 1 goal), not a statistically powered sweep.
   Bootstrap 95% CI (spec §4.7) was added in Phase 3 — every `result.json`
   now carries `svr_mcp_ci`/`tsr_ci`, resampled from that run's own n
   per-task outcomes — but at n=12 those intervals are wide (e.g. this
   fp16 run's SVR-MCP 95% CI is [0.583, 1.000]), wide enough that they
   don't distinguish most of these quant levels from each other. That
   width is itself the honest signal that n=12/single-repeat isn't enough
   to make a precise claim, which is exactly what the CI is for.
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
| Qwen3-0.6B | Q8_0 | 1.000 | 1.000 |
| Qwen3-0.6B | Q5_K_M | 1.000 | 1.000 |
| Qwen3-0.6B | Q4_K_M | 1.000 | 0.900 |
| Llama-3.2-1B | fp16 | 0.600 | 0.200 |
| Llama-3.2-1B | Q8_0 | 0.400 | 0.200 |
| Llama-3.2-1B | Q5_K_M | 0.500 | 0.200 |
| Llama-3.2-1B | Q4_K_M | 0.600 | 0.300 |

Raw results + manifests: `results/{qwen3-0.6b,llama3.2-1b}-u2/*.result.json`.
Llama-3.2-1B doesn't show a monotonic decline here either — Q4_K_M matches
its own fp16 SVR-MCP rather than falling below it, consistent with the
same schema-echo-vs-flat-call failure-mode shift described above rather
than a filesystem-tier-specific artifact. (These exact numbers were
regenerated once, to populate the bootstrap CI fields added later in
Phase 3 — see the note on run-to-run variability in the CBC section below;
Qwen3-0.6B's numbers reproduced identically, Llama-3.2-1B's shifted by
1-2 tasks out of 10 between the two runs of the same seed=0 greedy config.)

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

## U4 memory tier (Phase 6, stretch)

Spec §10 lists U4 (the official reference `memory` knowledge-graph server)
as a stretch goal specifically because it has "the most nested/union-heavy
schemas — the best stress test of H2 (SCI)." Verified current at the npm
registry on 2026-07-07 (`@modelcontextprotocol/server-memory`, version
2026.7.4, still present). 10 naturalistic tasks against a small seed
knowledge graph (4 entities, 3 relations), covering all 9 real tools
(`create_entities`, `create_relations`, `add_observations`,
`delete_entities`, `delete_observations`, `delete_relations`,
`read_graph`, `search_nodes`, `open_nodes`).

| Model | Quant | SVR-MCP | TSR |
|---|---|---|---|
| Qwen3-0.6B | fp16 | 0.800 | 0.500 |
| Qwen3-0.6B | Q8_0 | 0.800 | 0.500 |
| Qwen3-0.6B | Q5_K_M | 0.800 | 0.700 |
| Qwen3-0.6B | Q4_K_M | 0.700 | 0.700 |
| Llama-3.2-1B | fp16 | 0.200 | 0.200 |
| Llama-3.2-1B | Q8_0 | 0.200 | 0.200 |
| Llama-3.2-1B | Q5_K_M | 0.200 | 0.200 |
| Llama-3.2-1B | Q4_K_M | 0.200 | 0.200 |

**Single run per config, not 3 repeats** — disclosed honestly as a scope
limit for this stretch tier rather than silently treated as equally
precise to the repeat-averaged U1-U3 numbers above.

Llama-3.2-1B is flat at 0.200 across all four quants — the lowest and
flattest of any tier so far. Raw model output was inspected directly
before trusting this (same cross-check discipline as every other tier).
It is the *same* schema-echo failure mode already documented for U1
(Phase 2), just more consistent here: e.g. asked to search the graph, fp16
answered
`{"type": "function", "function": {"name": "read_graph", "parameters": {"$schema": "...", "type": "object", "properties": {}}}}`
— an OpenAI-style tool-call envelope whose `parameters` field is the
tool's own JSON-Schema *definition*, not filled-in argument values. The
parser correctly does not accept this as a valid call (same reasoning as
Phase 2), and this is not a parser bug: several other U4 tasks *did* parse
into well-formed, closely-relevant calls from the same model (e.g.
`{"type": "function", "function": "search_nodes", "parameters": {"query": "Alice"}}`
for a task that expected `open_nodes` — a reasonable tool substitution
that still passed its checker). The schema-echo behavior specifically
concentrates on tasks needing array-of-object arguments
(`create_entities`, `create_relations`, `delete_relations`), i.e. exactly
the tool shapes spec §10 predicted would stress-test this the hardest.

This U4 data point also exposes a real limitation in the SCI metric as
currently implemented (`schema/complexity.py::_max_depth`): depth is
computed by recursing into a schema's `properties` dict, but JSON Schema
nests object structure inside array-typed properties via `items`, not
`properties` — so an array-of-objects argument (exactly memory's dominant
shape) contributes only depth 1, the same as a flat string argument.
Memory's schemas are the *most* deeply nested by inspection but score
among the *lowest* SCI values below for this reason — a second, distinct
methodological caveat alongside Phase 3's "SCI measures shape, not
argument-content difficulty" one, not a contradiction of it.

Raw results + manifests: `results/{qwen3-0.6b,llama3.2-1b}-u4/*.result.json`.

## Constrained decoding (GBNF, Phase 6 stretch)

Spec §10 makes this conditional: "constrained decoding on MCP schemas if
H3's family effect motivates it." It does — every tier measured so far
shows the same qualitative pattern (Qwen3-0.6B stable, Llama-3.2-1B
degraded/unreliable in free decoding), and QuantCall's own GBNF writeup
(`quant-toolcall-bench/docs/constrained_decoding_findings.md`) explicitly
flagged that its "no measurable benefit" finding was Qwen3-specific and
"may differ for models that are less reliable at free-form JSON generation
than Qwen3 turned out to be" — Llama-3.2-1B here is exactly that case, and
QuantCall never tested it. `decoding/gbnf.py`'s `build_tool_call_grammar`
was vendored back in Phase 1 but never exercised until now — this is the
first time it has been run against real MCP tool schemas at all (fixed:
zero test coverage before this pass; see `tests/test_gbnf.py`).

**Setup:** Llama-3.2-1B, U1 filesystem tier (14 tools, the same config
already measured in free decoding above), all 4 quants, `decoding:
constrained` in `configs/llama3.2-1b-u1-constrained-sweep/*.yaml`.

**Grammar generation works correctly against real MCP schemas** —
verified by loading the actual filesystem-tier tools live and building a
grammar from them (`tests/test_gbnf.py`'s new real-schema tests), then
confirming llama.cpp's own GBNF parser accepts it without crashing
(regression-tests the exact `_schema_rule_name` mixed-`_`/`-` segfault
QuantCall found and fixed, this time against a real array-of-objects MCP
schema, not just a synthetic one).

**The result: identical SVR-MCP/TSR to free decoding, substantially
slower.**

| Quant | SVR-MCP free → constrained | TSR free → constrained | Latency (total, 12 tasks) free → constrained |
|---|---|---|---|
| fp16 | 0.000 → 0.000 | 0.000 → 0.000 | 49.7s → 85.1s (+71%) |
| Q8_0 | 0.000 → 0.000 | 0.000 → 0.000 | 32.6s → 69.0s (+112%) |
| Q5_K_M | 0.250 → 0.250 | 0.250 → 0.250 | 29.9s → 55.9s (+87%) |
| Q4_K_M | 0.250 → 0.250 | 0.250 → 0.250 | 29.1s → 55.5s (+91%) |

Every SVR-MCP/TSR value is *exactly* identical, not just within noise —
this is a stronger version of QuantCall's own "no measurable benefit"
conclusion for Qwen3, now shown for a genuinely unreliable-in-free-decoding
model too, and latency cost (71-112% slower) is in the same range as
QuantCall's measured 55-89% for their smaller model.

**Why the grammar didn't help — investigated directly, not assumed.**
`build_tool_call_grammar`'s root rule is `tool-call-path | no-call`, where
`no-call ::= .*` (unconstrained free text) — a deliberate design so a
model can correctly abstain instead of being forced into a spurious call
(this exact design fixed a real Abstention-collapse bug in QuantCall's own
history, see their findings doc's "What was wrong before" section). But
`.*` matches *anything*, including a malformed near-JSON attempt: if the
model's own token probabilities don't favor starting with the literal
`<tool_call>` tag, the constrained sampler can legally continue down the
unconstrained `no-call` branch instead — which is exactly what happened
here. A raw-output check under constrained decoding showed the *same*
schema-echoing text already documented for free decoding
(`{"type": "function", "function": {"name": "read_file", "parameters": {"$schema": ...`),
produced entirely inside the grammar's own escape hatch, not despite it.

To confirm this diagnosis rather than assume it, the escape hatch was
removed for one manual test (`build_tool_call_grammar(tools,
allow_no_call=False)`, forcing `root ::= tool-call-path` unconditionally).
Result: the model got stuck emitting whitespace immediately after
`<tool_call>` — `'<tool_call> \n                    ...'` padded out to
`max_tokens` — never producing the literal `{` needed to proceed into the
grammar's object body. This means the model's own weights don't have
strong probability mass on continuing correctly even when structurally
forced into the right envelope; constrained decoding can restrict *which*
tokens are legal, but it cannot manufacture a correct continuation the
model doesn't have any real preference for. This is a materially different
mechanism from QuantCall's Qwen3 result (already reliable, nothing to
recover) and is, as far as this project is aware, a novel, concrete
illustration of *why* grammar constraints don't help a genuinely unreliable
small model on tool-shaped generation: forcing conformance surfaces
decoding degeneracy instead of hidden correct output.

**Honest conclusion, matching QuantCall's own framing:** constrained
decoding gave zero measured benefit and substantial latency cost for
Llama-3.2-1B on this tier — not because the model was already reliable
(QuantCall's reason for Qwen3), but because its underlying weights don't
reliably know how to complete a tool call at all, with or without a
grammar forcing the envelope. Single-tier, single-repeat scope (not the
full 4-tier, 3-repeat treatment given to free decoding) — a real Phase 6
stretch result, not a fully resourced study.

Raw results + manifests: `results/llama3.2-1b-u1-constrained/*.result.json`.

## Repeat-run stability: 3 independent runs per config

Every table above shows a *single* run per (model, quant, tier). Given the
already-documented GPU decode non-determinism, and to have an honest
answer to "how much does a single-run point estimate actually move
around," every one of the 24 configs above was re-run 2 more times
(`*.rep2.result.json`, `*.rep3.result.json` — same config, same seed,
same greedy/temperature=0 settings; only wall-clock time separates the 3
runs). This is the most reliable set of numbers in this document — where
it disagrees with a table above, trust this section.

| Model | Tier | Quant | SVR-MCP mean [min, max] | TSR mean [min, max] |
|---|---|---|---|---|
| Qwen3-0.6B | filesystem | fp16 | 0.833 [0.833, 0.833] | 0.694 [0.583, 0.750] |
| Qwen3-0.6B | filesystem | Q8_0 | 0.833 [0.833, 0.833] | 0.667 [0.667, 0.667] |
| Qwen3-0.6B | filesystem | Q5_K_M | 0.806 [0.750, 0.833] | 0.694 [0.667, 0.750] |
| Qwen3-0.6B | filesystem | Q4_K_M | 0.889 [0.833, 0.917] | 0.694 [0.583, 0.750] |
| Qwen3-0.6B | git | fp16 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Qwen3-0.6B | git | Q8_0 | 1.000 [1.000, 1.000] | 0.967 [0.900, 1.000] |
| Qwen3-0.6B | git | Q5_K_M | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Qwen3-0.6B | git | Q4_K_M | 0.967 [0.900, 1.000] | 0.867 [0.800, 0.900] |
| Qwen3-0.6B | sqlite | fp16 | 0.500 [0.500, 0.500] | 0.400 [0.400, 0.400] |
| Qwen3-0.6B | sqlite | Q8_0 | 0.500 [0.500, 0.500] | 0.300 [0.300, 0.300] |
| Qwen3-0.6B | sqlite | Q5_K_M | 0.400 [0.400, 0.400] | 0.400 [0.400, 0.400] |
| Qwen3-0.6B | sqlite | Q4_K_M | 0.800 [0.800, 0.800] | 0.500 [0.500, 0.500] |
| Llama-3.2-1B | filesystem | fp16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Llama-3.2-1B | filesystem | Q8_0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Llama-3.2-1B | filesystem | Q5_K_M | 0.222 [0.167, 0.250] | 0.222 [0.167, 0.250] |
| Llama-3.2-1B | filesystem | Q4_K_M | 0.250 [0.250, 0.250] | 0.250 [0.250, 0.250] |
| Llama-3.2-1B | git | fp16 | 0.500 [0.400, 0.600] | 0.200 [0.200, 0.200] |
| Llama-3.2-1B | git | Q8_0 | 0.400 [0.400, 0.400] | 0.233 [0.200, 0.300] |
| Llama-3.2-1B | git | Q5_K_M | 0.533 [0.500, 0.600] | 0.233 [0.200, 0.300] |
| Llama-3.2-1B | git | Q4_K_M | 0.600 [0.600, 0.600] | 0.233 [0.200, 0.300] |
| Llama-3.2-1B | sqlite | fp16 | 0.200 [0.200, 0.200] | 0.100 [0.100, 0.100] |
| Llama-3.2-1B | sqlite | Q8_0 | 0.200 [0.200, 0.200] | 0.100 [0.100, 0.100] |
| Llama-3.2-1B | sqlite | Q5_K_M | 0.100 [0.100, 0.100] | 0.100 [0.100, 0.100] |
| Llama-3.2-1B | sqlite | Q4_K_M | 0.300 [0.300, 0.300] | 0.100 [0.100, 0.100] |

What this shows:

1. **Most rows are perfectly stable across 3 runs** (min == max) — mostly
   the extreme rows (0.000, 1.000, or a very confidently-wrong/right
   quant) where the model's behavior doesn't sit near a decision boundary.
2. **The rows that do move are exactly the ones that matter most for any
   quantization-degradation claim**: Qwen3-0.6B filesystem Q5_K_M/Q4_K_M
   (the two quants closest to fp16's own score), Qwen3-0.6B git Q4_K_M,
   Llama-3.2-1B filesystem Q5_K_M, and Llama-3.2-1B git (all 4 quants show
   some spread). These are precisely the comparisons this project's core
   hypothesis depends on, and single-run point estimates for them should
   not be trusted at n=10-12 without this range attached.
3. **No row's range crosses from "clearly degrading" to "clearly
   improving"** — the qualitative findings above (Llama-3.2-1B's schema-
   echo-vs-flat-call shift, the sqlite hallucination/refusal modes) hold
   up across all 3 runs, even where the exact point estimate doesn't.

## Cross-Benchmark Consistency (CBC, spec §4.5) and the SVR-vs-TSR gap (H4)

CBC asks: does QuantCall's BFCL-measured quantization degradation pattern
predict what happens on real MCP schemas? It is computed as the Spearman
correlation between each (model, quant) pair's BFCL SVR delta (vs. that
model's own fp16 baseline) and its SVR-MCP delta, where SVR-MCP is pooled
across every result file supplied (weighted by task count):

```
quantmcp cross-bench results/*/*.result.json --bfcl-results docs/bfcl_reference_svr.json
```

**CBC was computed three times as this project's own methodology matured,
and the three numbers are reported together deliberately, not just the
final one** — the convergence itself is evidence about how much to trust
a single-run point estimate at this task-set size:

| Computation | Data behind it | CBC (Spearman rho) |
|---|---|---|
| First (Phase 2) | 1 run per (model, quant, tier) | -0.824 |
| Second (Phase 3, bootstrap-CI re-run) | 1 run per config, re-run once | -0.265 |
| **Third (this pass, 3 repeats/config)** | **mean of 3 runs per config** | **-0.551** |

All three used the identical configs, seeds, and greedy/temperature=0
settings — only the number of independent runs pooled into each point
estimate changed. **The sign (negative) has been stable across all three;
the magnitude swung by more than 3x (-0.824 to -0.265) between the first
two single-run computations, then landed at -0.551 once each point
estimate was itself an average of 3 runs rather than 1.** This is the
single clearest piece of evidence in this document that CBC — computed as
a correlation of deltas of deltas, from tasks sets of only 10-12 items —
amplifies single-run GPU decode noise, and that the 3-repeat mean is the
number to trust going forward, not either single-run value.

Using the 3-repeat means for U1-U3 (see "Repeat-run stability" above) plus
U4's single run (see "U4 memory tier" above — disclosed as single-repeat,
not equally precise) as the SVR-MCP input, pooled by task count:

| Model | Quant | Δ SVR bfcl | Δ SVR-MCP (pooled U1+U2+U3+U4) |
|---|---|---|---|
| Llama-3.2-1B | Q4_K_M | -0.047 | +0.142 |
| Llama-3.2-1B | Q5_K_M | -0.014 | +0.057 |
| Llama-3.2-1B | Q8_0 | -0.022 | -0.028 |
| Qwen3-0.6B | Q4_K_M | -0.004 | +0.085 |
| Qwen3-0.6B | Q5_K_M | +0.001 | -0.038 |
| Qwen3-0.6B | Q8_0 | +0.001 | 0.000 |

Adding U4 shifted every pooled delta slightly (U4 contributes 10 of each
model's ~42 pooled tasks) but **did not change the rank ordering of the 6
pairs, so CBC is unchanged: -0.551 (n=6 pairs)** — a useful stability
check in its own right: the headline number is not sensitive to adding one
more single-repeat tier into an otherwise 3-repeat-averaged pool.

**CBC = -0.551 (n=6 pairs).** QuantCall's BFCL-measured degradation
pattern does not carry over cleanly to real MCP tool schemas for these
two families — the direction is consistent across every computation of
this number so far, even though the exact magnitude isn't.

**This should still be read as a preliminary, directional finding, not a
statistically established one**: n=6 (model, quant) pairs is far too few
for a meaningful p-value or CI on a Spearman correlation, even with 3x
more underlying task-instance data behind each point than the original
computation. Bootstrap CI is implemented on SVR-MCP/TSR themselves
(spec §4.7, `svr_mcp_ci`/`tsr_ci` in every `result.json`) but not yet
propagated through to a CI on CBC itself — that would need many more
repeats to build a real empirical distribution over Δ, not the 3 used
here (3 is enough to demonstrate the instability and get a noticeably
more stable point estimate; it is not enough for a rigorous CI).

### Schema Complexity Index vs. degradation (H2) — preliminary, 4 tiers now

H2 asks whether a tier's Schema Complexity Index (SCI, spec §4.3, computed
from each tier's live tool schemas) predicts how much quantization
degradation that tier shows. Real SCI was computed across all 39 tools
from all four live tiers (filesystem, git, sqlite, memory) in a single
z-normalized corpus (recomputed from scratch, not just extended, once U4
was added — the z-normalization is corpus-relative, so adding a 4th tier
shifts every tier's score slightly versus the original 3-tier numbers):

| Tier | Tools | Mean SCI | Mean \|Δ SVR-MCP\| across quants (both models) |
|---|---|---|---|
| filesystem (U1) | 14 | +0.333 | 0.093 (3-repeat mean) |
| git (U2) | 12 | +0.115 | 0.044 (3-repeat mean) |
| memory (U4) | 9 | -0.359 | 0.017 (single run) |
| sqlite (U3) | 4 | -0.515 | 0.100 (3-repeat mean) |

At face value this still runs **opposite** to H2's hypothesis, now with a
4th point that doesn't resolve it either way: memory has the *second-lowest*
SCI yet the *smallest* degradation swing of all four tiers, while sqlite
(lowest SCI) has the largest. **This is not treated as evidence against
H2** — 4 tiers is still nowhere near enough for a correlation coefficient
to mean anything (spec §4.3 calls for a regression once "enough tool
schemas across tiers" exist; 4 is not enough), and U4's number is a single
run, not a 3-repeat mean like the other three.

Two distinct methodological caveats now stand alongside each other:

1. **(Phase 3) SCI measures schema shape, not argument-content
   difficulty.** sqlite's `query: string` parameter is trivially simple by
   every SCI component, yet composing a correct SQL query for a natural-
   language question is exactly where both models struggled. This still
   explains sqlite's outsized degradation despite its low SCI.
2. **(Phase 6, new) `_max_depth`'s nesting metric doesn't traverse array
   items.** Memory's dominant tool shape is an *array of objects*
   (`entities: [{name, entityType, observations}, ...]`) — genuinely
   deeply nested by inspection, but `schema/complexity.py::_max_depth`
   only recurses into a schema's `properties` dict, and JSON Schema nests
   object structure inside array-typed properties via `items`, not
   `properties`. An array-of-objects argument therefore scores depth 1,
   identical to a flat string argument. This likely explains why memory's
   SCI came out low despite being the tier spec §5 specifically flagged as
   having "the most nested/union-heavy schemas." Not fixed in this pass —
   changing the depth formula mid-project would silently invalidate the
   already-published 3-tier SCI numbers without re-deriving them, and the
   spec's SCI formula (§4.3) doesn't specify array-traversal semantics —
   but disclosed here because it materially affects how memory's low SCI
   number should be read: as an artifact of the metric's array blind spot,
   not as evidence the tier is actually schema-simple.

### The SVR-vs-TSR gap (H4)

Across every (model, quant, tier) combination measured so far, TSR is never
higher than SVR-MCP, and is often meaningfully lower (e.g. Llama-3.2-1B
Q4_K_M on U2: SVR-MCP=0.600 but TSR=0.300 — 2 of the 6 schema-valid calls
still failed to produce the *correct* outcome; Qwen3-0.6B Q4_K_M on U3:
SVR-MCP=0.800 but TSR=0.500 — 3 of 8 schema-valid calls used a
syntactically fine but semantically wrong query). This confirms the
expected execution gap: passing schema validation is necessary but not
sufficient for a tool call to actually do what was asked, and that gap is
not uniform across quants, model families, or tiers, so a leaderboard
built on SVR-MCP alone would overstate real-world reliability — the sqlite
tier makes this most visible of the four (U4 memory shows the same
direction but a smaller gap, e.g. Qwen3-0.6B fp16 there: SVR-MCP=0.800,
TSR=0.500).

## Leaderboard and reliability-per-VRAM (η, spec §4.6)

```
quantmcp leaderboard results/ --output-dir leaderboard
```

Builds `leaderboard/mcp_leaderboard.{json,md}`, `mcp_runs.csv` (one row per
real result file, with η computed as `(0.5*SVR-MCP + 0.5*TSR) /
peak_VRAM_GB` — equal weighting, since the spec gives no other guidance),
and `mcp_tier_breakdown.csv` (mean SVR-MCP/TSR/η per tier, annotated with
that tier's real SCI from the table above). This is the MCP-native
leaderboard; the vendored, BFCL-shaped `report/leaderboard.py` is left
as-is per the reuse rule rather than retrofitted, since its svr/tsa/ac/fcr
columns have no server-tier concept to extend cleanly.

If `plotly` is installed (`uv sync --extra space` — the same optional
dependency the HF Space uses), the same command also writes
`leaderboard/pareto.html`: a self-contained scatter of reliability
(`0.5*SVR-MCP + 0.5*TSR`) against peak VRAM, with the Pareto-optimal
(model, quant, tier) configs marked with a star. Frontier membership
reuses the vendored `report/pareto.py` selector rather than a bespoke
implementation. Without `plotly` installed, `quantmcp leaderboard` still
runs and simply skips the chart (printed to stdout) — this keeps the
offline `make verify` gate free of a hard plotly dependency. `leaderboard/`
itself is gitignored (regenerate it from `results/` rather than committing
it — the chart alone is several MB, embedding plotly.js for offline
viewing).

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
- `@modelcontextprotocol/server-memory` — version `2026.7.4` per the npm
  registry, checked 2026-07-07; launched via `npx -y
  @modelcontextprotocol/server-memory` with `MEMORY_FILE_PATH` pointing at
  the per-instance sandbox root's `memory.json`. Still present and current,
  matching spec §5's table exactly (description: "official reference
  `memory` (knowledge-graph) server").
