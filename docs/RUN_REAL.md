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

## What has actually been run (Phase 1)

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

| Quant | SVR-MCP | TSR | Peak VRAM (GB) |
|---|---|---|---|
| fp16 (bf16.gguf) | 1.000 | 0.917 | 1.995 |
| Q8_0 | 0.917 | 0.917 | 1.474 |
| Q5_K_M | 0.917 | 0.917 | 1.292 |
| Q4_K_M | 1.000 | 0.917 | 1.247 |

Raw results + manifests: `results/qwen3-0.6b-u1/*.result.json` /
`*.manifest.json` (git commit, config hash, fixture hash, MCP server
package version `0.2.0` for `@modelcontextprotocol/server-filesystem`,
hardware fingerprint — all captured automatically, not hand-entered).

### Honest scope limitations of this first result

1. **n=12, single seed, single repeat.** This is a plumbing-proving MVP
   result (spec's Phase 1 goal), not a statistically powered sweep. No
   bootstrap CI is reported here for that reason — at n=12 a CI would be
   too wide to say anything the raw numbers don't already say. Phase 3 adds
   bootstrap CI across a larger task set and multiple seeds/repeats.
2. **Not directly comparable to the published QuantCall BFCL SVR numbers.**
   QuantCall's BFCL-based SVR for Qwen3-0.6B is fp16=0.877, Q8_0=0.878,
   Q5_K_M=0.878, Q4_K_M=0.873 (free decoding, T1+T6) — noticeably lower than
   the SVR-MCP numbers above. This was investigated before being written
   here (per the cross-check requirement): it is **not a harness bug**. It
   reflects a genuine task-design difference, not schema realism alone:
   BFCL's natural-language queries never name the function to call, so SVR
   there also captures *tool selection* difficulty. Every one of this
   project's 12 U1 tasks explicitly says "use the `<tool_name>` tool," which
   removes tool selection and leaves only argument-construction — a
   substantially easier structural-validity bar. This was confirmed by
   manually driving the live `filesystem` server with the exact intended
   call for all 12 tasks (`tests/test_servers_filesystem.py`) and by
   observing run-to-run GPU decode variance directly (a repeated Q4_K_M run
   produced 12/12 passing where the recorded sweep run got 11/12 — greedy
   decoding on GPU is not perfectly bit-deterministic across process
   invocations due to non-associative floating-point reduction order in
   parallel kernels; this is expected, not a bug, and is exactly why
   multiple seeds/repeats and bootstrap CI matter for any claim stronger
   than "plumbing works").
   **Implication for H1/CBC (Phase 2):** the Cross-Benchmark Consistency
   comparison against QuantCall's numbers needs either naturalistic (tool
   name withheld) task instructions to be a fair difficulty match, or an
   explicit disclosure that SVR-MCP as measured here is not the same
   difficulty axis as BFCL's SVR. This is flagged now as the first concrete
   design decision Phase 2 needs to make before computing CBC for real.
3. **TSR is currently execution-success only for this tool corpus** — all
   12 tasks are single-call, no multi-step chaining yet (matches spec's
   "single/few-step, cheap-to-execute tasks" scope for tractability on a
   4 GB GPU).

## Reference server versions used

- `@modelcontextprotocol/server-filesystem` — version `0.6.3` per its
  `package.json` at the pinned check (2026-07-07) against
  `github.com/modelcontextprotocol/servers`; the live server reports its own
  MCP `serverInfo.version` as `0.2.0` (the MCP protocol/server version, not
  the npm package version — both are legitimate, distinct version strings,
  and the manifest records the one the live server actually reports).
- `sqlite` is **not** present in the current reference servers list (checked
  the same day) — Phase 3's U3 tier will need the documented fallback (a
  minimal self-written FastMCP wrapper), per the scope note this project's
  spec already anticipated.
