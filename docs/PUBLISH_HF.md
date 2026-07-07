# Publishing to HuggingFace

This document describes how to create or re-create the HuggingFace artifacts
for QuantMCP. Mirrors `quant-toolcall-bench`'s own `docs/PUBLISH_HF.md` flow,
adapted for this project's MCP-native metrics (SVR-MCP/TSR/η/SCI/CBC) instead
of BFCL's (SVR/TSA/AC/FCR).

## Prerequisites

- `HF_TOKEN` environment variable set with a write-scope token for the
  `happynood` account (or run `hf auth login` interactively)
- The `hf` CLI installed (`pip install huggingface_hub[cli]`)
- SOCKS proxy issue: if `ALL_PROXY=socks://...` is set in your environment,
  unset it before running `hf` commands (httpx does not support SOCKS):
  ```bash
  unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  ```

## Artifacts

| Artifact | URL | Type |
|----------|-----|------|
| Eval suite | https://huggingface.co/datasets/happynood/quantmcp-suite | dataset |
| Results | https://huggingface.co/datasets/happynood/quantmcp-results | dataset |
| Leaderboard | https://huggingface.co/spaces/happynood/quantmcp-leaderboard | space |

## 1. Create repos (first time only)

```bash
unset ALL_PROXY all_proxy
hf repo create happynood/quantmcp-suite --type dataset
hf repo create happynood/quantmcp-results --type dataset
hf repo create happynood/quantmcp-leaderboard --type space --space-sdk gradio
```

## 2. Build and upload the suite

Unlike QuantCall, every QuantMCP task fixture is originated by this project
under MIT — there's no third-party license to gate, so the suite is uploaded
directly rather than as a manifest-only reference.

```bash
mkdir -p /tmp/qm-suite/tasks/{u1_filesystem,u2_git,u3_sqlite} /tmp/qm-suite/schemas
cp src/quantmcp/tasks/fixtures/u0_tasks.yaml /tmp/qm-suite/tasks/
cp src/quantmcp/tasks/fixtures/u1_filesystem_tasks.yaml /tmp/qm-suite/tasks/
cp -r src/quantmcp/tasks/fixtures/u1_filesystem/* /tmp/qm-suite/tasks/u1_filesystem/
cp src/quantmcp/tasks/fixtures/u2_git_tasks.yaml /tmp/qm-suite/tasks/
cp src/quantmcp/tasks/fixtures/u2_git/repo.tar.gz /tmp/qm-suite/tasks/u2_git/
cp src/quantmcp/tasks/fixtures/u3_sqlite_tasks.yaml /tmp/qm-suite/tasks/
cp src/quantmcp/tasks/fixtures/u3_sqlite/fixture.db /tmp/qm-suite/tasks/u3_sqlite/

# Frozen tool schemas — the exact corpus the SCI table in docs/RUN_REAL.md is
# computed from (14 filesystem + 12 git + 4 sqlite = 30 tools)
uv run quantmcp dump-schemas --tiers filesystem,git,sqlite \
    --output /tmp/qm-suite/schemas/tool_schemas.json

unset ALL_PROXY all_proxy
hf upload happynood/quantmcp-suite <dataset_card.md> README.md --repo-type dataset
hf upload happynood/quantmcp-suite /tmp/qm-suite/tasks tasks --repo-type dataset
hf upload happynood/quantmcp-suite /tmp/qm-suite/schemas schemas --repo-type dataset
```

## 3. Upload the results dataset

```bash
uv run quantmcp leaderboard results/ --output-dir /tmp/qm-leaderboard
uv run quantmcp cross-bench results/*/*.result.json \
    --bfcl-results docs/bfcl_reference_svr.json --output /tmp/qm-leaderboard/cbc.json

mkdir -p /tmp/qm-results/data
cp -r results /tmp/qm-results/data/raw_results
cp /tmp/qm-leaderboard/mcp_runs.csv /tmp/qm-leaderboard/mcp_tier_breakdown.csv \
    /tmp/qm-leaderboard/cbc.json /tmp/qm-results/data/

unset ALL_PROXY all_proxy
hf upload happynood/quantmcp-results <results_dataset_card.md> README.md --repo-type dataset
hf upload happynood/quantmcp-results /tmp/qm-results/data data --repo-type dataset
```

Every raw result/manifest file uses a portable `~/models/...` model path
(never a specific machine's absolute home directory — `results/*.result.json`
committed to GitHub are already sanitized this way; see `docs/RUN_REAL.md`).
`mcp_runs.csv` additionally passes each `model` value through
`report/published.py::sanitize_model_name` to collapse local GGUF filenames
to a canonical model name.

## 4. Upload / update the Gradio Space

The Space's `app.py`/`README.md`/`requirements.txt` are not committed to
this GitHub repo (same convention as `quant-toolcall-bench`) — they live
only in the Space's own HF git history. Re-derive `app.py` from the live
Space if you need to edit it:

```bash
unset ALL_PROXY all_proxy
hf download happynood/quantmcp-leaderboard --repo-type space --local-dir /tmp/qm-space-live
```

Then upload after editing:

```bash
unset ALL_PROXY all_proxy
hf upload happynood/quantmcp-leaderboard README.md README.md --repo-type space
hf upload happynood/quantmcp-leaderboard requirements.txt requirements.txt --repo-type space
hf upload happynood/quantmcp-leaderboard app.py app.py --repo-type space
```

**Critical: Python version pin and short_description length**

- The Space README YAML must contain `python_version: "3.12"` — `audioop`
  was removed from the stdlib in 3.13 (PEP 594) and some Gradio audio
  sub-dependencies still import it. Don't bump until upstream fixes that.
- `short_description` in the README YAML must be **60 characters or fewer**
  or the upload is rejected outright.

## 5. After updating the Space

Check build/run status and logs:

```bash
unset ALL_PROXY all_proxy
hf spaces info happynood/quantmcp-leaderboard
hf spaces logs happynood/quantmcp-leaderboard -n 60
```

Look for a clean startup (`Running on local URL: http://0.0.0.0:7860`) and
confirm the three result files (`mcp_runs.csv`, `cbc.json`,
`mcp_tier_breakdown.csv`) download successfully from the results dataset. No
`ModuleNotFoundError: audioop` and no `ValueError: Unknown scheme for proxy
URL`.
