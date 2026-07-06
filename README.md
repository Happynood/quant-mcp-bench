# QuantMCP

Does quantization survive real, unmodified MCP tool schemas — not just curated
benchmark schemas? QuantMCP re-runs the `quant-toolcall-bench` (QuantCall)
measurement methodology (schema-validity, execution success, quantization
delta, bootstrap CI) against live Model Context Protocol servers, executed
end-to-end against sandboxed fixtures rather than only structurally validated.

Status: early build. This README will grow into the full leaderboard-first
layout once the first real results land (see `docs/RUN_REAL.md`).

## Quickstart (mock backend, zero GPU)

```bash
uv sync
uv run quantmcp run --config configs/smoke.yaml
```

## License

MIT — see `LICENSE`. Citation: see `CITATION.cff`.
