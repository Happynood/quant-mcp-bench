# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantmcp.report.published import (
    LEADERBOARD_COLS,
    RUNS_COLS,
    aggregate_leaderboard,
    run_row,
    write_csv,
)
from quantmcp.report.tables import _md_table


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if "svr" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return results


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_leaderboard(
    results_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build the published leaderboard from a directory of result.json files.

    Writes two files that make up the public schema (see docs/RESULTS_SCHEMA.md):
    runs.csv (one row per real run) and leaderboard.csv (aggregated over seeds,
    with bootstrap CIs and deltas vs the best-available baseline quant).
    """
    results_dir = Path(results_dir)
    results = _load_results(results_dir)

    run_rows = [run_row(r) for r in results]
    leaderboard_rows = aggregate_leaderboard(run_rows)

    leaderboard: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "columns": LEADERBOARD_COLS,
        "rows": leaderboard_rows,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2, default=str))

        write_csv(out / "runs.csv", RUNS_COLS, run_rows)
        write_csv(out / "leaderboard.csv", LEADERBOARD_COLS, leaderboard_rows)

        if leaderboard_rows:
            md_rows = [[_fmt(row.get(c)) for c in LEADERBOARD_COLS] for row in leaderboard_rows]
            md = _md_table(LEADERBOARD_COLS, md_rows)
        else:
            md = "No results yet."
        (out / "leaderboard.md").write_text(f"# Leaderboard\n\n{md}\n")

    return leaderboard
