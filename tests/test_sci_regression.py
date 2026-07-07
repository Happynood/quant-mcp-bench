from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantmcp.report.sci_regression import compute_sci_delta_regression

# Minimal live-schema-shaped fixtures: two tools with clearly different
# complexity (a flat 1-prop schema vs. a deeply nested array-of-objects one,
# the exact shape the Phase 7 SCI fix targets).
_SIMPLE_SCHEMA = {
    "tier": "t",
    "name": "simple_tool",
    "description": "",
    "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
}
_COMPLEX_SCHEMA = {
    "tier": "t",
    "name": "complex_tool",
    "description": "",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                },
            }
        },
    },
}


def _write_result(path: Path, quant: str, instances: list[tuple[str, bool]]) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {"quant": quant},
                "instances": [
                    {"task_id": f"t-{i}", "tool": tool, "svr_pass": passed, "tsr_pass": passed}
                    for i, (tool, passed) in enumerate(instances)
                ],
            }
        )
    )


def test_compute_sci_delta_regression_produces_one_point_per_tool(tmp_path: Path):
    fp16 = tmp_path / "fp16.result.json"
    q4 = tmp_path / "q4.result.json"
    _write_result(
        fp16, "fp16", [("simple_tool", True), ("simple_tool", True), ("complex_tool", True)]
    )
    _write_result(
        q4, "Q4_K_M", [("simple_tool", True), ("simple_tool", True), ("complex_tool", False)]
    )

    result = compute_sci_delta_regression([fp16, q4], [_SIMPLE_SCHEMA, _COMPLEX_SCHEMA])

    assert result.n == 2
    by_tool = {p.tool: p for p in result.points}
    assert by_tool["simple_tool"].delta_svr == pytest.approx(0.0)
    assert by_tool["complex_tool"].delta_svr == pytest.approx(1.0)
    # complex_tool has the higher SCI (deeper/more properties) and the
    # larger degradation -> a positive slope for this synthetic case.
    assert result.slope > 0


def test_compute_sci_delta_regression_pools_across_multiple_result_files(tmp_path: Path):
    # Two "models'" worth of fp16 files for the same tool, pooled by n.
    a = tmp_path / "a-fp16.result.json"
    b = tmp_path / "b-fp16.result.json"
    q4 = tmp_path / "q4.result.json"
    _write_result(a, "fp16", [("simple_tool", True)])
    _write_result(b, "fp16", [("simple_tool", False)])
    _write_result(q4, "Q4_K_M", [("simple_tool", True)])

    result = compute_sci_delta_regression([a, b, q4], [_SIMPLE_SCHEMA, _COMPLEX_SCHEMA])

    point = next(p for p in result.points if p.tool == "simple_tool")
    # pooled fp16 pass rate = 1/2 = 0.5; Q4_K_M pass rate = 1/1 = 1.0
    assert point.delta_svr == pytest.approx(0.5 - 1.0)
    assert point.n_baseline == 2
    assert point.n_quant == 1


def test_compute_sci_delta_regression_too_few_points_returns_zeroed_result(tmp_path: Path):
    fp16 = tmp_path / "fp16.result.json"
    q4 = tmp_path / "q4.result.json"
    _write_result(fp16, "fp16", [("simple_tool", True)])
    _write_result(q4, "Q4_K_M", [("simple_tool", True)])

    result = compute_sci_delta_regression([fp16, q4], [_SIMPLE_SCHEMA, _COMPLEX_SCHEMA])

    assert result.n == 1
    assert result.slope == 0.0
    assert result.slope_ci == (0.0, 0.0)


def test_compute_sci_delta_regression_ignores_tools_without_schema(tmp_path: Path):
    fp16 = tmp_path / "fp16.result.json"
    q4 = tmp_path / "q4.result.json"
    _write_result(fp16, "fp16", [("simple_tool", True), ("unknown_tool", True)])
    _write_result(q4, "Q4_K_M", [("simple_tool", True), ("unknown_tool", True)])

    result = compute_sci_delta_regression([fp16, q4], [_SIMPLE_SCHEMA, _COMPLEX_SCHEMA])

    assert all(p.tool != "unknown_tool" for p in result.points)
