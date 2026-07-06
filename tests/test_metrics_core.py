from __future__ import annotations

from quantmcp.execution.dispatcher import ExecutionResult
from quantmcp.metrics.core import (
    InstanceOutcome,
    compute_metrics,
    evaluate_svr_mcp,
    evaluate_tsr,
)
from quantmcp.parsing.base import ParsedCall
from quantmcp.tasks.base import MCPTaskInstance, SandboxState


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }


def test_svr_mcp_fails_when_parse_failed():
    assert evaluate_svr_mcp([], False, {}) is False


def test_svr_mcp_fails_when_no_calls():
    assert evaluate_svr_mcp([], True, {}) is False


def test_svr_mcp_fails_unknown_tool():
    calls = [ParsedCall(name="unknown", arguments={})]
    assert evaluate_svr_mcp(calls, True, {"add": _schema()}) is False


def test_svr_mcp_fails_invalid_args():
    calls = [ParsedCall(name="add", arguments={"a": 1})]
    assert evaluate_svr_mcp(calls, True, {"add": _schema()}) is False


def test_svr_mcp_passes():
    calls = [ParsedCall(name="add", arguments={"a": 1, "b": 2})]
    assert evaluate_svr_mcp(calls, True, {"add": _schema()}) is True


def test_evaluate_tsr_delegates_to_checker(tmp_path):
    task = MCPTaskInstance(id="t1", instruction="do it", checker=lambda state: True)
    state = SandboxState(root=tmp_path, results=[])
    assert evaluate_tsr(task, state) is True

    task_false = MCPTaskInstance(id="t2", instruction="do it", checker=lambda state: False)
    assert evaluate_tsr(task_false, state) is False


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m.n == 0
    assert m.svr_mcp == 0.0
    assert m.tsr == 0.0


def test_compute_metrics_mixed():
    outcomes = [
        InstanceOutcome(task_id="a", parse_succeeded=True, svr_pass=True, tsr_pass=True),
        InstanceOutcome(task_id="b", parse_succeeded=True, svr_pass=True, tsr_pass=False),
        InstanceOutcome(task_id="c", parse_succeeded=False, svr_pass=False, tsr_pass=False),
        InstanceOutcome(task_id="d", parse_succeeded=True, svr_pass=False, tsr_pass=False),
    ]
    m = compute_metrics(outcomes)
    assert m.n == 4
    assert abs(m.svr_mcp - 0.5) < 1e-9
    assert abs(m.tsr - 0.25) < 1e-9


def test_compute_metrics_bootstrap_ci_brackets_the_point_estimate():
    outcomes = [
        InstanceOutcome(task_id=str(i), parse_succeeded=True, svr_pass=(i % 2 == 0), tsr_pass=True)
        for i in range(20)
    ]
    m = compute_metrics(outcomes, bootstrap_seed=0)
    lo, hi = m.svr_mcp_ci
    assert lo <= m.svr_mcp <= hi
    assert m.tsr_ci == (1.0, 1.0)


def test_compute_metrics_ci_is_reproducible_with_fixed_seed():
    outcomes = [
        InstanceOutcome(task_id=str(i), parse_succeeded=True, svr_pass=(i % 3 == 0), tsr_pass=True)
        for i in range(15)
    ]
    m1 = compute_metrics(outcomes, bootstrap_seed=7)
    m2 = compute_metrics(outcomes, bootstrap_seed=7)
    assert m1.svr_mcp_ci == m2.svr_mcp_ci


def test_execution_result_dataclass_shape():
    call = ParsedCall(name="add", arguments={"a": 1, "b": 2})
    result = ExecutionResult(call=call, ok=True, raw_result={"content": []})
    assert result.call.name == "add"
    assert result.ok is True
    assert result.error is None
