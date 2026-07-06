"""SVR-MCP and TSR — the only genuinely new metric code (spec §4.1, §4.2).

Everything else in metrics/ (deltas.py, stats.py) is vendored verbatim from
quant-toolcall-bench, since the delta/CI math doesn't know or care where the
underlying SVR/TSR numbers came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantmcp.metrics.stats import bootstrap_ci
from quantmcp.parsing.base import ParsedCall
from quantmcp.tasks.base import MCPTaskInstance, SandboxState
from quantmcp.validation.schema_validator import validate_call


def evaluate_svr_mcp(
    parsed_calls: list[ParsedCall],
    parse_succeeded: bool,
    tool_schemas: dict[str, dict[str, Any]],
) -> bool:
    """SVR-MCP per spec §4.1: parse succeeds AND every predicted call names a
    live tool AND validates against that tool's live (unmodified) schema."""
    if not parse_succeeded or not parsed_calls:
        return False
    for call in parsed_calls:
        schema = tool_schemas.get(call.name)
        if schema is None or not validate_call(call, schema):
            return False
    return True


def evaluate_tsr(task: MCPTaskInstance, sandbox_state: SandboxState) -> bool:
    """TSR per spec §4.2: the task's checker judges post-execution state."""
    return bool(task.checker(sandbox_state))


@dataclass(frozen=True)
class InstanceOutcome:
    task_id: str
    parse_succeeded: bool
    svr_pass: bool
    tsr_pass: bool


@dataclass(frozen=True)
class MetricsResult:
    n: int
    svr_mcp: float
    tsr: float
    svr_mcp_ci: tuple[float, float] = (0.0, 0.0)
    tsr_ci: tuple[float, float] = (0.0, 0.0)


def compute_metrics(
    outcomes: list[InstanceOutcome], bootstrap_seed: int | None = 0
) -> MetricsResult:
    """Compute SVR-MCP/TSR plus a bootstrap 95% CI on each (spec §4.7).

    The bootstrap resamples this run's own per-task pass/fail outcomes
    (with replacement) — it does not require repeat GPU runs, since the CI
    is over "what would this same model/quant's accuracy look like on a
    different draw of tasks from the same distribution," which is fully
    determined by a single run's n binary outcomes.
    """
    n = len(outcomes)
    if n == 0:
        return MetricsResult(n=0, svr_mcp=0.0, tsr=0.0)
    svr_values = [1.0 if o.svr_pass else 0.0 for o in outcomes]
    tsr_values = [1.0 if o.tsr_pass else 0.0 for o in outcomes]
    svr = sum(svr_values) / n
    tsr = sum(tsr_values) / n
    svr_ci = bootstrap_ci(svr_values, seed=bootstrap_seed)
    tsr_ci = bootstrap_ci(tsr_values, seed=bootstrap_seed)
    return MetricsResult(n=n, svr_mcp=svr, tsr=tsr, svr_mcp_ci=svr_ci, tsr_ci=tsr_ci)
