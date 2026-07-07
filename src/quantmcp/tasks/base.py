"""MCPTaskInstance: {instruction, server_config, checker, expects_call} (spec §6.2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from quantmcp.execution.dispatcher import ExecutionResult


@dataclass(frozen=True)
class SandboxState:
    """What a task's checker gets to inspect after execution: the sandboxed
    fixture directory's post-execution state, plus every call's execution
    result (so checkers can validate a returned value, e.g. `add`'s sum,
    not just filesystem side effects)."""

    root: Path
    results: list[ExecutionResult] = field(default_factory=list)


CheckerFn = Callable[[SandboxState], bool]


@dataclass(frozen=True)
class MCPTaskInstance:
    id: str
    instruction: str
    checker: CheckerFn
    expects_call: bool = True
    fixture_subdir: str | None = None
    # The one tool this task is designed to exercise (declared, not inferred,
    # since a task's instruction deliberately never names it) -- lets a
    # per-instance result be attributed to a specific tool's SCI for the
    # H2 regression (spec §4.3), without needing to guess from the model's
    # actual (possibly wrong) parsed call.
    tool: str | None = None
