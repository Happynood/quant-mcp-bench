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
