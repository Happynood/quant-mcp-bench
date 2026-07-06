"""Loads task fixtures (YAML) per server (spec §6.2)."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from quantmcp.tasks.base import MCPTaskInstance
from quantmcp.tasks.checkers import CHECKERS


def load_tasks(path: str | Path) -> list[MCPTaskInstance]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    tasks: list[MCPTaskInstance] = []
    for entry in data.get("tasks", []):
        checker_spec = entry["checker"]
        checker_fn = CHECKERS[checker_spec["name"]]
        checker_args = checker_spec.get("args", {})
        bound_checker = functools.partial(checker_fn, **checker_args)
        tasks.append(
            MCPTaskInstance(
                id=entry["id"],
                instruction=entry["instruction"],
                checker=bound_checker,
                expects_call=entry.get("expects_call", True),
                fixture_subdir=entry.get("fixture_subdir"),
            )
        )
    return tasks
