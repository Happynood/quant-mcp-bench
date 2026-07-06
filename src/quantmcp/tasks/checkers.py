"""Named, YAML-referenceable checker functions for MCPTaskInstance.checker.

Task fixtures (YAML) can't hold Python callables directly, so tasks/loader.py
resolves a `{name: ..., args: {...}}` block into `functools.partial(CHECKERS[name], **args)`.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from quantmcp.tasks.base import SandboxState


def file_contains(state: SandboxState, filename: str, contains: str) -> bool:
    """Pass if `filename` exists under the sandbox root and contains `contains`."""
    path = state.root / filename
    return path.exists() and contains in path.read_text()


def call_result_number_equals(state: SandboxState, index: int, value: float) -> bool:
    """Pass if the `index`-th executed call succeeded and its text content
    parses as a float equal to `value` (e.g. the U0 `add` tool's return)."""
    if index >= len(state.results):
        return False
    result = state.results[index]
    if not result.ok:
        return False
    content = getattr(result.raw_result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if text is None:
            continue
        try:
            if float(text.strip()) == value:
                return True
        except ValueError:
            continue
    return False


def call_result_contains(state: SandboxState, index: int, contains: str) -> bool:
    """Pass if the `index`-th executed call succeeded and its text content
    contains `contains` — for read-only tools (list_directory, search_files,
    read_text_file, ...) whose effect is only visible in the returned
    content, not in sandbox filesystem state."""
    if index >= len(state.results):
        return False
    result = state.results[index]
    if not result.ok:
        return False
    content = getattr(result.raw_result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None and contains in text:
            return True
    return False


def dir_exists(state: SandboxState, dirname: str) -> bool:
    """Pass if `dirname` exists as a directory under the sandbox root."""
    path = state.root / dirname
    return path.is_dir()


def sqlite_query_scalar(state: SandboxState, db_filename: str, query: str, expected: str) -> bool:
    """Pass if `query` against the sandboxed sqlite db at `db_filename`
    returns a single row whose first column, stringified, equals `expected`.

    Unlike `call_result_contains` (which only inspects the tool's own return
    text), this inspects the database's actual post-execution state
    directly — needed for write_query tasks, whose success text ("N row(s)
    affected") doesn't reveal whether the *correct* row was actually
    changed.
    """
    path = state.root / db_filename
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(query).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    return str(row[0]) == expected


CHECKERS: dict[str, Any] = {
    "file_contains": file_contains,
    "call_result_number_equals": call_result_number_equals,
    "call_result_contains": call_result_contains,
    "dir_exists": dir_exists,
    "sqlite_query_scalar": sqlite_query_scalar,
}
