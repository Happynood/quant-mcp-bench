from __future__ import annotations

from pathlib import Path

from quantmcp.execution.dispatcher import ExecutionResult
from quantmcp.parsing.base import ParsedCall
from quantmcp.tasks.base import SandboxState
from quantmcp.tasks.loader import load_tasks

U0_TASKS_FILE = (
    Path(__file__).parent.parent / "src" / "quantmcp" / "tasks" / "fixtures" / "u0_tasks.yaml"
)


def test_load_u0_tasks():
    tasks = load_tasks(U0_TASKS_FILE)
    assert len(tasks) == 2
    ids = {t.id for t in tasks}
    assert ids == {"u0-add-basic", "u0-write-note"}


def test_loaded_add_checker_passes_on_matching_result(tmp_path):
    tasks = {t.id: t for t in load_tasks(U0_TASKS_FILE)}
    task = tasks["u0-add-basic"]
    call = ParsedCall(name="add", arguments={"a": 2, "b": 3})

    class _Content:
        text = "5.0"

    class _Raw:
        content = [_Content()]

    result = ExecutionResult(call=call, ok=True, raw_result=_Raw())
    state = SandboxState(root=tmp_path, results=[result])
    assert task.checker(state) is True


def test_loaded_write_note_checker(tmp_path):
    tasks = {t.id: t for t in load_tasks(U0_TASKS_FILE)}
    task = tasks["u0-write-note"]
    (tmp_path / "note.txt").write_text("hello mcp world")
    state = SandboxState(root=tmp_path, results=[])
    assert task.checker(state) is True

    (tmp_path / "note.txt").write_text("something else")
    assert task.checker(state) is False
