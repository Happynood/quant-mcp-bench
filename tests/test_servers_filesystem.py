from __future__ import annotations

from pathlib import Path

import pytest

from quantmcp.execution.dispatcher import execute_calls
from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.parsing.base import ParsedCall
from quantmcp.servers.base import MCPServerHandle
from quantmcp.servers.filesystem import ARGS_TEMPLATE, COMMAND
from quantmcp.tasks.base import SandboxState
from quantmcp.tasks.loader import load_tasks

pytestmark = pytest.mark.integration

FIXTURE_DIR = (
    Path(__file__).parent.parent / "src" / "quantmcp" / "tasks" / "fixtures" / "u1_filesystem"
)
TASKS_FILE = (
    Path(__file__).parent.parent
    / "src"
    / "quantmcp"
    / "tasks"
    / "fixtures"
    / "u1_filesystem_tasks.yaml"
)


def _resolved_args(root: Path) -> list[str]:
    return [a.replace("{root}", str(root)) for a in ARGS_TEMPLATE]


@pytest.mark.asyncio
async def test_filesystem_server_lists_expected_tools():
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="fs-list-tools") as root:
        async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
            tools = await handle.list_tools()
            names = {t.name for t in tools}
            assert {"read_text_file", "write_file", "list_directory", "create_directory"} <= names


def test_load_u1_filesystem_tasks():
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) == 13


@pytest.mark.asyncio
async def test_u1_read_todo_task_end_to_end():
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    task = tasks["u1-read-todo"]
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="fs-read-todo") as root:
        instruction = task.instruction.format(root=str(root))
        assert str(root) in instruction
        async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
            call = ParsedCall(name="read_text_file", arguments={"path": f"{root}/notes/todo.txt"})
            results = await execute_calls(handle, [call])
            state = SandboxState(root=root, results=results)
            assert task.checker(state) is True


@pytest.mark.asyncio
async def test_u1_write_hello_task_end_to_end():
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    task = tasks["u1-write-hello"]
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="fs-write-hello") as root:
        async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
            call = ParsedCall(
                name="write_file",
                arguments={"path": f"{root}/hello.txt", "content": "hello world"},
            )
            results = await execute_calls(handle, [call])
            state = SandboxState(root=root, results=results)
            assert task.checker(state) is True


@pytest.mark.asyncio
async def test_u1_create_dir_task_end_to_end():
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    task = tasks["u1-create-dir"]
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="fs-create-dir") as root:
        async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
            call = ParsedCall(name="create_directory", arguments={"path": f"{root}/new_folder"})
            results = await execute_calls(handle, [call])
            state = SandboxState(root=root, results=results)
            assert task.checker(state) is True


@pytest.mark.asyncio
async def test_all_u1_tasks_pass_with_the_intended_call():
    """Drive every U1 task with the exact call it expects, proving each
    checker actually matches this server's real response shape (not a
    guess)."""
    tool_by_task = {
        "u1-read-todo": ("read_text_file", lambda r: {"path": f"{r}/notes/todo.txt"}),
        "u1-write-hello": (
            "write_file",
            lambda r: {"path": f"{r}/hello.txt", "content": "hello world"},
        ),
        "u1-create-dir": ("create_directory", lambda r: {"path": f"{r}/new_folder"}),
        "u1-list-notes": ("list_directory", lambda r: {"path": f"{r}/notes"}),
        "u1-list-sizes-data": ("list_directory_with_sizes", lambda r: {"path": f"{r}/data"}),
        "u1-search-md": (
            "search_files",
            lambda r: {"path": str(r), "pattern": "**/*.md"},
        ),
        "u1-directory-tree": ("directory_tree", lambda r: {"path": str(r)}),
        "u1-get-file-info": ("get_file_info", lambda r: {"path": f"{r}/config.json"}),
        "u1-move-file": (
            "move_file",
            lambda r: {
                "source": f"{r}/notes/draft.md",
                "destination": f"{r}/notes/draft_renamed.md",
            },
        ),
        "u1-list-allowed-dirs": ("list_allowed_directories", lambda r: {}),
        "u1-edit-file": (
            "edit_file",
            lambda r: {
                "path": f"{r}/notes/todo.txt",
                "edits": [{"oldText": "buy milk", "newText": "buy oat milk"}],
            },
        ),
        "u1-read-multiple": (
            "read_multiple_files",
            lambda r: {"paths": [f"{r}/notes/todo.txt", f"{r}/notes/draft.md"]},
        ),
        "u1-read-image": ("read_media_file", lambda r: {"path": f"{r}/pixel.png"}),
    }
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    assert set(tasks) == set(tool_by_task)

    for task_id, (tool_name, args_fn) in tool_by_task.items():
        with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id=f"fs-all-{task_id}") as root:
            async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
                call = ParsedCall(name=tool_name, arguments=args_fn(root))
                results = await execute_calls(handle, [call])
                assert results[0].ok, f"{task_id}: call failed: {results[0].error}"
                state = SandboxState(root=root, results=results)
                assert tasks[task_id].checker(state) is True, f"{task_id} checker failed"
