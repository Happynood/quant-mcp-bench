from __future__ import annotations

from pathlib import Path

import pytest

from quantmcp.execution.dispatcher import execute_calls
from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.parsing.base import ParsedCall
from quantmcp.servers.base import MCPServerHandle
from quantmcp.servers.git import ARGS_TEMPLATE, COMMAND
from quantmcp.tasks.base import SandboxState
from quantmcp.tasks.loader import load_tasks

pytestmark = pytest.mark.integration

_FIXTURES_DIR = Path(__file__).parent.parent / "src" / "quantmcp" / "tasks" / "fixtures"
FIXTURE = _FIXTURES_DIR / "u2_git" / "repo.tar.gz"
TASKS_FILE = _FIXTURES_DIR / "u2_git_tasks.yaml"


def _resolved_args(root: Path) -> list[str]:
    return [a.replace("{root}", str(root)) for a in ARGS_TEMPLATE]


@pytest.mark.asyncio
async def test_git_server_lists_expected_tools():
    with sandbox_instance(fixture_dir=FIXTURE, run_id="git-list-tools") as root:
        async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
            tools = await handle.list_tools()
            names = {t.name for t in tools}
            assert {"git_status", "git_log", "git_commit", "git_add"} <= names


def test_load_u2_git_tasks():
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) == 12


@pytest.mark.asyncio
async def test_all_u2_tasks_pass_with_the_intended_call():
    """Drive every U2 task with the exact call it expects, proving each
    checker actually matches this server's real response shape (not a
    guess)."""
    tool_by_task = {
        "u2-status": ("git_status", lambda r: {"repo_path": str(r)}),
        "u2-log": ("git_log", lambda r: {"repo_path": str(r), "max_count": 5}),
        "u2-diff-unstaged": ("git_diff_unstaged", lambda r: {"repo_path": str(r)}),
        "u2-diff-staged": ("git_diff_staged", lambda r: {"repo_path": str(r)}),
        "u2-branch-list": ("git_branch", lambda r: {"repo_path": str(r), "branch_type": "all"}),
        "u2-show-head": ("git_show", lambda r: {"repo_path": str(r), "revision": "HEAD"}),
        "u2-add-draft": ("git_add", lambda r: {"repo_path": str(r), "files": ["draft.txt"]}),
        "u2-reset": ("git_reset", lambda r: {"repo_path": str(r)}),
        "u2-create-branch": (
            "git_create_branch",
            lambda r: {"repo_path": str(r), "branch_name": "release-1"},
        ),
        "u2-commit": (
            "git_commit",
            lambda r: {"repo_path": str(r), "message": "Apply pending updates"},
        ),
        "u2-diff-feature-x": (
            "git_diff",
            lambda r: {"repo_path": str(r), "target": "feature-x"},
        ),
        "u2-checkout-main": (
            "git_checkout",
            lambda r: {"repo_path": str(r), "branch_name": "main"},
        ),
    }
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    assert set(tasks) == set(tool_by_task)

    for task_id, (tool_name, args_fn) in tool_by_task.items():
        with sandbox_instance(fixture_dir=FIXTURE, run_id=f"git-all-{task_id}") as root:
            async with MCPServerHandle(COMMAND, _resolved_args(root), cwd=root) as handle:
                call = ParsedCall(name=tool_name, arguments=args_fn(root))
                results = await execute_calls(handle, [call])
                assert results[0].ok, f"{task_id}: call failed: {results[0].error}"
                state = SandboxState(root=root, results=results)
                assert tasks[task_id].checker(state) is True, f"{task_id} checker failed"
