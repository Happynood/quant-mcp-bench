from __future__ import annotations

import sys
from pathlib import Path

import pytest

from quantmcp.execution.dispatcher import execute_calls
from quantmcp.execution.sandbox import sandbox_instance
from quantmcp.parsing.base import ParsedCall
from quantmcp.servers.base import MCPServerHandle
from quantmcp.tasks.base import SandboxState
from quantmcp.tasks.loader import load_tasks

pytestmark = pytest.mark.integration

_FIXTURES_DIR = Path(__file__).parent.parent / "src" / "quantmcp" / "tasks" / "fixtures"
FIXTURE_DIR = _FIXTURES_DIR / "u3_sqlite"
TASKS_FILE = _FIXTURES_DIR / "u3_sqlite_tasks.yaml"

COMMAND = sys.executable
ARGS = ["-m", "quantmcp.servers.sqlite_server"]


@pytest.mark.asyncio
async def test_sqlite_server_lists_expected_tools():
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="sqlite-list-tools") as root:
        env = {"QUANTMCP_U3_ROOT": str(root)}
        async with MCPServerHandle(COMMAND, ARGS, env=env, cwd=root) as handle:
            tools = await handle.list_tools()
            names = {t.name for t in tools}
            assert {"list_tables", "describe_table", "read_query", "write_query"} <= names


def test_load_u3_sqlite_tasks():
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) == 10


@pytest.mark.asyncio
async def test_all_u3_tasks_pass_with_the_intended_call():
    """Drive every U3 task with the exact call it expects, proving each
    checker actually matches this server's real response/state (not a
    guess)."""
    tool_by_task = {
        "u3-list-tables": ("list_tables", {}),
        "u3-describe-employees": ("describe_table", {"table": "employees"}),
        "u3-describe-inventory": ("describe_table", {"table": "inventory"}),
        "u3-query-engineering": (
            "read_query",
            {"query": "SELECT name FROM employees WHERE department = 'Engineering'"},
        ),
        "u3-query-salary": (
            "read_query",
            {"query": "SELECT salary FROM employees WHERE name = 'Carla Diaz'"},
        ),
        "u3-query-widget-count": (
            "read_query",
            {"query": "SELECT quantity FROM inventory WHERE item = 'widget'"},
        ),
        "u3-update-gadget-quantity": (
            "write_query",
            {"query": "UPDATE inventory SET quantity = 100 WHERE item = 'gadget'"},
        ),
        "u3-give-raise": (
            "write_query",
            {"query": "UPDATE employees SET salary = 90000 WHERE name = 'Bob Martins'"},
        ),
        "u3-delete-widget": (
            "write_query",
            {"query": "DELETE FROM inventory WHERE item = 'widget'"},
        ),
        "u3-insert-employee": (
            "write_query",
            {
                "query": (
                    "INSERT INTO employees (id, name, department, salary) "
                    "VALUES (4, 'Dana Kim', 'Marketing', 65000)"
                )
            },
        ),
    }
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    assert set(tasks) == set(tool_by_task)

    for task_id, (tool_name, args) in tool_by_task.items():
        with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id=f"sqlite-all-{task_id}") as root:
            env = {"QUANTMCP_U3_ROOT": str(root)}
            async with MCPServerHandle(COMMAND, ARGS, env=env, cwd=root) as handle:
                call = ParsedCall(name=tool_name, arguments=args)
                results = await execute_calls(handle, [call])
                assert results[0].ok, f"{task_id}: call failed: {results[0].error}"
                state = SandboxState(root=root, results=results)
                assert tasks[task_id].checker(state) is True, f"{task_id} checker failed"
