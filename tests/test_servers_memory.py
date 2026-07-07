from __future__ import annotations

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
FIXTURE_DIR = _FIXTURES_DIR / "u4_memory"
TASKS_FILE = _FIXTURES_DIR / "u4_memory_tasks.yaml"

COMMAND = "npx"
ARGS = ["-y", "@modelcontextprotocol/server-memory"]


def _env_for(root: Path) -> dict[str, str]:
    return {"MEMORY_FILE_PATH": str(root / "memory.json")}


@pytest.mark.asyncio
async def test_memory_server_lists_expected_tools():
    with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id="memory-list-tools") as root:
        async with MCPServerHandle(COMMAND, ARGS, env=_env_for(root), cwd=root) as handle:
            tools = await handle.list_tools()
            names = {t.name for t in tools}
            assert {
                "create_entities",
                "create_relations",
                "add_observations",
                "delete_entities",
                "delete_observations",
                "delete_relations",
                "read_graph",
                "search_nodes",
                "open_nodes",
            } <= names


def test_load_u4_memory_tasks():
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) == 10


@pytest.mark.asyncio
async def test_all_u4_tasks_pass_with_the_intended_call():
    """Drive every U4 task with the exact call it expects, proving each
    checker actually matches this server's real response/state (not a
    guess)."""
    tool_by_task = {
        "u4-read-graph": ("read_graph", {}),
        "u4-search-acme": ("search_nodes", {"query": "Acme Corp"}),
        "u4-delete-leads-relation": (
            "delete_relations",
            {"relations": [{"from": "Alice", "to": "Project Falcon", "relationType": "leads"}]},
        ),
        "u4-open-alice": ("open_nodes", {"names": ["Alice"]}),
        "u4-open-alice-and-bob": ("open_nodes", {"names": ["Alice", "Bob"]}),
        "u4-add-bob-hamburg": (
            "add_observations",
            {"observations": [{"entityName": "Bob", "contents": ["Lives in Hamburg"]}]},
        ),
        "u4-create-carol": (
            "create_entities",
            {"entities": [{"name": "Carol", "entityType": "Designer", "observations": []}]},
        ),
        "u4-carol-reports-to-alice": (
            "create_relations",
            {"relations": [{"from": "Carol", "to": "Alice", "relationType": "reports_to"}]},
        ),
        "u4-delete-acme": ("delete_entities", {"entityNames": ["Acme Corp"]}),
        "u4-delete-alice-german": (
            "delete_observations",
            {"deletions": [{"entityName": "Alice", "observations": ["Speaks German and English"]}]},
        ),
    }
    tasks = {t.id: t for t in load_tasks(TASKS_FILE)}
    assert set(tasks) == set(tool_by_task)

    for task_id, (tool_name, args) in tool_by_task.items():
        with sandbox_instance(fixture_dir=FIXTURE_DIR, run_id=f"memory-all-{task_id}") as root:
            async with MCPServerHandle(COMMAND, ARGS, env=_env_for(root), cwd=root) as handle:
                call = ParsedCall(name=tool_name, arguments=args)
                results = await execute_calls(handle, [call])
                assert results[0].ok, f"{task_id}: call failed: {results[0].error}"
                state = SandboxState(root=root, results=results)
                assert tasks[task_id].checker(state) is True, f"{task_id} checker failed"
