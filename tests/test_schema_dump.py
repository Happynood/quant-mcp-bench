from __future__ import annotations

import sys
from pathlib import Path

import pytest

from quantmcp.schema.dump import dump_tier_schemas

TOY_COMMAND = sys.executable
TOY_ARGS = ["-m", "quantmcp.servers.toy"]


@pytest.mark.asyncio
async def test_dump_tier_schemas_returns_expected_shape_for_u0():
    schemas = await dump_tier_schemas("u0", TOY_COMMAND, TOY_ARGS, fixture_dir=None)

    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"add", "write_note"}
    for entry in schemas:
        assert entry["tier"] == "u0"
        assert isinstance(entry["input_schema"], dict)
        assert "description" in entry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dump_tier_schemas_against_real_sqlite_server():
    import quantmcp

    fixture_dir = Path(quantmcp.__file__).parent / "tasks" / "fixtures" / "u3_sqlite"
    schemas = await dump_tier_schemas(
        "sqlite", sys.executable, ["-m", "quantmcp.servers.sqlite_server"], fixture_dir
    )

    names = {s["name"] for s in schemas}
    assert {"list_tables", "describe_table", "read_query", "write_query"} <= names
