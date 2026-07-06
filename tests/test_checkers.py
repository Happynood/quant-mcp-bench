from __future__ import annotations

import sqlite3
from pathlib import Path

from quantmcp.tasks.base import SandboxState
from quantmcp.tasks.checkers import sqlite_query_scalar


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (name TEXT, qty INTEGER)")
    conn.execute("INSERT INTO items (name, qty) VALUES ('widget', 42)")
    conn.commit()
    conn.close()


def test_sqlite_query_scalar_matches_expected_value(tmp_path: Path):
    _make_db(tmp_path / "fixture.db")
    state = SandboxState(root=tmp_path)
    assert sqlite_query_scalar(
        state, "fixture.db", "SELECT qty FROM items WHERE name = 'widget'", "42"
    )


def test_sqlite_query_scalar_fails_on_mismatched_value(tmp_path: Path):
    _make_db(tmp_path / "fixture.db")
    state = SandboxState(root=tmp_path)
    assert not sqlite_query_scalar(
        state, "fixture.db", "SELECT qty FROM items WHERE name = 'widget'", "99"
    )


def test_sqlite_query_scalar_fails_when_no_row_matches(tmp_path: Path):
    _make_db(tmp_path / "fixture.db")
    state = SandboxState(root=tmp_path)
    assert not sqlite_query_scalar(
        state, "fixture.db", "SELECT qty FROM items WHERE name = 'nonexistent'", "42"
    )


def test_sqlite_query_scalar_fails_when_db_missing(tmp_path: Path):
    state = SandboxState(root=tmp_path)
    assert not sqlite_query_scalar(state, "missing.db", "SELECT 1", "1")
