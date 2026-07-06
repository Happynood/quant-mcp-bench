from __future__ import annotations

from pathlib import Path

import pytest

from quantmcp.execution.sandbox import (
    SANDBOX_ROOT,
    SandboxViolation,
    assert_within_sandbox,
    sandbox_instance,
)


def test_assert_within_sandbox_passes_for_child_path():
    assert_within_sandbox(SANDBOX_ROOT / "abc" / "def")


def test_assert_within_sandbox_rejects_outside_path(tmp_path):
    with pytest.raises(SandboxViolation):
        assert_within_sandbox(tmp_path)


def test_assert_within_sandbox_rejects_home_dir():
    with pytest.raises(SandboxViolation):
        assert_within_sandbox(Path.home())


def test_sandbox_instance_creates_and_destroys_dir():
    captured: Path | None = None
    with sandbox_instance(fixture_dir=None, run_id="test-run") as instance_root:
        captured = instance_root
        assert instance_root.exists()
        assert_within_sandbox(instance_root)
    assert captured is not None
    assert not captured.exists()


def test_sandbox_instance_copies_fixture(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "hello.txt").write_text("hi")

    with sandbox_instance(fixture_dir=fixture, run_id="test-run-2") as instance_root:
        assert (instance_root / "hello.txt").read_text() == "hi"


def test_sandbox_instance_keep_on_failure():
    captured: Path | None = None
    with pytest.raises(RuntimeError):
        with sandbox_instance(
            fixture_dir=None, run_id="test-run-3", keep_on_failure=True
        ) as instance_root:
            captured = instance_root
            raise RuntimeError("boom")
    assert captured is not None
    assert captured.exists()
    import shutil

    shutil.rmtree(captured, ignore_errors=True)


def test_sandbox_instance_cleans_up_on_failure_without_keep():
    captured: Path | None = None
    with pytest.raises(RuntimeError):
        with sandbox_instance(
            fixture_dir=None, run_id="test-run-4", keep_on_failure=False
        ) as instance_root:
            captured = instance_root
            raise RuntimeError("boom")
    assert captured is not None
    assert not captured.exists()
