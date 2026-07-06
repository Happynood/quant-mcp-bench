"""Per-instance fixture copy + teardown (spec §5 "Determinism and sandboxing").

Every MCP task instance gets a fresh, ephemeral directory under
SANDBOX_ROOT, seeded from a committed fixture (if any), and destroyed after
the instance completes. This is a correctness requirement (statelessness
across instances) as much as a safety one: no server may ever be launched
with a cwd outside SANDBOX_ROOT. assert_within_sandbox is the single
enforcement point every caller (notably servers/base.py) routes through.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SANDBOX_ROOT = Path(os.environ.get("QUANTMCP_SANDBOX_ROOT", "/tmp/quantmcp-sandbox"))


class SandboxViolation(RuntimeError):
    """Raised when a path would be used outside the sandbox root."""


def assert_within_sandbox(path: Path) -> None:
    root = SANDBOX_ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise SandboxViolation(f"{path} escapes sandbox root {SANDBOX_ROOT}")


@contextmanager
def sandbox_instance(
    fixture_dir: Path | None,
    run_id: str,
    keep_on_failure: bool = False,
) -> Iterator[Path]:
    """Create a fresh sandboxed copy of `fixture_dir` (or an empty dir if
    None) under SANDBOX_ROOT/run_id/<uuid>, yield its path, then destroy it.
    """
    instance_root = SANDBOX_ROOT / run_id / uuid.uuid4().hex
    instance_root.mkdir(parents=True, exist_ok=False)
    assert_within_sandbox(instance_root)

    failed = False
    try:
        if fixture_dir is not None:
            if fixture_dir.is_file():
                # Some fixtures (e.g. the U2 git tier) are shipped as an
                # archive rather than a plain directory: a fixture that is
                # itself a git repository can't be committed as a real .git
                # directory inside this project's own repository (git treats
                # a nested .git as an embedded repository and stores only a
                # gitlink, not its contents), so it's packed as a tarball and
                # extracted fresh per instance instead.
                shutil.unpack_archive(str(fixture_dir), str(instance_root))
            else:
                shutil.copytree(fixture_dir, instance_root, dirs_exist_ok=True)
        yield instance_root
    except Exception:
        failed = True
        raise
    finally:
        if not (failed and keep_on_failure):
            shutil.rmtree(instance_root, ignore_errors=True)
