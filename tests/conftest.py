from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def smoke_config_path() -> Path:
    return REPO_ROOT / "configs" / "smoke.yaml"
