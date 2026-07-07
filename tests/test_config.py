from __future__ import annotations

import os

from quantmcp.config import QuantMCPConfig


def test_model_path_expands_home_tilde():
    cfg = QuantMCPConfig.model_validate({"model": "~/models/Qwen3-0.6B-Q4_K_M.gguf"})
    assert cfg.model == os.path.expanduser("~/models/Qwen3-0.6B-Q4_K_M.gguf")
    assert "~" not in cfg.model


def test_model_non_path_value_passes_through_unchanged():
    cfg = QuantMCPConfig.model_validate({"model": "mock"})
    assert cfg.model == "mock"


def test_model_hf_repo_id_passes_through_unchanged():
    cfg = QuantMCPConfig.model_validate({"model": "Qwen/Qwen3-0.6B"})
    assert cfg.model == "Qwen/Qwen3-0.6B"
