from __future__ import annotations

import subprocess
from unittest.mock import patch

from quantmcp.hardware import _collect_gpu_info


def _fake_run(cmd, **kwargs):
    if "--query-gpu=name,driver_version,memory.total" in cmd:
        return subprocess.CompletedProcess(
            cmd, 0, stdout="NVIDIA GeForce RTX 3050 Laptop GPU, 595.71.05, 4096\n", stderr=""
        )
    banner = (
        "+---------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 595.71.05  Driver Version: 595.71.05  CUDA Version: 13.2 |\n"
        "+---------------------------------------------------------------------+\n"
    )
    return subprocess.CompletedProcess(cmd, 0, stdout=banner, stderr="")


def test_collect_gpu_info_survives_query_gpu_not_supporting_cuda_version():
    """Regression test: some nvidia-smi releases reject
    `--query-gpu=...,cuda_version,...` (it's only in the plain-text banner on
    those releases), which must not blank out the other, valid fields."""
    with patch("subprocess.run", side_effect=_fake_run):
        info = _collect_gpu_info()
    assert info is not None
    assert info.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert info.driver_version == "595.71.05"
    assert info.vram_total_mb == 4096
    assert info.cuda_version == "13.2"


def test_collect_gpu_info_returns_none_when_nvidia_smi_absent():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        info = _collect_gpu_info()
    assert info is None
