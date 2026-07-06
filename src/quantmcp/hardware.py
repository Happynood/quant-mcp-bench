# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import importlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    name: str | None
    driver_version: str | None
    cuda_version: str | None
    vram_total_mb: int | None
    torch_cuda_available: bool | None
    torch_cuda_device_name: str | None


@dataclass(frozen=True)
class HardwareInfo:
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    gpu: GpuInfo | None


def collect_hardware() -> HardwareInfo:
    import sys

    return HardwareInfo(
        python_version=sys.version,
        platform_info=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count(),
        gpu=_collect_gpu_info(),
    )


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _collect_gpu_info() -> GpuInfo | None:
    name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    vram_total_mb: int | None = None
    torch_cuda_available: bool | None = None
    torch_cuda_device_name: str | None = None

    try:
        # "cuda_version" is not a valid --query-gpu field on every driver
        # release (it's populated in nvidia-smi's plain-text banner, not the
        # CSV query fields, on some versions) — query the fields that are
        # universally supported here and fetch cuda_version separately below
        # so one unsupported field doesn't blank out the whole fingerprint.
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                name, driver_version = parts[0], parts[1]
                try:
                    vram_total_mb = int(parts[2])
                except ValueError:
                    pass
    except Exception:
        pass

    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            match = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
            if match:
                cuda_version = match.group(1)
    except Exception:
        pass

    try:
        torch_module = importlib.import_module("torch")
        cuda = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        get_device_name = getattr(cuda, "get_device_name", None)
        if callable(is_available):
            torch_cuda_available = bool(is_available())
            if torch_cuda_available and callable(get_device_name):
                torch_cuda_device_name = str(get_device_name(0))
    except Exception:
        pass

    if all(
        v is None
        for v in (
            name,
            driver_version,
            cuda_version,
            vram_total_mb,
            torch_cuda_available,
            torch_cuda_device_name,
        )
    ):
        return None

    return GpuInfo(
        name=name,
        driver_version=driver_version,
        cuda_version=cuda_version,
        vram_total_mb=vram_total_mb,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_device_name=torch_cuda_device_name,
    )
