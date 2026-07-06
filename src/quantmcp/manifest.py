from __future__ import annotations

# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
# Diff: RunManifest gains `mcp_server_version` and `server_tier` fields (adds
# MCP server package version to the manifest schema); collect_manifest takes
# them as parameters instead of reading BFCL-tier config fields off the config.
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quantmcp.config import QuantMCPConfig
from quantmcp.hardware import GpuInfo, collect_hardware


@dataclass(frozen=True)
class RunManifest:
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    config_sha256: str
    fixture_sha256: str
    model: str
    backend: str
    quant: str
    decoding: str
    server_tier: str
    mcp_server_version: str | None
    seed: int
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    gpu: GpuInfo | None


def collect_manifest(
    config_path: str | Path,
    cfg: QuantMCPConfig,
    fixture_sha256: str = "",
    mcp_server_version: str | None = None,
) -> RunManifest:
    hw = collect_hardware()
    return RunManifest(
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        config_sha256=_file_sha256(config_path),
        fixture_sha256=fixture_sha256,
        model=cfg.model,
        backend=cfg.backend,
        quant=cfg.quant,
        decoding=cfg.decoding,
        server_tier=cfg.server.tier,
        mcp_server_version=mcp_server_version,
        seed=cfg.seed,
        python_version=hw.python_version,
        platform_info=hw.platform_info,
        cpu_model=hw.cpu_model,
        cpu_count=hw.cpu_count,
        gpu=hw.gpu,
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def compute_fixture_sha256(paths: list[Path]) -> str:
    """Hash the sorted contents of every fixture file for reproducibility."""
    hasher = hashlib.sha256()
    for p in sorted(paths):
        hasher.update(str(p).encode())
        try:
            hasher.update(p.read_bytes())
        except OSError:
            pass
    return hasher.hexdigest()


def _file_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None
