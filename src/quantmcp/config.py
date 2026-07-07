from __future__ import annotations

# Backend sub-configs (MockBackendConfig, LlamaCppBackendConfig, HFBackendConfig,
# OpenAIEndpointConfig, VLLMBackendConfig) are vendored verbatim from
# Happynood/quant-toolcall-bench @ 6b6e29e5c83a91a52bdfc68b5f3063de172c0e55 —
# the backend layer doesn't care whether the schema came from BFCL or MCP.
# QuantMCPConfig itself is new: the top-level shape (server/tasks/sandbox
# instead of tiers/bfcl_data_dir) reflects the MCP-server-driven pipeline
# described in spec §6.2, not the BFCL tier system.
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class MockBackendConfig(BaseModel):
    latency_ms: float = Field(default=5.0, ge=0.0)


class LlamaCppBackendConfig(BaseModel):
    n_ctx: int = Field(default=4096, ge=1)
    n_gpu_layers: int = -1
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    n_threads: int | None = None
    verbose: bool = False
    chat_format: str | None = None


class HFBackendConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=1)
    device: str = "cpu"
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class OpenAIEndpointConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    api_key_env: str | None = None
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_s: float = Field(default=60.0, gt=0.0)


class VLLMBackendConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    guided_decoding_backend: str = "xgrammar"


class ServerConfig(BaseModel):
    """Which MCP server tier to launch and how to find its fixture/tasks."""

    tier: Literal["u0", "filesystem", "git", "sqlite", "memory"] = "u0"
    fixture_dir: str | None = None
    tasks_file: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)


class SandboxConfig(BaseModel):
    root: str = "/tmp/quantmcp-sandbox"
    keep_on_failure: bool = False


class ReferenceConfig(BaseModel):
    quant: str = "fp16"
    backend: str = "transformers"
    result_file: str | None = None


class QuantMCPConfig(BaseModel):
    model: str = "mock"
    backend: Literal["mock", "llama-cpp", "transformers", "vllm", "openai"] = "mock"
    quant: str = "fp16"
    decoding: Literal["free", "constrained"] = "free"
    chat_variant: Literal["default", "qwen3_nothink"] = "default"
    server: ServerConfig = Field(default_factory=ServerConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    sample_size: int = Field(default=50, ge=1)
    seed: int = 42
    temperature: float = Field(default=0.0, ge=0.0)
    repeats: int = Field(default=1, ge=1)
    reference: ReferenceConfig | None = None
    mock: MockBackendConfig = Field(default_factory=MockBackendConfig)
    llama_cpp: LlamaCppBackendConfig = Field(default_factory=LlamaCppBackendConfig)
    hf: HFBackendConfig = Field(default_factory=HFBackendConfig)
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig)
    vllm: VLLMBackendConfig = Field(default_factory=VLLMBackendConfig)

    @field_validator("model")
    @classmethod
    def _expand_home_tilde(cls, v: str) -> str:
        # Committed sweep configs use a portable "~/models/..." path rather
        # than one machine's absolute home directory (see docs/RUN_REAL.md);
        # expanding it here means the same config file actually works on
        # anyone's machine instead of just the one it was authored on.
        return os.path.expanduser(v)


def load_config(path: str | Path) -> QuantMCPConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return QuantMCPConfig.model_validate(data or {})
