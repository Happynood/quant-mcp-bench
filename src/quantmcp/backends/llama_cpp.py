# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import ctypes
import json
import site
import subprocess
import time
from pathlib import Path
from typing import Any

from quantmcp.backends.base import Backend, ToolCallResult, tools_to_openai_spec
from quantmcp.decoding.gbnf import build_tool_call_grammar


def _preload_cuda_libs() -> None:
    """Load CUDA runtime via ctypes (RTLD_GLOBAL) before importing llama_cpp.

    When llama-cpp-python is installed from a pre-built CUDA wheel the bundled
    libllama.so links against libcudart.so.12.  If the CUDA toolkit is not in
    the system library path (common on laptops with only the driver installed),
    the import fails with "libcudart.so.12: cannot open shared object file".

    Workaround: pre-load the shared object from the nvidia-cuda-runtime-cu12
    pip package with RTLD_GLOBAL so it is visible to every subsequent dlopen.
    """
    for site_dir in site.getsitepackages():
        for subpath in (
            "nvidia/cuda_runtime/lib",
            "nvidia/cublas/lib",
            "nvidia/cuda_nvrtc/lib",
        ):
            lib_dir = Path(site_dir) / subpath
            if not lib_dir.exists():
                continue
            for lib_file in sorted(lib_dir.glob("*.so*")):
                if lib_file.is_symlink():
                    continue
                try:
                    ctypes.CDLL(str(lib_file), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _get_vram_mb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _tool_call_to_json(tc: dict[str, Any]) -> str:
    fn = tc.get("function", {})
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return json.dumps({"name": fn.get("name", ""), "arguments": args}, ensure_ascii=False)


class LlamaCppBackend(Backend):
    """llama.cpp inference backend using llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        chat_format: str | None = None,
        verbose: bool = False,
        n_threads: int | None = None,
        decoding: str = "free",
    ) -> None:
        _preload_cuda_libs()
        from llama_cpp import Llama

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model_path = model_path
        self._decoding = decoding

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            chat_format=chat_format,
            n_threads=n_threads,
        )

    @property
    def name(self) -> str:
        return "llama-cpp"

    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        vram_before = _get_vram_mb()
        t_start = time.perf_counter()

        openai_tools = tools_to_openai_spec(tools) if tools else []

        grammar = None
        if self._decoding == "constrained" and openai_tools:
            from llama_cpp import LlamaGrammar

            gbnf_text = build_tool_call_grammar(openai_tools)
            grammar = LlamaGrammar.from_string(gbnf_text, verbose=False)

        # Try native tool-call completion first; fall back to plain completion.
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                grammar=grammar,
            )
        except Exception:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                grammar=grammar,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        vram_after = _get_vram_mb()

        choice = response["choices"][0]
        msg = choice.get("message", {})

        # Build raw_output that the parser can consume.
        native_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
        if native_calls:
            raw_output = "\n".join(_tool_call_to_json(tc) for tc in native_calls)
        else:
            raw_output = msg.get("content") or ""

        usage = response.get("usage", {})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)
        peak_vram = (
            max(v for v in (vram_before, vram_after) if v is not None)
            if (vram_before is not None or vram_after is not None)
            else None
        )

        return ToolCallResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            peak_vram_mb=float(peak_vram) if peak_vram is not None else None,
            tokens_per_second=(
                output_tokens / (latency_ms / 1000.0)
                if latency_ms > 0 and output_tokens > 0
                else None
            ),
        )
