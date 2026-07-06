# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
"""vLLM offline inference backend (vllm.LLM.chat()).

Optional dependency — install with: uv sync --extra vllm
Requires a CUDA-capable GPU with enough VRAM for vLLM's paged-attention KV
cache allocator; vLLM has no CPU inference path. Not verified on hardware
with less than ~8GB VRAM — see docs/hf/VLLM_SCOPE_NOTE.md.
"""

from __future__ import annotations

import time
from typing import Any

from quantmcp.backends.base import Backend, ToolCallResult


class VLLMBackend(Backend):
    """Inference backend using vLLM's offline `LLM.chat()` API."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
    ) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )

    @property
    def name(self) -> str:
        return "vllm"

    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        sampling_params = self._sampling_params_cls(
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

        t_start = time.perf_counter()
        outputs = self._llm.chat(
            messages,
            sampling_params=sampling_params,
            tools=tools if tools else None,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        output = outputs[0]
        completion = output.outputs[0]
        raw_output: str = completion.text
        input_tokens: int = len(output.prompt_token_ids)
        output_tokens: int = len(completion.token_ids)

        return ToolCallResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tokens_per_second=(
                output_tokens / (latency_ms / 1000.0)
                if latency_ms > 0 and output_tokens > 0
                else None
            ),
        )
