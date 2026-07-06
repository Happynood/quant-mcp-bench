# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from quantmcp.backends.base import Backend, ToolCallResult

_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "auto": "auto",
}


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


class HFBackend(Backend):
    """HuggingFace Transformers inference backend (AutoModelForCausalLM).

    Optional dependency — install with: uv sync --extra transformers
    Templating uses tokenizer.apply_chat_template(..., tools=...), which is
    the standard HF mechanism for tool-calling models (Qwen, Llama, etc.).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        # httpx (used by huggingface_hub) rejects the bare socks:// proxy scheme
        # some shells export; drop it so HF Hub falls through to http(s)_proxy.
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

        dtype_name = _DTYPE_MAP.get(torch_dtype, "auto")
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"

        if quantization_config is not None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                device_map=device,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(
                device
            )
        self._model.eval()

    @property
    def name(self) -> str:
        return "transformers"

    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult:
        torch = self._torch
        vram_before = _get_vram_mb()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len: int = inputs["input_ids"].shape[1]

        t_start = time.perf_counter()
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self._max_new_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "do_sample": self._temperature > 0,
            }
            if self._temperature > 0:
                gen_kwargs["temperature"] = self._temperature
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        vram_after = _get_vram_mb()

        output_len: int = output_ids.shape[1] - input_len
        raw_output: str = self._tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )

        peak_vram = (
            max(v for v in (vram_before, vram_after) if v is not None)
            if (vram_before is not None or vram_after is not None)
            else None
        )

        return ToolCallResult(
            raw_output=raw_output,
            input_tokens=input_len,
            output_tokens=output_len,
            latency_ms=latency_ms,
            peak_vram_mb=float(peak_vram) if peak_vram is not None else None,
            tokens_per_second=(
                output_len / (latency_ms / 1000.0) if latency_ms > 0 and output_len > 0 else None
            ),
        )
