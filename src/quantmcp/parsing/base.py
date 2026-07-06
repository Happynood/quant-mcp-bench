# Vendored from Happynood/quant-toolcall-bench @6b6e29e5c83a (quantcall->quantmcp).
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]

    def __hash__(self) -> int:
        import json

        return hash((self.name, json.dumps(self.arguments, sort_keys=True)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments


class CallParser(ABC):
    @abstractmethod
    def parse(self, raw_output: str) -> list[ParsedCall]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
