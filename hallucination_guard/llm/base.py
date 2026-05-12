from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Thin synchronous interface for LLM completion used by the LLM-based checkers."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        ...

    @abstractmethod
    def model_name(self) -> str:
        ...
