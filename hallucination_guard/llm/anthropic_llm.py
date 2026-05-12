from __future__ import annotations

from hallucination_guard.llm.base import BaseLLM

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude integration for LLM-based hallucination checking.

    Requires: pip install llm-hallucination-guard[anthropic]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install with: "
                "pip install llm-hallucination-guard[anthropic]"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        return str(block.text) if hasattr(block, "text") else ""

    def model_name(self) -> str:
        return self._model
