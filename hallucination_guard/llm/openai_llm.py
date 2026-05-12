from __future__ import annotations

from hallucination_guard.llm.base import BaseLLM

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT integration for LLM-based hallucination checking.

    Requires: pip install llm-hallucination-guard[openai]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: "
                "pip install llm-hallucination-guard[openai]"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def model_name(self) -> str:
        return self._model
