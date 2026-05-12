from __future__ import annotations

from typing import TYPE_CHECKING

from hallucination_guard.checkers.citation import CitationChecker
from hallucination_guard.checkers.contradiction import ContradictionChecker
from hallucination_guard.checkers.grounding import GroundingChecker
from hallucination_guard.config import GuardConfig
from hallucination_guard.models import (
    Contradiction,
    Document,
    GuardResult,
    HallucinationType,
)
from hallucination_guard.utils import clamp

if TYPE_CHECKING:
    from hallucination_guard.llm.base import BaseLLM


class HallucinationGuard:
    """
    LLM Hallucination Guard — RAG reliability layer.

    Wraps any RAG pipeline response and scores it across three dimensions:
      1. Grounding  — is the response supported by the retrieved context?
      2. Contradiction — does the response contradict the retrieved context?
      3. Citation     — are quoted claims traceable to source documents?

    Quick start
    -----------
    >>> guard = HallucinationGuard()
    >>> result = guard.check(
    ...     query="What is the boiling point of water?",
    ...     context=["Water boils at 100°C (212°F) at sea level."],
    ...     response="Water boils at 100 degrees Celsius.",
    ... )
    >>> print(result.grounding_score)   # 0.87
    >>> print(result.is_hallucination)  # False
    """

    def __init__(
        self,
        config: GuardConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.llm = llm

        self._grounding_checker = GroundingChecker(
            min_sentence_length=self.config.min_sentence_length,
            use_semantic=self.config.use_semantic_similarity,
            semantic_model=self.config.semantic_model,
        )
        self._contradiction_checker = ContradictionChecker(
            llm=self.llm if self.config.use_llm_checker else None,
        )
        self._citation_checker = CitationChecker(
            similarity_threshold=self.config.citation_similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        query: str,
        response: str,
        context: list[str] | list[Document],
    ) -> GuardResult:
        """
        Evaluate a RAG response for hallucination.

        Parameters
        ----------
        query:    The original user query.
        response: The LLM-generated response to evaluate.
        context:  Retrieved source documents (strings or Document objects).

        Returns
        -------
        GuardResult with per-checker scores and an overall verdict.
        """
        docs = self._to_documents(context)

        grounding_result = self._grounding_checker.check(query, response, docs)
        contradiction_result = self._contradiction_checker.check(query, response, docs)
        citation_result = self._citation_checker.check(query, response, docs)

        composite = clamp(
            self.config.grounding_weight * grounding_result.score
            + self.config.contradiction_weight * contradiction_result.score
            + self.config.citation_weight * citation_result.score
        )

        is_hallucination = composite < self.config.hallucination_threshold

        hallucination_type = HallucinationType.NONE
        if is_hallucination:
            n_contradictions = contradiction_result.details.get("contradiction_count", 0)
            hallucination_type = (
                HallucinationType.INTRINSIC if n_contradictions > 0
                else HallucinationType.EXTRINSIC
            )

        contradictions = [
            Contradiction(**c)
            for c in contradiction_result.details.get("contradictions", [])
            if isinstance(c, dict) and "claim" in c and "conflicting_source" in c
               and "explanation" in c and "severity" in c
        ]

        return GuardResult(
            query=query,
            response=response,
            grounding_score=grounding_result.score,
            is_hallucination=is_hallucination,
            hallucination_type=hallucination_type,
            confidence=composite,
            contradictions=contradictions,
            checker_scores={
                "grounding": grounding_result.score,
                "contradiction": contradiction_result.score,
                "citation": citation_result.score,
            },
            metadata={
                **self.config.extra_metadata,
                "grounding_details": grounding_result.details,
                "contradiction_details": contradiction_result.details,
                "citation_details": citation_result.details,
            },
        )

    def is_hallucination(
        self,
        query: str,
        response: str,
        context: list[str] | list[Document],
    ) -> bool:
        """Convenience method — returns True if the response is likely hallucinated."""
        return self.check(query, response, context).is_hallucination

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_documents(context: list[str] | list[Document]) -> list[Document]:
        if not context:
            return []
        if isinstance(context[0], str):
            return [Document(content=str(c)) for c in context]
        return [d for d in context if isinstance(d, Document)]
