from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from hallucination_guard.checkers.base import BaseChecker
from hallucination_guard.models import CheckerResult, Contradiction, Document
from hallucination_guard.utils import clamp, extract_claims, jaccard_similarity

if TYPE_CHECKING:
    from hallucination_guard.llm.base import BaseLLM


_CONTRADICTION_PROMPT = """\
You are a fact-checking assistant. Given retrieved context and an AI response,
identify any claims in the response that directly contradict the context.

Retrieved Context:
{context}

AI Response:
{response}

Return a JSON array of contradiction objects. Each object must have:
  - "claim": the exact sentence from the response that is contradictory
  - "conflicting_source": the relevant snippet from the context it contradicts
  - "explanation": brief explanation of the contradiction
  - "severity": float 0.0–1.0 (1.0 = definite contradiction)

If there are no contradictions, return an empty array [].
Return ONLY valid JSON, no markdown fences."""


def _parse_contradictions(raw: str) -> list[Contradiction]:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```$", "", raw)
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        result: list[Contradiction] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append(
                Contradiction(
                    claim=item.get("claim", ""),
                    conflicting_source=item.get("conflicting_source", ""),
                    explanation=item.get("explanation", ""),
                    severity=float(item.get("severity", 0.5)),
                )
            )
        return result
    except (json.JSONDecodeError, ValueError):
        return []


class ContradictionChecker(BaseChecker):
    """
    Detects claims in the response that contradict the retrieved context.

    Without an LLM provider, uses token-overlap heuristics.
    With an LLM provider, uses a structured LLM prompt for higher accuracy.
    """

    name = "contradiction"

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    def _heuristic_check(
        self, response: str, context: list[Document]
    ) -> list[Contradiction]:
        claims = extract_claims(response)
        chunks = [doc.content for doc in context]
        contradictions: list[Contradiction] = []

        for claim in claims:
            sims = [jaccard_similarity(claim, chunk) for chunk in chunks]
            if not sims:
                continue
            best_sim = max(sims)
            best_chunk = chunks[sims.index(best_sim)]

            # If the claim has very low overlap with all context, it may be extrinsic
            if best_sim < 0.05 and len(claim.split()) > 6:
                contradictions.append(
                    Contradiction(
                        claim=claim,
                        conflicting_source=best_chunk[:200],
                        explanation="Claim has no lexical support in retrieved context.",
                        severity=0.4,
                    )
                )
        return contradictions

    def _llm_check(self, response: str, context: list[Document]) -> list[Contradiction]:
        assert self.llm is not None
        context_text = "\n\n".join(
            f"[Doc {i+1}]: {doc.content}" for i, doc in enumerate(context)
        )
        prompt = _CONTRADICTION_PROMPT.format(
            context=context_text,
            response=response,
        )
        raw = self.llm.complete(prompt)
        return _parse_contradictions(raw)

    def check(
        self,
        query: str,
        response: str,
        context: list[Document],
    ) -> CheckerResult:
        if not context:
            return CheckerResult(score=1.0, passed=True, details={"reason": "no_context"})

        if self.llm is not None:
            contradictions = self._llm_check(response, context)
        else:
            contradictions = self._heuristic_check(response, context)

        if not contradictions:
            score = 1.0
        else:
            avg_severity = sum(c.severity for c in contradictions) / len(contradictions)
            score = clamp(1.0 - avg_severity)

        return CheckerResult(
            score=score,
            passed=score >= 0.6,
            details={
                "contradiction_count": len(contradictions),
                "contradictions": [
                    {
                        "claim": c.claim,
                        "explanation": c.explanation,
                        "severity": c.severity,
                    }
                    for c in contradictions
                ],
                "method": "llm" if self.llm else "heuristic",
            },
        )
