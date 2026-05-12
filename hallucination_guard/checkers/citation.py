from __future__ import annotations

import re

from hallucination_guard.checkers.base import BaseChecker
from hallucination_guard.models import CheckerResult, CitationResult, Document
from hallucination_guard.utils import clamp, tfidf_cosine_similarity

# Patterns for common citation styles: [1], (Smith 2020), [Smith et al.]
_CITATION_PATTERNS = [
    r'\[(\d+)\]',
    r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4})\)',
    r'\[([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\d{4})\]',
]

# Quoted claims: "some fact"
_QUOTE_PATTERN = re.compile(r'"([^"]{10,200})"')


def _extract_citations(text: str) -> list[str]:
    found: list[str] = []
    for pat in _CITATION_PATTERNS:
        found.extend(re.findall(pat, text))
    return found


def _extract_quoted_claims(text: str) -> list[str]:
    return _QUOTE_PATTERN.findall(text)


class CitationChecker(BaseChecker):
    """
    Verifies that quoted claims and explicit citations in the response
    can be traced back to the retrieved context documents.
    """

    name = "citation"

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self.similarity_threshold = similarity_threshold

    def check(
        self,
        query: str,
        response: str,
        context: list[Document],
    ) -> CheckerResult:
        if not context:
            return CheckerResult(score=1.0, passed=True, details={"reason": "no_context"})

        chunks = [doc.content for doc in context]
        quoted_claims = _extract_quoted_claims(response)
        citations = _extract_citations(response)

        results: list[CitationResult] = []

        # Verify quoted claims against context
        for claim in quoted_claims:
            sims = tfidf_cosine_similarity(claim, chunks)
            best_sim = max(sims) if sims else 0.0
            best_idx = sims.index(best_sim) if sims else 0
            verified = best_sim >= self.similarity_threshold
            results.append(
                CitationResult(
                    citation=f'"{claim}"',
                    verified=verified,
                    matched_source=chunks[best_idx][:150] if verified else "",
                    similarity=round(best_sim, 4),
                )
            )

        # For numbered citations like [1], check if doc index exists
        for ref in citations:
            if ref.isdigit():
                idx = int(ref) - 1
                verified = 0 <= idx < len(context)
                results.append(
                    CitationResult(
                        citation=f"[{ref}]",
                        verified=verified,
                        matched_source=chunks[idx][:150] if verified else "",
                        similarity=1.0 if verified else 0.0,
                    )
                )

        if not results:
            return CheckerResult(
                score=1.0,
                passed=True,
                details={"citation_count": 0, "reason": "no_citations_found"},
            )

        verified_count = sum(1 for r in results if r.verified)
        score = clamp(verified_count / len(results))

        return CheckerResult(
            score=score,
            passed=score >= 0.5,
            details={
                "citation_count": len(results),
                "verified_count": verified_count,
                "citations": [
                    {
                        "citation": r.citation,
                        "verified": r.verified,
                        "similarity": r.similarity,
                    }
                    for r in results
                ],
            },
        )
