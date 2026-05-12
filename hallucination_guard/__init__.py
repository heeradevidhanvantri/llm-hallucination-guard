"""
LLM Hallucination Guard
=======================
Open-source RAG reliability layer that detects and prevents LLM hallucinations.

Quick start:
    from hallucination_guard import HallucinationGuard

    guard = HallucinationGuard()
    result = guard.check(
        query="What is the capital of France?",
        context=["Paris is the capital and most populous city of France."],
        response="The capital of France is Paris.",
    )
    print(result.grounding_score)    # 0.91
    print(result.is_hallucination)   # False
    print(result.summary())
"""

from hallucination_guard.config import GuardConfig
from hallucination_guard.guard import HallucinationGuard
from hallucination_guard.models import (
    CitationResult,
    Contradiction,
    Document,
    GuardResult,
    HallucinationType,
)
from hallucination_guard.retrievers import SimpleRetriever

__version__ = "0.1.0"
__all__ = [
    "HallucinationGuard",
    "GuardConfig",
    "GuardResult",
    "Document",
    "Contradiction",
    "CitationResult",
    "HallucinationType",
    "SimpleRetriever",
]
