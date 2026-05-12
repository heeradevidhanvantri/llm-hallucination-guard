from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HallucinationType(str, Enum):
    INTRINSIC = "intrinsic"       # contradicts retrieved context
    EXTRINSIC = "extrinsic"       # adds info not in context
    NONE = "none"


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = str(id(self))


@dataclass
class Contradiction:
    claim: str
    conflicting_source: str
    explanation: str
    severity: float  # 0.0–1.0


@dataclass
class CitationResult:
    citation: str
    verified: bool
    matched_source: str = ""
    similarity: float = 0.0


@dataclass
class CheckerResult:
    score: float               # 0.0 (bad) – 1.0 (good)
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardResult:
    query: str
    response: str
    grounding_score: float          # 0.0–1.0, how well grounded in context
    is_hallucination: bool
    hallucination_type: HallucinationType
    confidence: float               # overall guard confidence 0.0–1.0
    contradictions: list[Contradiction] = field(default_factory=list)
    citation_results: list[CitationResult] = field(default_factory=list)
    checker_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        return self.grounding_score >= 0.5

    def summary(self) -> str:
        status = "HALLUCINATION" if self.is_hallucination else "GROUNDED"
        return (
            f"[{status}] grounding={self.grounding_score:.2f} "
            f"confidence={self.confidence:.2f} "
            f"contradictions={len(self.contradictions)}"
        )
