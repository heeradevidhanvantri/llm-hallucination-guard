from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardConfig:
    # Grounding thresholds
    grounding_threshold: float = 0.5
    hallucination_threshold: float = 0.4

    # Checker weights (must sum to 1.0)
    grounding_weight: float = 0.5
    contradiction_weight: float = 0.35
    citation_weight: float = 0.15

    # Sentence splitting
    min_sentence_length: int = 10

    # LLM checker settings
    use_llm_checker: bool = False       # requires an LLM provider
    llm_checker_model: str = ""         # defaults to provider default

    # Semantic similarity (requires sentence-transformers)
    use_semantic_similarity: bool = False
    semantic_model: str = "all-MiniLM-L6-v2"

    # Citation settings
    citation_similarity_threshold: float = 0.6

    # Extra metadata to attach to every GuardResult
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.grounding_weight + self.contradiction_weight + self.citation_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Checker weights must sum to 1.0, got {total:.4f}. "
                "Adjust grounding_weight, contradiction_weight, or citation_weight."
            )
