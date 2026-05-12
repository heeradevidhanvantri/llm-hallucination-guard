from __future__ import annotations

from hallucination_guard.checkers.base import BaseChecker
from hallucination_guard.models import CheckerResult, Document
from hallucination_guard.utils import (
    best_context_similarity,
    clamp,
    split_sentences,
)


class GroundingChecker(BaseChecker):
    """
    Measures how well each sentence in the response is supported by the
    retrieved context using TF-IDF cosine similarity.

    Optionally upgrades to semantic embeddings when sentence-transformers
    is installed and use_semantic=True.
    """

    name = "grounding"

    def __init__(
        self,
        min_sentence_length: int = 10,
        use_semantic: bool = False,
        semantic_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.min_sentence_length = min_sentence_length
        self.use_semantic = use_semantic
        self._encoder = None

        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np  # noqa: F401
                self._encoder = SentenceTransformer(semantic_model)
            except ImportError:
                pass  # silently fall back to TF-IDF

    def _semantic_similarity(self, sentence: str, chunks: list[str]) -> float:
        import numpy as np

        assert self._encoder is not None
        vecs = self._encoder.encode([sentence] + chunks, convert_to_numpy=True)
        q = vecs[0]
        docs = vecs[1:]
        norms = np.linalg.norm(docs, axis=1) * np.linalg.norm(q)
        scores = np.where(norms > 0, docs @ q / norms, 0.0)
        return float(scores.max()) if len(scores) else 0.0

    def check(
        self,
        query: str,
        response: str,
        context: list[Document],
    ) -> CheckerResult:
        if not context:
            return CheckerResult(score=0.0, passed=False, details={"reason": "no_context"})

        chunks = [doc.content for doc in context]
        sentences = split_sentences(response, self.min_sentence_length)

        if not sentences:
            return CheckerResult(score=0.0, passed=False, details={"reason": "empty_response"})

        sentence_scores: list[float] = []
        for sent in sentences:
            if self._encoder is not None:
                sim = self._semantic_similarity(sent, chunks)
            else:
                sim = best_context_similarity(sent, chunks)
            sentence_scores.append(sim)

        avg_score = clamp(sum(sentence_scores) / len(sentence_scores))
        passed = avg_score >= 0.5

        return CheckerResult(
            score=avg_score,
            passed=passed,
            details={
                "sentence_count": len(sentences),
                "sentence_scores": {s: round(sc, 4) for s, sc in zip(sentences, sentence_scores)},
                "method": "semantic" if self._encoder else "tfidf",
            },
        )
