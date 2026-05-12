import pytest
from hallucination_guard.checkers.grounding import GroundingChecker
from hallucination_guard.checkers.contradiction import ContradictionChecker
from hallucination_guard.checkers.citation import CitationChecker
from hallucination_guard.models import Document


CONTEXT = [
    Document(content="Paris is the capital and most populous city of France."),
    Document(content="The Eiffel Tower, located in Paris, was built in 1889."),
]


# ── Grounding ─────────────────────────────────────────────────────────────────

class TestGroundingChecker:
    def setup_method(self):
        self.checker = GroundingChecker()

    def test_well_grounded_response(self):
        result = self.checker.check(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context=CONTEXT,
        )
        assert result.score > 0.3
        assert "sentence_scores" in result.details

    def test_off_topic_response(self):
        result = self.checker.check(
            query="What is the capital of France?",
            response="Quantum mechanics describes the behavior of subatomic particles.",
            context=CONTEXT,
        )
        assert result.score < 0.5

    def test_no_context(self):
        result = self.checker.check(
            query="q", response="some response", context=[]
        )
        assert result.score == 0.0
        assert not result.passed

    def test_empty_response(self):
        result = self.checker.check(
            query="q", response=".", context=CONTEXT
        )
        assert result.score == 0.0


# ── Contradiction ─────────────────────────────────────────────────────────────

class TestContradictionChecker:
    def setup_method(self):
        self.checker = ContradictionChecker(llm=None)

    def test_no_contradiction(self):
        result = self.checker.check(
            query="When was the Eiffel Tower built?",
            response="The Eiffel Tower was built in 1889.",
            context=CONTEXT,
        )
        assert result.passed

    def test_no_context_passes(self):
        result = self.checker.check(
            query="q", response="some text", context=[]
        )
        assert result.score == 1.0
        assert result.passed

    def test_score_range(self):
        result = self.checker.check(
            query="q", response="The sky is green.", context=CONTEXT
        )
        assert 0.0 <= result.score <= 1.0


# ── Citation ──────────────────────────────────────────────────────────────────

class TestCitationChecker:
    def setup_method(self):
        self.checker = CitationChecker(similarity_threshold=0.3)

    def test_no_citations_passes(self):
        result = self.checker.check(
            query="q",
            response="Paris is the capital of France.",
            context=CONTEXT,
        )
        assert result.score == 1.0
        assert result.details.get("reason") == "no_citations_found"

    def test_valid_numbered_citation(self):
        result = self.checker.check(
            query="q",
            response="Paris is the capital [1].",
            context=CONTEXT,
        )
        assert result.score == 1.0

    def test_invalid_numbered_citation(self):
        result = self.checker.check(
            query="q",
            response="Something happened [99].",
            context=CONTEXT,
        )
        assert result.score < 1.0

    def test_quoted_claim_verified(self):
        result = self.checker.check(
            query="q",
            response='According to the source, "Paris is the capital and most populous city of France."',
            context=CONTEXT,
        )
        citations = result.details.get("citations", [])
        assert len(citations) == 1
        assert citations[0]["verified"]
