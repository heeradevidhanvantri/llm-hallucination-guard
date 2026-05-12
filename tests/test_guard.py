import pytest

from hallucination_guard import Document, GuardConfig, HallucinationGuard, HallucinationType

CONTEXT_STRINGS = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower was constructed between 1887 and 1889.",
]

CONTEXT_DOCS = [Document(content=c) for c in CONTEXT_STRINGS]


class TestHallucinationGuard:
    def setup_method(self):
        self.guard = HallucinationGuard()

    # ── Basic checks ─────────────────────────────────────────────────────────

    def test_grounded_response_string_context(self):
        result = self.guard.check(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context=CONTEXT_STRINGS,
        )
        assert 0.0 <= result.grounding_score <= 1.0
        assert result.hallucination_type in HallucinationType.__members__.values()

    def test_grounded_response_document_context(self):
        result = self.guard.check(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context=CONTEXT_DOCS,
        )
        assert result.grounding_score > 0.0

    def test_checker_scores_present(self):
        result = self.guard.check(
            query="q", response="Paris.", context=CONTEXT_STRINGS
        )
        assert "grounding" in result.checker_scores
        assert "contradiction" in result.checker_scores
        assert "citation" in result.checker_scores

    def test_empty_context(self):
        result = self.guard.check(
            query="q", response="The sky is blue.", context=[]
        )
        assert result.grounding_score == 0.0  # no context → zero grounding score
        assert result.checker_scores["grounding"] == 0.0

    def test_convenience_method(self):
        flag = self.guard.is_hallucination(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context=CONTEXT_STRINGS,
        )
        assert isinstance(flag, bool)

    def test_summary_string(self):
        result = self.guard.check(
            query="q", response="Paris is the capital of France.", context=CONTEXT_STRINGS
        )
        summary = result.summary()
        assert "grounding=" in summary
        assert "confidence=" in summary

    # ── Config ───────────────────────────────────────────────────────────────

    def test_custom_config_accepted(self):
        config = GuardConfig(grounding_threshold=0.7, hallucination_threshold=0.5)
        guard = HallucinationGuard(config=config)
        result = guard.check(
            query="q", response="Paris.", context=CONTEXT_STRINGS
        )
        assert result is not None

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            GuardConfig(grounding_weight=0.8, contradiction_weight=0.8, citation_weight=0.1)

    # ── Result fields ─────────────────────────────────────────────────────────

    def test_result_query_and_response_preserved(self):
        q, r = "What is Paris?", "Paris is a city."
        result = self.guard.check(query=q, response=r, context=CONTEXT_STRINGS)
        assert result.query == q
        assert result.response == r

    def test_is_grounded_property(self):
        result = self.guard.check(
            query="q", response="Paris is the capital of France.", context=CONTEXT_STRINGS
        )
        assert result.is_grounded == (result.grounding_score >= 0.5)
