"""
Basic usage of LLM Hallucination Guard.

Run:
    python examples/basic_usage.py
"""

from hallucination_guard import HallucinationGuard, GuardConfig

guard = HallucinationGuard()

cases = [
    {
        "label": "Well-grounded response",
        "query": "What is the capital of France?",
        "context": [
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
        "response": "The capital of France is Paris.",
    },
    {
        "label": "Hallucinated response (wrong fact)",
        "query": "When was the Eiffel Tower built?",
        "context": [
            "The Eiffel Tower was constructed between 1887 and 1889 for the 1889 World's Fair.",
        ],
        "response": "The Eiffel Tower was built in 1950 as a post-war monument.",
    },
    {
        "label": "Response with extrinsic hallucination",
        "query": "Tell me about the Amazon river.",
        "context": [
            "The Amazon River is the largest river by discharge volume of water in the world.",
            "It flows through Brazil, Peru, and Colombia.",
        ],
        "response": (
            "The Amazon River is the largest river in the world. "
            "It was first explored by Christopher Columbus in 1492 "
            "and contains over 5,000 species of fish."
        ),
    },
]

for case in cases:
    print(f"\n{'='*60}")
    print(f"  {case['label']}")
    print(f"{'='*60}")
    print(f"  Query:    {case['query']}")
    print(f"  Response: {case['response'][:80]}...")

    result = guard.check(
        query=case["query"],
        context=case["context"],
        response=case["response"],
    )

    print(f"\n  {result.summary()}")
    print(f"  Checker scores: {result.checker_scores}")
    if result.contradictions:
        for c in result.contradictions:
            print(f"  Contradiction: {c.explanation}")
