"""
End-to-end RAG pipeline with Hallucination Guard.

This example shows how to integrate HallucinationGuard into a RAG pipeline
that uses SimpleRetriever for retrieval and any LLM for generation.

To use with a real LLM, set ANTHROPIC_API_KEY or OPENAI_API_KEY env var
and uncomment the relevant section below.

Run:
    python examples/rag_pipeline.py
"""

from hallucination_guard import HallucinationGuard, Document, GuardConfig
from hallucination_guard.retrievers import SimpleRetriever

# ── 1. Build a knowledge base ──────────────────────────────────────────────────
knowledge_base = [
    Document(
        content="The speed of light in a vacuum is approximately 299,792,458 metres per second.",
        metadata={"source": "physics_textbook", "page": 12},
    ),
    Document(
        content="Albert Einstein published his special theory of relativity in 1905.",
        metadata={"source": "history_of_science", "page": 44},
    ),
    Document(
        content="E=mc² expresses the equivalence of energy (E) and mass (m) multiplied by c squared.",
        metadata={"source": "physics_textbook", "page": 18},
    ),
    Document(
        content="The Large Hadron Collider is located at CERN near Geneva, Switzerland.",
        metadata={"source": "particle_physics", "page": 3},
    ),
]

retriever = SimpleRetriever(documents=knowledge_base)

# ── 2. Configure the guard ─────────────────────────────────────────────────────
config = GuardConfig(
    grounding_threshold=0.5,
    hallucination_threshold=0.4,
    grounding_weight=0.5,
    contradiction_weight=0.35,
    citation_weight=0.15,
)
guard = HallucinationGuard(config=config)

# ── 3. Simulate LLM responses (replace with actual LLM calls in production) ───
queries_and_responses = [
    (
        "What is the speed of light?",
        "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
    ),
    (
        "When did Einstein publish his relativity theory?",
        "Einstein published his special theory of relativity in 1915.",  # wrong year
    ),
    (
        "What does E=mc² mean?",
        "E=mc² is a famous equation by Einstein showing mass-energy equivalence.",
    ),
]

print("RAG Pipeline with Hallucination Guard\n" + "=" * 50)

for query, llm_response in queries_and_responses:
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=3)

    # Guard check
    result = guard.check(
        query=query,
        response=llm_response,
        context=retrieved_docs,
    )

    print(f"\nQuery: {query}")
    print(f"Response: {llm_response}")
    print(f"Retrieved {len(retrieved_docs)} doc(s)")
    print(f"Result: {result.summary()}")
    if result.is_hallucination:
        print("  ⚠  Hallucination detected — consider regenerating or flagging this response.")
    else:
        print("  ✓  Response appears grounded in the retrieved context.")
