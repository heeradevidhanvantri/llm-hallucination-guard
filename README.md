# LLM Hallucination Guard

**Open-source RAG reliability layer that detects and prevents LLM hallucinations.**

[![CI](https://github.com/heeradhanvantri99/llm-hallucination-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/heeradhanvantri99/llm-hallucination-guard/actions)
[![PyPI](https://img.shields.io/pypi/v/llm-hallucination-guard)](https://pypi.org/project/llm-hallucination-guard/)
[![Python](https://img.shields.io/pypi/pyversions/llm-hallucination-guard)](https://pypi.org/project/llm-hallucination-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

RAG (Retrieval-Augmented Generation) pipelines retrieve relevant documents and feed them to an LLM — but the model can still:

- **Ignore** the retrieved context and generate plausible-sounding but wrong facts
- **Contradict** the sources it was given
- **Exaggerate** or add information that was never in the retrieved documents

`llm-hallucination-guard` sits between your retriever and your application to catch these failures before they reach your users.

---

## How It Works

The guard evaluates every LLM response across three independent dimensions:

| Checker | What it measures | Method |
|---|---|---|
| **Grounding** | Is each sentence in the response supported by the retrieved context? | TF-IDF cosine similarity (or semantic embeddings) |
| **Contradiction** | Does the response directly contradict the source documents? | Heuristic overlap + optional LLM judge |
| **Citation** | Are quoted claims and `[n]` references traceable to real sources? | Similarity matching |

A weighted composite score produces a final **grounding_score** (0–1) and an `is_hallucination` verdict.

---

## Install

```bash
# Core (zero ML dependencies)
pip install llm-hallucination-guard

# With Anthropic Claude as the LLM judge
pip install "llm-hallucination-guard[anthropic]"

# With OpenAI as the LLM judge
pip install "llm-hallucination-guard[openai]"

# With semantic embeddings (sentence-transformers)
pip install "llm-hallucination-guard[semantic]"

# Everything
pip install "llm-hallucination-guard[all]"
```

---

## Quick Start

```python
from hallucination_guard import HallucinationGuard

guard = HallucinationGuard()

result = guard.check(
    query="What is the capital of France?",
    context=[
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.",
    ],
    response="The capital of France is Paris.",
)

print(result.grounding_score)    # 0.91
print(result.is_hallucination)   # False
print(result.summary())          # [GROUNDED] grounding=0.91 confidence=0.94 contradictions=0
```

---

## Drop-in RAG Integration

```python
from hallucination_guard import HallucinationGuard, Document
from hallucination_guard.retrievers import SimpleRetriever

# Build your knowledge base
docs = [
    Document(content="Water boils at 100°C at sea level.", metadata={"source": "chemistry.pdf"}),
    Document(content="Einstein published special relativity in 1905.", metadata={"source": "history.pdf"}),
]
retriever = SimpleRetriever(documents=docs)
guard = HallucinationGuard()

def rag_query(user_query: str, llm_response: str) -> str:
    # Retrieve context
    context = retriever.retrieve(user_query, top_k=3)

    # Guard check
    result = guard.check(query=user_query, response=llm_response, context=context)

    if result.is_hallucination:
        return f"[FLAGGED] {result.summary()}"
    return llm_response
```

---

## LLM-Powered Contradiction Detection

For higher accuracy, plug in an LLM as a judge:

```python
from hallucination_guard import HallucinationGuard, GuardConfig
from hallucination_guard.llm import AnthropicLLM  # or OpenAILLM

llm = AnthropicLLM(api_key="sk-ant-...")  # uses claude-sonnet-4-6 by default

guard = HallucinationGuard(
    config=GuardConfig(use_llm_checker=True),
    llm=llm,
)

result = guard.check(
    query="When was the Eiffel Tower built?",
    context=["The Eiffel Tower was constructed between 1887 and 1889."],
    response="The Eiffel Tower was built in 1950.",
)

print(result.is_hallucination)   # True
for c in result.contradictions:
    print(c.explanation)
```

---

## Semantic Similarity Mode

Upgrade from TF-IDF to dense embeddings for better grounding scores:

```python
from hallucination_guard import HallucinationGuard, GuardConfig

guard = HallucinationGuard(
    config=GuardConfig(
        use_semantic_similarity=True,
        semantic_model="all-MiniLM-L6-v2",  # any sentence-transformers model
    )
)
```

Requires: `pip install "llm-hallucination-guard[semantic]"`

---

## GuardResult Fields

```python
@dataclass
class GuardResult:
    query: str
    response: str
    grounding_score: float          # 0.0 (not grounded) – 1.0 (fully grounded)
    is_hallucination: bool          # True if composite score < threshold
    hallucination_type: HallucinationType   # NONE | INTRINSIC | EXTRINSIC
    confidence: float               # overall composite score
    contradictions: list[Contradiction]
    citation_results: list[CitationResult]
    checker_scores: dict[str, float]    # per-checker breakdown
    metadata: dict[str, Any]
```

`HallucinationType.INTRINSIC` — response directly contradicts the context  
`HallucinationType.EXTRINSIC` — response adds facts not present in the context

---

## Configuration

```python
from hallucination_guard import GuardConfig

config = GuardConfig(
    # Score thresholds
    grounding_threshold=0.5,       # min score to be considered "grounded"
    hallucination_threshold=0.4,   # composite score below this → hallucination

    # Checker weights (must sum to 1.0)
    grounding_weight=0.50,
    contradiction_weight=0.35,
    citation_weight=0.15,

    # Features
    use_llm_checker=False,         # enable LLM-based contradiction detection
    use_semantic_similarity=False, # enable embedding-based grounding

    # Citation verification
    citation_similarity_threshold=0.6,
)
```

---

## Hallucination Types

| Type | Description | Example |
|---|---|---|
| `NONE` | Response is grounded and consistent | Correct answer supported by context |
| `INTRINSIC` | Response contradicts the retrieved documents | Says 1950 when context says 1889 |
| `EXTRINSIC` | Response adds facts absent from context | Invents statistics not in any source |

---

## Architecture

```
hallucination_guard/
├── guard.py              # HallucinationGuard — main public API
├── config.py             # GuardConfig
├── models.py             # GuardResult, Document, Contradiction, CitationResult
├── utils.py              # TF-IDF, Jaccard, sentence splitter
├── checkers/
│   ├── grounding.py      # Sentence-level grounding score
│   ├── contradiction.py  # Contradiction detection (heuristic + LLM)
│   └── citation.py       # Citation and quoted-claim verification
├── llm/
│   ├── anthropic_llm.py  # Anthropic Claude integration
│   └── openai_llm.py     # OpenAI GPT integration
└── retrievers/
    └── base.py           # SimpleRetriever (TF-IDF, zero dependencies)
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss major changes.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest`)
4. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE).
