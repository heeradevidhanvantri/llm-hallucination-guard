from __future__ import annotations

import re
import math
from collections import Counter


def split_sentences(text: str, min_length: int = 10) -> list[str]:
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def tfidf_cosine_similarity(query: str, documents: list[str]) -> list[float]:
    corpus = [query] + documents
    tokenized = [tokenize(doc) for doc in corpus]

    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    N = len(corpus)

    def tfidf_vec(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((N + 1) / (df[term] + 1)) + 1.0
            vec[term] = (count / len(tokens)) * idf if tokens else 0.0
        return vec

    vecs = [tfidf_vec(t) for t in tokenized]
    query_vec = vecs[0]

    scores: list[float] = []
    for doc_vec in vecs[1:]:
        dot = sum(query_vec.get(t, 0.0) * doc_vec.get(t, 0.0) for t in query_vec)
        mag_q = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        mag_d = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
        if mag_q == 0 or mag_d == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (mag_q * mag_d))

    return scores


def best_context_similarity(sentence: str, context_chunks: list[str]) -> float:
    if not context_chunks:
        return 0.0
    scores = tfidf_cosine_similarity(sentence, context_chunks)
    return max(scores) if scores else 0.0


def extract_claims(text: str) -> list[str]:
    sentences = split_sentences(text)
    claims: list[str] = []
    for s in sentences:
        lower = s.lower()
        if any(
            lower.startswith(w)
            for w in ("is ", "are ", "was ", "were ", "the ", "a ", "an ",
                      "it ", "this ", "that ", "these ", "those ")
        ):
            claims.append(s)
        elif len(s) > 20:
            claims.append(s)
    return claims


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
