from hallucination_guard.utils import (
    best_context_similarity,
    clamp,
    jaccard_similarity,
    split_sentences,
    tfidf_cosine_similarity,
    tokenize,
)


def test_split_sentences_basic():
    text = "The sky is blue. Water is wet. Fire is hot."
    sentences = split_sentences(text, min_length=5)
    assert len(sentences) == 3


def test_split_sentences_min_length():
    text = "Hi. The quick brown fox jumps over the lazy dog."
    sentences = split_sentences(text, min_length=10)
    assert "Hi." not in sentences
    assert any("fox" in s for s in sentences)


def test_tokenize():
    tokens = tokenize("Hello, World! 123")
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens


def test_jaccard_identical():
    assert jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard_similarity("hello", "world") == 0.0


def test_jaccard_partial():
    score = jaccard_similarity("hello world", "hello there")
    assert 0.0 < score < 1.0


def test_jaccard_empty():
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("hello", "") == 0.0


def test_tfidf_cosine_similarity():
    query = "Paris is the capital of France"
    docs = [
        "Paris is the capital city of France.",
        "The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany.",
    ]
    scores = tfidf_cosine_similarity(query, docs)
    assert len(scores) == 3
    assert scores[0] > scores[2], "France doc should score higher than Germany doc"
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_best_context_similarity():
    sentence = "Water boils at 100 degrees Celsius."
    context = [
        "Water boils at 100°C at sea level.",
        "The moon orbits the Earth.",
    ]
    score = best_context_similarity(sentence, context)
    assert score > 0.0


def test_clamp():
    assert clamp(1.5) == 1.0
    assert clamp(-0.5) == 0.0
    assert clamp(0.7) == 0.7
