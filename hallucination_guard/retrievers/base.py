from __future__ import annotations

from abc import ABC, abstractmethod

from hallucination_guard.models import Document


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        ...


class SimpleRetriever(BaseRetriever):
    """
    In-memory retriever that ranks documents by TF-IDF similarity to the query.
    Zero external dependencies — useful for testing and small document sets.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def add(self, document: Document) -> None:
        self.documents.append(document)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        from hallucination_guard.utils import tfidf_cosine_similarity

        if not self.documents:
            return []

        chunks = [doc.content for doc in self.documents]
        scores = tfidf_cosine_similarity(query, chunks)
        ranked = sorted(zip(scores, self.documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
