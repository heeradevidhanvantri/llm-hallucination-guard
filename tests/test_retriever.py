from hallucination_guard.models import Document
from hallucination_guard.retrievers import SimpleRetriever

DOCS = [
    Document(content="Paris is the capital of France.", doc_id="d1"),
    Document(content="Berlin is the capital of Germany.", doc_id="d2"),
    Document(content="Tokyo is the capital of Japan.", doc_id="d3"),
    Document(content="The Eiffel Tower is located in Paris.", doc_id="d4"),
]


class TestSimpleRetriever:
    def setup_method(self):
        self.retriever = SimpleRetriever(list(DOCS))

    def test_returns_documents(self):
        results = self.retriever.retrieve("capital of France", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(d, Document) for d in results)

    def test_top_k_respected(self):
        results = self.retriever.retrieve("capital", top_k=2)
        assert len(results) <= 2

    def test_most_relevant_first(self):
        results = self.retriever.retrieve("France capital Paris", top_k=3)
        doc_ids = [d.doc_id for d in results]
        assert "d1" in doc_ids[:2] or "d4" in doc_ids[:2]

    def test_add_document(self):
        self.retriever.add(Document(content="Rome is the capital of Italy.", doc_id="d5"))
        results = self.retriever.retrieve("capital of Italy", top_k=1)
        assert results[0].doc_id == "d5"

    def test_empty_retriever(self):
        r = SimpleRetriever([])
        assert r.retrieve("anything") == []
