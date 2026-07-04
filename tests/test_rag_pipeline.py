"""
Unit tests for ``app.rag.rag_pipeline``.

Covers:
- ``Document``        — basic container behaviour.
- ``TextChunker``      — word-based chunking with overlap, edge cases.
- ``FAISSVectorStore`` — add/search/size, with ``faiss`` mocked out so the
                          real dependency need not be installed.
- ``RAGPipeline``      — retrieve/build_context orchestration and the
                          uninitialized-store guard, with the embedder and
                          vector store mocked so no real model download or
                          FAISS index is required.

Note: ``faiss`` and ``sentence-transformers`` are imported lazily inside
the relevant methods (not at module import time), so importing this test
module does not require either package to be installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.exceptions import RAGError, VectorStoreError
from app.rag.rag_pipeline import Document, FAISSVectorStore, RAGPipeline, TextChunker

# ─────────────────────────────────────────────────────────────────────────────
# Document
# ─────────────────────────────────────────────────────────────────────────────


class TestDocument:
    def test_stores_content_and_metadata(self):
        doc = Document("hello world", metadata={"source": "test"})
        assert doc.content == "hello world"
        assert doc.metadata == {"source": "test"}

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document("hello")
        assert doc.metadata == {}

    def test_repr_includes_char_count(self):
        doc = Document("12345")
        assert "chars=5" in repr(doc)


# ─────────────────────────────────────────────────────────────────────────────
# TextChunker
# ─────────────────────────────────────────────────────────────────────────────


class TestTextChunker:
    def test_short_text_produces_single_chunk(self):
        chunker = TextChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_text("a short piece of text")
        assert len(chunks) == 1
        assert chunks[0].content == "a short piece of text"

    def test_long_text_is_split_into_multiple_chunks(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = " ".join(f"word{i}" for i in range(50))
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        # Every chunk should have at most chunk_size words.
        for c in chunks:
            assert len(c.content.split()) <= 10

    def test_overlap_causes_shared_words_between_consecutive_chunks(self):
        chunker = TextChunker(chunk_size=10, overlap=4)
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunker.chunk_text(text)

        first_words = chunks[0].content.split()
        second_words = chunks[1].content.split()
        overlap_words = set(first_words) & set(second_words)
        assert len(overlap_words) > 0

    def test_chunk_metadata_includes_index_and_word_start(self):
        chunker = TextChunker(chunk_size=5, overlap=0)
        text = " ".join(f"w{i}" for i in range(12))
        chunks = chunker.chunk_text(text, metadata={"source": "doc1"})

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["source"] == "doc1"
            assert "word_start" in chunk.metadata

    def test_empty_text_produces_no_chunks(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        assert chunker.chunk_text("") == []

    def test_overlap_larger_than_chunk_size_still_makes_progress(self):
        """
        Regression guard: step = max(1, chunk_size - overlap) must never be
        <= 0, otherwise chunking would loop forever on pathological configs.
        """
        chunker = TextChunker(chunk_size=5, overlap=100)
        text = " ".join(f"w{i}" for i in range(20))
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0  # completes without hanging

    def test_chunk_documents_aggregates_across_multiple_docs(self):
        chunker = TextChunker(chunk_size=100, overlap=0)
        docs = [Document("first document"), Document("second document")]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 2
        assert chunks[0].content == "first document"
        assert chunks[1].content == "second document"


# ─────────────────────────────────────────────────────────────────────────────
# FAISSVectorStore (faiss mocked)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_faiss(monkeypatch):
    """Install a minimal fake ``faiss`` module backed by numpy cosine search."""

    class _FakeIndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
            self._vectors = np.empty((0, dim), dtype=np.float32)

        def add(self, embeddings):
            self._vectors = np.vstack([self._vectors, embeddings])

        def search(self, query, k):
            if self._vectors.shape[0] == 0:
                return np.array([[]]), np.array([[]], dtype=int)
            scores = self._vectors @ query[0]
            top_k_idx = np.argsort(-scores)[:k]
            return (
                scores[top_k_idx].reshape(1, -1),
                top_k_idx.reshape(1, -1),
            )

    fake_module = SimpleNamespace(
        IndexFlatIP=_FakeIndexFlatIP,
        write_index=MagicMock(),
        read_index=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_module)
    yield fake_module


class TestFAISSVectorStore:
    def test_search_on_empty_store_raises_vector_store_error(self, fake_faiss):
        store = FAISSVectorStore(embedding_dim=4)
        query = np.random.rand(4).astype(np.float32)
        with pytest.raises(VectorStoreError):
            store.search(query)

    def test_add_and_search_returns_documents_with_scores(self, fake_faiss):
        store = FAISSVectorStore(embedding_dim=4)
        docs = [Document("doc a"), Document("doc b")]
        embeddings = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        store.add(docs, embeddings)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)

        assert len(results) == 1
        doc, score = results[0]
        assert doc.content == "doc a"
        assert score == pytest.approx(1.0)

    def test_size_reflects_number_of_added_documents(self, fake_faiss):
        store = FAISSVectorStore(embedding_dim=2)
        docs = [Document("x"), Document("y"), Document("z")]
        embeddings = np.random.rand(3, 2).astype(np.float32)
        store.add(docs, embeddings)
        assert store.size == 3

    def test_add_wraps_underlying_errors_in_vector_store_error(
        self, fake_faiss, monkeypatch
    ):
        store = FAISSVectorStore(embedding_dim=4)
        store._init_index()
        monkeypatch.setattr(
            store._index, "add", MagicMock(side_effect=RuntimeError("bad shape"))
        )
        with pytest.raises(VectorStoreError):
            store.add([Document("x")], np.zeros((1, 4), dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline (embedder + vector_store mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestRAGPipelineRetrieve:
    def _pipeline_with_mocks(self):
        pipeline = RAGPipeline()
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed.return_value = np.zeros((1, 4), dtype=np.float32)
        pipeline.vector_store = MagicMock()
        return pipeline

    def test_retrieve_raises_when_store_not_initialized(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = False
        with pytest.raises(RAGError):
            pipeline.retrieve("what is the outlook for AAPL?")

    def test_retrieve_returns_formatted_results(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        doc = Document("AAPL guidance raised", metadata={"source": "reuters.com"})
        pipeline.vector_store.search.return_value = [(doc, 0.87654)]

        results = pipeline.retrieve("AAPL guidance")

        assert len(results) == 1
        assert results[0]["content"] == "AAPL guidance raised"
        assert results[0]["score"] == 0.8765  # rounded to 4 dp
        assert results[0]["metadata"]["source"] == "reuters.com"

    def test_retrieve_wraps_vector_store_error_passthrough(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        pipeline.vector_store.search.side_effect = VectorStoreError("index corrupted")

        with pytest.raises(VectorStoreError):
            pipeline.retrieve("query")

    def test_retrieve_wraps_unexpected_errors_in_rag_error(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        pipeline.embedder.embed.side_effect = RuntimeError("embedding blew up")

        with pytest.raises(RAGError):
            pipeline.retrieve("query")


class TestRAGPipelineBuildContext:
    def _pipeline_with_mocks(self):
        pipeline = RAGPipeline()
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed.return_value = np.zeros((1, 4), dtype=np.float32)
        pipeline.vector_store = MagicMock()
        return pipeline

    def test_build_context_returns_placeholder_when_uninitialized(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = False
        context = pipeline.build_context("any query")
        assert context == "No relevant context found."

    def test_build_context_returns_placeholder_when_no_results(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        pipeline.vector_store.search.return_value = []
        context = pipeline.build_context("any query")
        assert context == "No relevant context found."

    def test_build_context_formats_results_with_source_and_score(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        doc = Document(
            "Fed signals rate pause",
            metadata={"url": "https://bloomberg.com/x", "title": "Fed Watch"},
        )
        pipeline.vector_store.search.return_value = [(doc, 0.9)]

        context = pipeline.build_context("fed rates")

        assert "Fed Watch" in context
        assert "bloomberg.com" in context
        assert "Fed signals rate pause" in context

    def test_build_context_falls_back_to_source_when_no_title(self):
        pipeline = self._pipeline_with_mocks()
        pipeline._store_initialized = True
        doc = Document("Some content", metadata={"source": "manual"})
        pipeline.vector_store.search.return_value = [(doc, 0.5)]

        context = pipeline.build_context("query")
        assert "manual" in context


class TestRAGPipelineIngestTexts:
    def test_ingest_texts_builds_documents_and_calls_ingest(self, monkeypatch):
        pipeline = RAGPipeline()
        called_with = {}

        def fake_ingest(documents):
            called_with["documents"] = documents

        monkeypatch.setattr(pipeline, "ingest", fake_ingest)
        pipeline.ingest_texts(["text one", "text two"], source="unit-test")

        docs = called_with["documents"]
        assert len(docs) == 2
        assert all(d.metadata["source"] == "unit-test" for d in docs)
        assert docs[0].content == "text one"

    def test_ingest_wraps_failures_in_rag_error(self, monkeypatch):
        pipeline = RAGPipeline()
        pipeline.chunker = MagicMock()
        pipeline.chunker.chunk_documents.side_effect = RuntimeError("chunking failed")

        with pytest.raises(RAGError):
            pipeline.ingest([Document("x")])


class TestRAGPipelineIntrospection:
    def test_document_count_delegates_to_vector_store_size(self):
        pipeline = RAGPipeline()
        pipeline.vector_store = MagicMock()
        pipeline.vector_store.size = 7
        assert pipeline.document_count == 7

    def test_ingested_url_count_reflects_registry_size(self):
        pipeline = RAGPipeline()
        pipeline._ingested_urls = {"https://a.com": "ts1", "https://b.com": "ts2"}
        assert pipeline.ingested_url_count == 2
