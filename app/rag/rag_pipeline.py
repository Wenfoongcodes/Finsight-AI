"""
FinSight AI — Phase 8: RAG Pipeline
Ingests financial documents, generates sentence-transformer embeddings,
stores them in a FAISS index, and retrieves relevant context for LLM grounding.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.exceptions import EmbeddingError, RAGError, VectorStoreError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("rag")


# ─────────────────────────────────────────────────────────────────────────────
# Document Model
# ─────────────────────────────────────────────────────────────────────────────

class Document:
    """Lightweight document container."""

    def __init__(self, content: str, metadata: Optional[dict] = None) -> None:
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(chars={len(self.content)}, meta={self.metadata})"


# ─────────────────────────────────────────────────────────────────────────────
# Text Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits text into overlapping fixed-size chunks suitable for embedding.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """
        Split text into overlapping chunks.

        Args:
            text: Source text to chunk.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of Document objects.
        """
        words = text.split()
        chunks: list[Document] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), max(1, step)):
            chunk_words = words[i: i + self.chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            meta = {**(metadata or {}), "chunk_index": len(chunks), "word_start": i}
            chunks.append(Document(content=chunk_text, metadata=meta))

        logger.debug("Chunked text into %d chunks (size=%d)", len(chunks), self.chunk_size)
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of Document objects."""
        all_chunks: list[Document] = []
        for doc in documents:
            chunks = self.chunk_text(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Generator
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingGenerator:
    """
    Generates dense vector embeddings using sentence-transformers.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings.
            batch_size: Encoding batch size.

        Returns:
            Embeddings array of shape (n_texts, embedding_dim).

        Raises:
            EmbeddingError: On encoding failure.
        """
        try:
            self._load_model()
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-backed vector store for semantic similarity retrieval.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self._index: Optional[Any] = None
        self._documents: list[Document] = []

    def _init_index(self) -> None:
        """Lazy-initialize FAISS flat L2 index."""
        if self._index is not None:
            return
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine w/ normed)
            logger.info("FAISS index initialized (dim=%d)", self.embedding_dim)
        except ImportError as exc:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu") from exc

    def add(self, documents: list[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the index.

        Args:
            documents: List of Document objects.
            embeddings: Corresponding embeddings array.

        Raises:
            VectorStoreError: On indexing failure.
        """
        try:
            self._init_index()
            self._index.add(embeddings)
            self._documents.extend(documents)
            logger.info("Added %d documents. Total: %d", len(documents), len(self._documents))
        except Exception as exc:
            raise VectorStoreError(f"Failed to add documents: {exc}") from exc

    def search(
        self, query_embedding: np.ndarray, top_k: int = settings.RAG_TOP_K
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-k most similar documents.

        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,).
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending similarity.

        Raises:
            VectorStoreError: If index is empty or search fails.
        """
        if self._index is None or len(self._documents) == 0:
            raise VectorStoreError("Vector store is empty. Ingest documents first.")

        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            scores, indices = self._index.search(query_embedding, min(top_k, len(self._documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self._documents[idx], float(score)))
            return results
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

    def save(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and documents to disk."""
        try:
            import faiss
            base = Path(path or settings.VECTOR_DB_PATH)
            base.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(base) + ".faiss")
            with open(str(base) + "_docs.pkl", "wb") as f:
                pickle.dump(self._documents, f)
            logger.info("Vector store saved to %s", base)
        except Exception as exc:
            raise VectorStoreError(f"Save failed: {exc}") from exc

    def load(self, path: Optional[str] = None) -> None:
        """Load FAISS index and documents from disk."""
        try:
            import faiss
            base = Path(path or settings.VECTOR_DB_PATH)
            self._index = faiss.read_index(str(base) + ".faiss")
            with open(str(base) + "_docs.pkl", "rb") as f:
                self._documents = pickle.load(f)
            logger.info("Vector store loaded: %d documents", len(self._documents))
        except Exception as exc:
            raise VectorStoreError(f"Load failed: {exc}") from exc

    @property
    def size(self) -> int:
        return len(self._documents)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Combines chunking, embedding, vector storage, and context retrieval
    for grounded LLM responses.
    """

    def __init__(self) -> None:
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self._store_initialized = False

    def ingest(self, documents: list[Document]) -> None:
        """
        Process and index a list of documents.

        Args:
            documents: Source documents to ingest.

        Raises:
            RAGError: On ingestion failure.
        """
        try:
            chunks = self.chunker.chunk_documents(documents)
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed(texts)

            if self.vector_store.embedding_dim != embeddings.shape[1]:
                self.vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])

            self.vector_store.add(chunks, embeddings)
            self._store_initialized = True
            logger.info("Ingested %d source docs → %d chunks indexed", len(documents), len(chunks))
        except Exception as exc:
            raise RAGError(f"Ingestion failed: {exc}") from exc

    def ingest_texts(self, texts: list[str], source: str = "manual") -> None:
        """Convenience wrapper to ingest raw text strings."""
        docs = [Document(content=t, metadata={"source": source}) for t in texts]
        self.ingest(docs)

    def ingest_file(self, file_path: str) -> None:
        """
        Ingest a text (.txt) or JSON (.json) file into the RAG index.

        Args:
            file_path: Path to file.
        """
        path = Path(file_path)
        if not path.exists():
            raise RAGError(f"File not found: {file_path}")

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                texts = [str(d) for d in data]
            else:
                texts = [json.dumps(data)]
        else:
            texts = [path.read_text(encoding="utf-8")]

        self.ingest_texts(texts, source=path.name)

    def retrieve(self, query: str, top_k: int = settings.RAG_TOP_K) -> list[dict]:
        """
        Retrieve top-k documents most relevant to a query.

        Args:
            query: Natural language query.
            top_k: Number of documents to return.

        Returns:
            List of dicts with 'content', 'score', and 'metadata'.
        """
        try:
            query_emb = self.embedder.embed([query])
            results = self.vector_store.search(query_emb, top_k=top_k)
            return [
                {
                    "content": doc.content,
                    "score": round(score, 4),
                    "metadata": doc.metadata,
                }
                for doc, score in results
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise RAGError(f"Retrieval failed: {exc}") from exc

    def build_context(self, query: str, top_k: int = settings.RAG_TOP_K) -> str:
        """
        Build a formatted context string for LLM injection.

        Args:
            query: User query.
            top_k: Number of context chunks.

        Returns:
            Formatted context string.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant context found."

        parts = ["Relevant financial context:\n"]
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "unknown")
            parts.append(f"[{i}] (source: {src}, score: {r['score']})\n{r['content']}\n")

        return "\n".join(parts)

    def save(self, path: Optional[str] = None) -> None:
        self.vector_store.save(path)

    def load(self, path: Optional[str] = None) -> None:
        self.vector_store.load(path)
        self._store_initialized = True
