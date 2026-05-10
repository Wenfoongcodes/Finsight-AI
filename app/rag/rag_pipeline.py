"""
FinSight AI — Phase 8: RAG Pipeline
Ingests financial documents, generates sentence-transformer embeddings,
stores them in a FAISS index, and retrieves relevant context for LLM grounding.

New in this revision
--------------------
* ``WebArticleFetcher`` — fetches a URL and extracts clean article text using
  ``requests`` + ``BeautifulSoup`` + ``lxml``.  Handles noise-tag removal,
  semantic container detection, title extraction, and a minimum-length guard
  so bot-blocked or nav-only pages are rejected with a clear ``RAGError``
  instead of silently ingesting garbage.

* ``RAGPipeline.ingest_url()`` — public entry-point for URL ingestion.
  Stores the source URL and fetched timestamp in each ``Document``'s metadata
  so retrieved context can be traced back to its origin.

* URL deduplication — ``RAGPipeline`` keeps a ``_ingested_urls`` set.  Calling
  ``ingest_url()`` twice with the same URL is a no-op (logged at INFO), which
  prevents duplicate embeddings from accumulating in the FAISS index.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np

from app.core.exceptions import EmbeddingError, RAGError, VectorStoreError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("rag")

# Minimum article text length; shorter responses are treated as blocked/nav pages.
_MIN_ARTICLE_CHARS = 200

# HTTP request timeout for URL fetching (seconds).
_FETCH_TIMEOUT = 15

# Realistic browser User-Agent to avoid trivial bot blocks.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# HTML tags that contain only navigation, ads, or boilerplate — always removed.
_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "table", "sup", "figure", "figcaption", "iframe",
    "noscript", "button", "input", "select", "textarea",
]


# ─────────────────────────────────────────────────────────────────────────────
# Web Article Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class WebArticleFetcher:
    """
    Fetches a web URL and extracts clean article text.

    Strategy
    --------
    1. Download the page with a realistic User-Agent and a hard timeout.
    2. Parse with ``lxml`` (fastest, most lenient HTML parser available).
    3. Strip all noise tags (scripts, ads, nav, footer, etc.).
    4. Detect the semantic article container using a priority cascade:
       ``<article>`` → ``<main>`` → ``<div id|class ~= content/article/story>``
       → ``<body>`` (fallback).
    5. Extract plain text, collapse whitespace, strip leading/trailing spaces.
    6. Reject pages shorter than ``_MIN_ARTICLE_CHARS`` — these are almost
       always bot-blocked responses, login walls, or navigation-only pages.

    Raises:
        ImportError: If ``requests`` or ``bs4`` are not installed.
        RAGError: On network error, HTTP error, or insufficient content.
    """

    def fetch(self, url: str) -> tuple[str, str]:
        """
        Download and extract article text from *url*.

        Args:
            url: Fully-qualified URL (must start with http/https).

        Returns:
            ``(title, body_text)`` — both are clean plain-text strings.

        Raises:
            RAGError: On any fetch or extraction failure.
        """
        self._validate_url(url)

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is required for URL ingestion: pip install requests"
            ) from exc

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 + lxml are required: pip install beautifulsoup4 lxml"
            ) from exc

        # ── Fetch ────────────────────────────────────────────────────────────
        try:
            response = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_FETCH_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RAGError(
                f"URL fetch timed out after {_FETCH_TIMEOUT}s: {url}",
                detail="The server did not respond in time.",
            )
        except requests.exceptions.HTTPError as exc:
            raise RAGError(
                f"HTTP {exc.response.status_code} fetching {url}",
                detail=str(exc),
            )
        except Exception as exc:
            raise RAGError(f"Failed to fetch {url}: {exc}", detail=str(exc))

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as exc:
            raise RAGError(f"HTML parsing failed for {url}: {exc}")

        # ── Extract title ─────────────────────────────────────────────────────
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # ── Remove noise tags ─────────────────────────────────────────────────
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # ── Detect semantic article container ─────────────────────────────────
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find(
                "div",
                {"id": re.compile(r"content|article|story|body|post", re.I)},
            )
            or soup.find(
                "div",
                {"class": re.compile(r"content|article|story|body|post", re.I)},
            )
            or soup.body
        )

        raw_text = container.get_text(separator=" ", strip=True) if container else ""

        # ── Clean whitespace ──────────────────────────────────────────────────
        text = re.sub(r"\s+", " ", raw_text).strip()

        # ── Content length guard ──────────────────────────────────────────────
        if len(text) < _MIN_ARTICLE_CHARS:
            raise RAGError(
                f"Extracted only {len(text)} characters from {url}. "
                f"Minimum is {_MIN_ARTICLE_CHARS}.",
                detail=(
                    "The page may be behind a login wall, returning a CAPTCHA, "
                    "or blocking automated requests."
                ),
            )

        logger.info(
            "Article fetched: url=%s title=%r chars=%d", url, title[:60], len(text)
        )
        return title, text

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_url(url: str) -> None:
        """Reject obviously invalid URLs before making a network request."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise RAGError(
                f"Invalid URL scheme {parsed.scheme!r} in {url!r}. "
                "Only http:// and https:// are supported."
            )
        if not parsed.netloc:
            raise RAGError(f"URL has no host: {url!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Document Model
# ─────────────────────────────────────────────────────────────────────────────

class Document:
    """Lightweight document container with rich metadata support."""

    def __init__(self, content: str, metadata: Optional[dict] = None) -> None:
        self.content  = content
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
        self.overlap    = overlap

    def chunk_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> list[Document]:
        """
        Split text into overlapping word-level chunks.

        Args:
            text: Source text.
            metadata: Metadata attached to every chunk (e.g. source URL).

        Returns:
            List of ``Document`` objects.
        """
        words  = text.split()
        chunks: list[Document] = []
        step   = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if not chunk_words:
                break
            meta = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "word_start":  i,
            }
            chunks.append(Document(content=" ".join(chunk_words), metadata=meta))

        logger.debug(
            "Chunked text into %d chunks (size=%d, overlap=%d)",
            len(chunks), self.chunk_size, self.overlap,
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of ``Document`` objects."""
        all_chunks: list[Document] = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc.content, metadata=doc.metadata))
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
        Generate normalized embeddings for a list of texts.

        Args:
            texts: List of text strings.
            batch_size: Encoding batch size.

        Returns:
            Float32 array of shape ``(n_texts, embedding_dim)``.
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
        if self._index is not None:
            return
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("FAISS index initialized (dim=%d)", self.embedding_dim)
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required: pip install faiss-cpu"
            ) from exc

    def add(self, documents: list[Document], embeddings: np.ndarray) -> None:
        try:
            self._init_index()
            self._index.add(embeddings)
            self._documents.extend(documents)
            logger.info(
                "Added %d documents. Total: %d", len(documents), len(self._documents)
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to add documents: {exc}") from exc

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = settings.RAG_TOP_K,
    ) -> list[tuple[Document, float]]:
        if self._index is None or len(self._documents) == 0:
            raise VectorStoreError(
                "Vector store is empty. Ingest documents first."
            )
        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            scores, indices = self._index.search(
                query_embedding, min(top_k, len(self._documents))
            )
            return [
                (self._documents[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

    def save(self, path: Optional[str] = None) -> None:
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
        try:
            import faiss
            base = Path(path or settings.VECTOR_DB_PATH)
            self._index = faiss.read_index(str(base) + ".faiss")
            with open(str(base) + "_docs.pkl", "rb") as f:
                self._documents = pickle.load(f)
            logger.info(
                "Vector store loaded: %d documents", len(self._documents)
            )
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

    Combines chunking, embedding, vector storage, context retrieval,
    and URL-based web article ingestion.
    """

    def __init__(self) -> None:
        self.chunker      = TextChunker()
        self.embedder     = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self._fetcher     = WebArticleFetcher()

        self._store_initialized = False

        # Track ingested URLs to prevent duplicate embeddings.
        # Key: normalized URL string.  Value: ISO timestamp of first ingestion.
        self._ingested_urls: dict[str, str] = {}

    # ── Core ingestion ────────────────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        """
        Process and index a list of ``Document`` objects.

        Args:
            documents: Source documents to chunk, embed, and store.

        Raises:
            RAGError: On any ingestion failure.
        """
        try:
            chunks     = self.chunker.chunk_documents(documents)
            texts      = [c.content for c in chunks]
            embeddings = self.embedder.embed(texts)

            if self.vector_store.embedding_dim != embeddings.shape[1]:
                self.vector_store = FAISSVectorStore(
                    embedding_dim=embeddings.shape[1]
                )

            self.vector_store.add(chunks, embeddings)
            self._store_initialized = True
            logger.info(
                "Ingested %d source docs → %d chunks indexed",
                len(documents), len(chunks),
            )
        except Exception as exc:
            raise RAGError(f"Ingestion failed: {exc}") from exc

    def ingest_texts(
        self, texts: list[str], source: str = "manual"
    ) -> None:
        """Convenience wrapper to ingest raw text strings."""
        docs = [
            Document(content=t, metadata={"source": source})
            for t in texts
        ]
        self.ingest(docs)

    def ingest_url(self, url: str) -> dict:
        """
        Fetch a web article by URL and ingest it into the knowledge base.

        The article's URL, title, fetch timestamp, and character count are
        stored in every chunk's metadata so retrieved context can always be
        traced back to its origin.

        Duplicate ingestion is silently skipped: calling this method twice
        with the same URL returns the original ingestion record without
        adding new embeddings to the index.

        Args:
            url: Fully-qualified article URL (http or https).

        Returns:
            Dict with ingestion metadata::

                {
                    "url":         "https://...",
                    "title":       "Article Title",
                    "char_count":  1842,
                    "chunks":      12,
                    "fetched_at":  "2026-05-10T09:41:22+00:00",
                    "duplicate":   False,
                }

        Raises:
            RAGError: On fetch failure, HTTP error, or insufficient content.
        """
        # Normalise: strip trailing slash, lowercase scheme+host
        normalized = url.strip().rstrip("/")

        if normalized in self._ingested_urls:
            logger.info("URL already ingested — skipping: %s", normalized)
            return {
                "url":        normalized,
                "title":      "",
                "char_count": 0,
                "chunks":     0,
                "fetched_at": self._ingested_urls[normalized],
                "duplicate":  True,
            }

        fetched_at = datetime.now(timezone.utc).isoformat()
        title, text = self._fetcher.fetch(normalized)

        metadata = {
            "source":     "url",
            "url":        normalized,
            "title":      title,
            "fetched_at": fetched_at,
        }

        chunks_before = self.vector_store.size
        self.ingest([Document(content=text, metadata=metadata)])
        chunks_added = self.vector_store.size - chunks_before

        self._ingested_urls[normalized] = fetched_at

        result = {
            "url":        normalized,
            "title":      title,
            "char_count": len(text),
            "chunks":     chunks_added,
            "fetched_at": fetched_at,
            "duplicate":  False,
        }
        logger.info(
            "URL ingested: %s | title=%r | chars=%d | chunks=%d",
            normalized, title[:60], len(text), chunks_added,
        )
        return result

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
            texts = [str(d) for d in data] if isinstance(data, list) else [json.dumps(data)]
        else:
            texts = [path.read_text(encoding="utf-8")]

        self.ingest_texts(texts, source=path.name)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = settings.RAG_TOP_K) -> list[dict]:
        """
        Retrieve the top-k documents most relevant to *query*.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of dicts with ``'content'``, ``'score'``, and ``'metadata'``.

        Raises:
            RAGError: If the store is uninitialized or retrieval fails.
        """
        if not self._store_initialized:
            raise RAGError(
                "Knowledge base is empty. Ingest documents or a URL first."
            )
        try:
            query_emb = self.embedder.embed([query])
            results   = self.vector_store.search(query_emb, top_k=top_k)
            return [
                {
                    "content":  doc.content,
                    "score":    round(score, 4),
                    "metadata": doc.metadata,
                }
                for doc, score in results
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise RAGError(f"Retrieval failed: {exc}") from exc

    def build_context(
        self, query: str, top_k: int = settings.RAG_TOP_K
    ) -> str:
        """
        Build a formatted context string for LLM injection.

        Args:
            query: User query.
            top_k: Number of context chunks.

        Returns:
            Formatted context string, or ``"No relevant context found."``
            when the store is empty or the query matches nothing.
        """
        if not self._store_initialized:
            return "No relevant context found."

        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant context found."

        parts = ["Relevant financial context:\n"]
        for i, r in enumerate(results, 1):
            meta  = r["metadata"]
            src   = meta.get("url") or meta.get("source", "unknown")
            title = meta.get("title", "")
            label = f"{title} ({src})" if title else src
            parts.append(
                f"[{i}] (source: {label}, score: {r['score']})\n{r['content']}\n"
            )
        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        self.vector_store.save(path)

    def load(self, path: Optional[str] = None) -> None:
        self.vector_store.load(path)
        self._store_initialized = True

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def ingested_url_count(self) -> int:
        """Number of unique URLs ingested in this session."""
        return len(self._ingested_urls)

    @property
    def document_count(self) -> int:
        """Total number of chunks currently in the vector store."""
        return self.vector_store.size