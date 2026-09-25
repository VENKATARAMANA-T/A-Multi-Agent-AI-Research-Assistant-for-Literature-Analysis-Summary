"""Step 5 — persistent vector storage and semantic retrieval (ChromaDB).

Embeddings are computed by our own `Embedder` and handed to Chroma explicitly,
so the embedding backend stays swappable and Chroma never tries to download a
model of its own.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.chunking import TextChunk
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    text: str
    score: float
    index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    paper_title: str | None = None
    source: str = "native"
    kind: str = "text"              # "text" or "figure"
    figure_id: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "text": self.text,
            "score": round(self.score, 4),
            "index": self.index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "source": self.source,
            "kind": self.kind,
            "figure_id": self.figure_id,
            "label": self.label,
        }

    def citation(self) -> str:
        title = self.paper_title or self.paper_id[:8]
        if self.page_start:
            return f"{title}, p.{self.page_start}"
        return title


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        self._persist_dir = str(persist_dir or settings.chroma_dir)
        self._collection_name = collection_name or settings.chroma_collection
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", **self._backend_stamp()},
        )
        self._check_embedding_backend()

    @staticmethod
    def _backend_stamp() -> dict[str, str]:
        try:
            return {"embedding_backend": get_embedder().name}
        except Exception:  # pragma: no cover - embedder failures surface elsewhere
            return {}

    def _check_embedding_backend(self) -> None:
        """Guard against querying vectors written by a different embedder.

        Two backends can share a dimension (MiniLM and the hashing fallback are
        both 384-d) while living in completely different vector spaces, so a
        silent switch would return plausible-looking nonsense. We stamp the
        collection with the backend that wrote it and loudly warn on a mismatch.
        """
        try:
            current = get_embedder().name
        except Exception:  # pragma: no cover - embedder failures surface elsewhere
            return

        metadata = dict(self._collection.metadata or {})
        recorded = metadata.get("embedding_backend")

        if recorded is None:
            if self._collection.count() == 0:
                # Chroma rejects hnsw:* keys on modify(); they are immutable anyway.
                mutable = {k: v for k, v in metadata.items() if not k.startswith("hnsw:")}
                self._collection.modify(metadata={**mutable, "embedding_backend": current})
            return

        if recorded != current:
            logger.error(
                "Vector store was built with embedding backend '%s' but '%s' is active. "
                "Existing vectors are not comparable to new queries — reindex every paper "
                "(POST /api/papers/{id}/reindex) or delete %s to rebuild the collection.",
                recorded,
                current,
                self._persist_dir,
            )
        self._embedding_backend_mismatch = recorded != current

    @property
    def embedding_backend_mismatch(self) -> bool:
        return getattr(self, "_embedding_backend_mismatch", False)

    # --- writes --------------------------------------------------------------

    def add_chunks(
        self,
        paper_id: str,
        chunks: list[TextChunk],
        chunk_ids: list[str],
        paper_title: str | None = None,
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(chunk_ids):
            raise ValueError("chunks and chunk_ids must be the same length")

        embedder = get_embedder()
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)

        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            metadatas.append(
                {
                    "paper_id": paper_id,
                    "paper_title": paper_title or "",
                    "index": chunk.index,
                    "page_start": chunk.page_start if chunk.page_start is not None else -1,
                    "page_end": chunk.page_end if chunk.page_end is not None else -1,
                    "section": chunk.section or "",
                    "source": chunk.source or "native",
                }
            )

        # Chroma has a per-call batch ceiling; stay well under it.
        batch = 256
        for start in range(0, len(texts), batch):
            stop = start + batch
            self._collection.upsert(
                ids=chunk_ids[start:stop],
                documents=texts[start:stop],
                embeddings=vectors[start:stop],
                metadatas=metadatas[start:stop],
            )
        return len(texts)

    def add_figures(self, figures: list[Any], paper_title: str | None = None) -> int:
        """Index figures in the same collection as text, tagged `kind=figure`.

        A chart's meaning lives in its caption and its analysis, so embedding
        those makes "which paper shows accuracy dropping after epoch 50?"
        answerable — something no text chunk contains.
        """
        if not figures:
            return 0

        documents = [figure.search_text for figure in figures]
        ids = [f"figure:{figure.id}" for figure in figures]
        vectors = get_embedder().embed_documents(documents)

        metadatas = [
            {
                "paper_id": figure.paper_id,
                "paper_title": paper_title or "",
                "kind": "figure",
                "figure_id": figure.id,
                "figure_kind": figure.kind,
                "label": figure.label or "",
                "index": 0,
                "page_start": figure.page or -1,
                "page_end": figure.page or -1,
                "section": "",
                "source": "figure",
            }
            for figure in figures
        ]

        batch = 128
        for start in range(0, len(documents), batch):
            stop = start + batch
            self._collection.upsert(
                ids=ids[start:stop],
                documents=documents[start:stop],
                embeddings=vectors[start:stop],
                metadatas=metadatas[start:stop],
            )
        return len(documents)

    def delete_paper_figures(self, paper_id: str) -> None:
        self._collection.delete(where={"$and": [{"paper_id": paper_id}, {"kind": "figure"}]})

    def delete_paper(self, paper_id: str) -> None:
        self._collection.delete(where={"paper_id": paper_id})

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", **self._backend_stamp()},
        )
        self._embedding_backend_mismatch = False

    # --- reads ---------------------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k

        where: dict[str, Any] | None = None
        if paper_ids:
            where = {"paper_id": {"$in": list(paper_ids)}} if len(paper_ids) > 1 else {"paper_id": paper_ids[0]}

        vector = get_embedder().embed_query(query)
        try:
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Querying an empty collection raises rather than returning nothing.
            # Checking count() first would cost an extra round trip on every
            # search just to handle the empty case, so handle it here instead.
            if self.count() == 0:
                return []
            raise

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] or {}
            distance = distances[i] if i < len(distances) else 1.0
            page_start = meta.get("page_start", -1)
            page_end = meta.get("page_end", -1)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    paper_id=str(meta.get("paper_id", "")),
                    paper_title=str(meta.get("paper_title") or "") or None,
                    text=documents[i] if i < len(documents) else "",
                    score=round(1.0 - float(distance), 4),  # cosine distance -> similarity
                    index=int(meta.get("index", 0) or 0),
                    page_start=int(page_start) if page_start not in (None, -1) else None,
                    page_end=int(page_end) if page_end not in (None, -1) else None,
                    section=str(meta.get("section") or "") or None,
                    source=str(meta.get("source") or "native"),
                    kind=str(meta.get("kind") or "text"),
                    figure_id=str(meta.get("figure_id") or "") or None,
                    label=str(meta.get("label") or "") or None,
                )
            )
        return retrieved

    def get_paper_chunks(self, paper_id: str, limit: int = 500) -> list[RetrievedChunk]:
        """Ordered chunks for one paper — used by summarisation and extraction."""
        result = self._collection.get(
            where={"paper_id": paper_id},
            limit=limit,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks = [
            RetrievedChunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title=str((metadatas[i] or {}).get("paper_title") or "") or None,
                text=documents[i] if i < len(documents) else "",
                score=1.0,
                index=int((metadatas[i] or {}).get("index", 0) or 0),
                section=str((metadatas[i] or {}).get("section") or "") or None,
            )
            for i, chunk_id in enumerate(ids)
        ]
        return sorted(chunks, key=lambda c: c.index)


_store: VectorStore | None = None
_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Test hook."""
    global _store
    with _lock:
        _store = None
