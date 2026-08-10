"""Step 4 — embedding generation.

Primary backend is HuggingFace `sentence-transformers` (all-MiniLM-L6-v2).
If the model cannot be loaded (no network, no torch, air-gapped CI), we fall
back to a deterministic hashed-bigram embedder. Retrieval quality drops, but
the whole pipeline — ingestion, vector search, agents — keeps working, and the
active backend is reported through /api/health so the degradation is visible.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


class HashingEmbedder(Embedder):
    """Deterministic offline embedder: L2-normalised hashed unigrams + bigrams."""

    def __init__(self, dimension: int = 384) -> None:
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
        return vectors


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Process-wide singleton; loading the transformer model is expensive."""
    global _embedder
    if _embedder is not None:
        return _embedder

    with _lock:
        if _embedder is not None:
            return _embedder

        if settings.embedding_offline_fallback:
            logger.warning("EMBEDDING_OFFLINE_FALLBACK=true — using the hashing embedder.")
            _embedder = HashingEmbedder()
            return _embedder

        try:
            _embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
            logger.info("Loaded sentence-transformers model %s (dim=%d)", _embedder.name, _embedder.dimension)
        except Exception as exc:  # pragma: no cover - depends on the host
            logger.warning(
                "Could not load '%s' (%s). Falling back to the hashing embedder; "
                "semantic search quality will be reduced.",
                settings.embedding_model,
                exc,
            )
            _embedder = HashingEmbedder()
        return _embedder


def reset_embedder() -> None:
    """Test hook."""
    global _embedder
    with _lock:
        _embedder = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
