"""A collection written by one embedder must not be silently queried by another."""

from __future__ import annotations

from app.services import embeddings as embeddings_module
from app.services.chunking import TextChunk
from app.services.vector_store import VectorStore


def test_backend_is_stamped_on_a_fresh_collection():
    store = VectorStore()
    assert store.embedding_backend_mismatch is False
    assert dict(store._collection.metadata)["embedding_backend"].startswith("hashing-")


def test_mismatch_is_detected_and_reported(monkeypatch, caplog):
    store = VectorStore()
    store.add_chunks("p1", [TextChunk(index=0, text="some text")], ["c1"])

    # Simulate restarting with a different embedding backend.
    monkeypatch.setattr(
        embeddings_module, "get_embedder", lambda: embeddings_module.HashingEmbedder(dimension=384)
    )
    monkeypatch.setattr(
        "app.services.vector_store.get_embedder",
        lambda: type("E", (), {"name": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384})(),
    )

    with caplog.at_level("ERROR"):
        reopened = VectorStore()

    assert reopened.embedding_backend_mismatch is True
    assert "not comparable" in caplog.text


def test_reset_restamps_the_collection():
    store = VectorStore()
    store.add_chunks("p1", [TextChunk(index=0, text="some text")], ["c1"])
    store.reset()

    assert store.count() == 0
    assert store.embedding_backend_mismatch is False
    assert dict(store._collection.metadata)["embedding_backend"].startswith("hashing-")
