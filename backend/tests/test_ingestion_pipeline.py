"""Steps 2-5: extraction, chunking, embedding, vector storage."""

from __future__ import annotations

import pytest

from app.services.chunking import chunk_document, chunk_text
from app.services.embeddings import HashingEmbedder, cosine_similarity, get_embedder
from app.services.pdf_extract import extract_pdf, split_sections
from app.services.vector_store import VectorStore, get_vector_store


# --- PDF extraction ----------------------------------------------------------


def test_extracts_text_and_pages(sample_pdf):
    document = extract_pdf(sample_pdf)

    assert document.page_count >= 1
    assert document.char_count > 1000
    assert "SparseSum" in document.text
    assert all(page.number == index + 1 for index, page in enumerate(document.pages))


def test_extracts_title_from_largest_font(sample_pdf):
    document = extract_pdf(sample_pdf)
    assert "Sparse Attention Transformers" in document.title


def test_extracts_authors_abstract_and_sections(sample_pdf):
    document = extract_pdf(sample_pdf)

    assert any("Lovelace" in author for author in document.authors)
    assert document.abstract and "quadratically" in document.abstract
    assert "introduction" in document.sections
    assert "limitations" in document.sections
    assert "English scientific text" in document.sections["limitations"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_pdf("does-not-exist.pdf")


def test_split_sections_handles_unstructured_text():
    assert split_sections("just a blob of prose with no headings at all") == {}


# --- chunking ----------------------------------------------------------------


def test_chunking_respects_size_and_overlap(sample_pdf):
    document = extract_pdf(sample_pdf)
    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)  # splitter allows minor overshoot
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_carry_page_and_section_metadata(sample_pdf):
    document = extract_pdf(sample_pdf)
    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)

    assert all(chunk.page_start is not None for chunk in chunks)
    assert any(chunk.section for chunk in chunks)
    assert all(chunk.token_estimate > 0 for chunk in chunks)


def test_chunk_opening_with_a_heading_takes_that_section(sample_pdf):
    """A chunk that starts at "5 Results" belongs to results, not to experiments."""
    document = extract_pdf(sample_pdf)
    chunks = chunk_document(document, chunk_size=400, chunk_overlap=40)

    opening = [chunk for chunk in chunks if chunk.text.lstrip().lower().startswith("5 results")]
    assert opening, "expected at least one chunk to begin at the Results heading"
    assert all(chunk.section == "results" for chunk in opening)


def test_chunking_empty_text_returns_nothing():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# --- embeddings --------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimension=128)
    first, second = embedder.embed_documents(["retrieval augmented generation"] * 2)

    assert first == second
    assert len(first) == 128
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_related_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(dimension=512)
    query = embedder.embed_query("sparse attention for long documents")
    related = embedder.embed_query("sparse attention reduces cost on long documents")
    unrelated = embedder.embed_query("baking sourdough bread at home")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_settings_force_the_offline_embedder():
    embedder = get_embedder()
    assert embedder.name.startswith("hashing-")
    assert embedder.dimension > 0


# --- vector store ------------------------------------------------------------


def test_add_search_and_delete(sample_pdf):
    document = extract_pdf(sample_pdf)
    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)
    ids = [f"chunk-{index}" for index in range(len(chunks))]

    store = get_vector_store()
    added = store.add_chunks("paper-1", chunks, ids, paper_title=document.title)

    assert added == len(chunks)
    assert store.count() == len(chunks)

    hits = store.search("sparse attention routing", top_k=3)
    assert 0 < len(hits) <= 3
    assert all(hit.paper_id == "paper-1" for hit in hits)
    assert hits[0].paper_title == document.title
    assert hits == sorted(hits, key=lambda hit: hit.score, reverse=True)

    store.delete_paper("paper-1")
    assert store.count() == 0
    assert store.search("sparse attention") == []


def test_search_can_be_scoped_to_specific_papers(sample_pdf, second_pdf):
    store = get_vector_store()

    for paper_id, path in (("paper-a", sample_pdf), ("paper-b", second_pdf)):
        document = extract_pdf(path)
        chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)
        store.add_chunks(
            paper_id,
            chunks,
            [f"{paper_id}-{index}" for index in range(len(chunks))],
            paper_title=document.title,
        )

    scoped = store.search("evaluation", top_k=5, paper_ids=["paper-b"])
    assert scoped
    assert {hit.paper_id for hit in scoped} == {"paper-b"}


def test_get_paper_chunks_returns_ordered_chunks(sample_pdf):
    document = extract_pdf(sample_pdf)
    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)
    store = get_vector_store()
    store.add_chunks("paper-x", chunks, [f"x-{i}" for i in range(len(chunks))])

    ordered = store.get_paper_chunks("paper-x")
    assert [chunk.index for chunk in ordered] == list(range(len(chunks)))


def test_mismatched_ids_are_rejected():
    from app.services.chunking import TextChunk

    store = VectorStore()
    chunks = [TextChunk(index=0, text="a"), TextChunk(index=1, text="b")]
    with pytest.raises(ValueError):
        store.add_chunks("p", chunks, ["only-one-id"])
