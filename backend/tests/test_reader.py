"""Reader features: citation highlighting, explain-selection, citation export."""

from __future__ import annotations

import pytest

from app.services.citations import (
    citation_key,
    escape_latex,
    format_citation,
    format_many,
    initials,
    split_name,
)
from app.services.highlight import candidate_phrases, find_highlight


class FakePaper:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title")
        self.authors = kwargs.get("authors", [])
        self.year = kwargs.get("year")
        self.venue = kwargs.get("venue")
        self.doi = kwargs.get("doi")
        self.keywords = kwargs.get("keywords", [])
        self.abstract = kwargs.get("abstract")


PAPER = FakePaper(
    title="Sparse Attention for Long Documents",
    authors=["Ada Lovelace", "Alan Turing", "Grace Hopper"],
    year=2023,
    venue="NeurIPS",
    doi="10.1000/xyz123",
    keywords=["attention", "summarization"],
    abstract="We introduce SparseSum.",
)


# --- name handling -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,family,given",
    [
        ("Ada Lovelace", "Lovelace", "Ada"),
        ("Lovelace, Ada", "Lovelace", "Ada"),
        ("Jean-Luc Picard", "Picard", "Jean-Luc"),
        ("Plato", "Plato", ""),
        ("Ada Byron King Lovelace", "Lovelace", "Ada Byron King"),
    ],
)
def test_names_split_either_way_round(raw, family, given):
    assert split_name(raw) == (family, given)


def test_initials():
    assert initials("Ada Byron") == "A. B."
    assert initials("") == ""


# --- BibTeX ------------------------------------------------------------------


def test_bibtex_has_a_usable_key_and_fields():
    entry = format_citation(PAPER, "bibtex")

    assert entry.startswith("@article{lovelace2023sparse,")
    assert "author = {Lovelace, Ada and Turing, Alan and Hopper, Grace}" in entry
    assert "doi = {10.1000/xyz123}" in entry


def test_bibtex_escapes_latex_specials():
    """An unescaped & or % makes the .bib file fail to compile."""
    paper = FakePaper(title="Cost & Benefit: 50% Faster", authors=["Ada Lovelace"], year=2024)
    entry = format_citation(paper, "bibtex")

    assert r"\&" in entry
    assert r"\%" in entry


def test_citation_key_survives_accents_and_punctuation():
    paper = FakePaper(title="Über Attention", authors=["Émile Borel"], year=2021)
    key = citation_key(paper)

    assert key.isascii()
    assert " " not in key


def test_escape_latex_leaves_plain_text_alone():
    assert escape_latex("plain text") == "plain text"


def test_citation_key_never_contains_a_period():
    """A period in a BibTeX key breaks \\cite in LaTeX.

    A paper with no year produced "baen.d.multimodal" from the "n.d." placeholder.
    """
    undated = FakePaper(title="Multimodal AI for Risk", authors=["Sookyung Bae"], year=None)
    key = citation_key(undated)

    assert "." not in key
    assert key == "baenodatemultimodal"
    assert f"@misc{{{key}," in format_citation(undated, "bibtex")


def test_citation_keys_are_stable():
    assert citation_key(PAPER) == citation_key(PAPER)


# --- other styles ------------------------------------------------------------


def test_apa_format():
    entry = format_citation(PAPER, "apa")
    assert entry.startswith("Lovelace, A., Turing, A., & Hopper, G. (2023).")
    assert "https://doi.org/10.1000/xyz123" in entry


def test_ieee_format():
    entry = format_citation(PAPER, "ieee")
    assert "A. Lovelace" in entry
    assert '"Sparse Attention for Long Documents,"' in entry


def test_mla_uses_et_al_for_multiple_authors():
    assert "et al" in format_citation(PAPER, "mla")


def test_ris_format():
    entry = format_citation(PAPER, "ris")
    assert entry.startswith("TY  - JOUR")
    assert entry.rstrip().endswith("ER  -")
    assert "AU  - Lovelace, Ada" in entry


def test_missing_metadata_does_not_crash():
    bare = FakePaper(title=None, authors=[], year=None)
    for style in ("bibtex", "apa", "ieee", "mla", "ris"):
        assert format_citation(bare, style)


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        format_citation(PAPER, "chicago")


def test_ieee_bibliography_is_numbered():
    body = format_many([PAPER, PAPER], "ieee")
    assert "[1] " in body and "[2] " in body


# --- highlight search --------------------------------------------------------


def test_phrases_prefer_sentence_starts():
    text = (
        "SparseSum reaches 44.1 ROUGE-L on arXiv. "
        "The routing network is trained jointly with the summarizer."
    )
    phrases = candidate_phrases(text)

    assert phrases
    assert any(p.startswith("SparseSum reaches") for p in phrases)
    assert all(len(p) > 12 for p in phrases)


def test_no_phrases_from_empty_text():
    assert candidate_phrases("") == []
    assert candidate_phrases("short") == []


def test_highlight_locates_a_real_passage(sample_pdf):
    """The rectangles are what let a citation be shown in place, not just its page."""
    result = find_highlight(sample_pdf, "SparseSum computes a routing distribution over key blocks")

    assert result.rects, "the passage was not located"
    assert result.page >= 1
    assert result.page_width > 0
    x0, y0, x1, y1 = result.rects[0]
    assert x1 > x0 and y1 > y0


def test_highlight_reports_the_page_even_when_unmatched(sample_pdf):
    """A miss must still navigate to the right page rather than failing."""
    result = find_highlight(sample_pdf, "this sentence appears nowhere at all in the document", page_number=2)

    assert result.rects == []
    assert result.page == 2
    assert result.to_dict()["found"] is False


def test_highlight_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        find_highlight("nope.pdf", "anything")


# --- HTTP --------------------------------------------------------------------


def upload(client, path):
    return client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (path.name, path.read_bytes(), "application/pdf"))],
    ).json()["results"][0]


def test_highlight_endpoint_resolves_a_chunk(client, sample_pdf):
    paper_id = upload(client, sample_pdf)["paper_id"]
    chunks = client.get(f"/api/papers/{paper_id}/chunks", params={"limit": 3}).json()["chunks"]

    payload = client.get(
        "/api/reader/highlight", params={"paper_id": paper_id, "chunk_id": chunks[1]["id"]}
    ).json()

    assert payload["page"] >= 1
    assert payload["found"] is True
    assert payload["rects"]


def test_highlight_endpoint_validates_input(client, sample_pdf):
    paper_id = upload(client, sample_pdf)["paper_id"]

    assert client.get("/api/reader/highlight", params={"paper_id": paper_id}).status_code == 400
    assert client.get("/api/reader/highlight", params={"paper_id": "nope", "text": "x"}).status_code == 404
    assert (
        client.get(
            "/api/reader/highlight", params={"paper_id": paper_id, "chunk_id": "missing"}
        ).status_code
        == 404
    )


def test_citation_endpoints(client, sample_pdf):
    paper_id = upload(client, sample_pdf)["paper_id"]

    single = client.get(f"/api/reader/citation/{paper_id}", params={"style": "bibtex"}).json()
    assert single["style"] == "bibtex"
    assert single["text"].startswith("@")

    bulk = client.get("/api/reader/citations", params={"style": "apa"})
    assert bulk.status_code == 200
    assert "(" in bulk.text

    download = client.get("/api/reader/citations", params={"style": "bibtex", "download": True})
    assert "attachment" in download.headers["content-disposition"]
    assert download.headers["content-disposition"].endswith('.bib"')

    assert client.get("/api/reader/citations", params={"style": "chicago"}).status_code == 400


def test_explain_endpoint(client, sample_pdf, fake_llm):
    paper_id = upload(client, sample_pdf)["paper_id"]

    payload = client.post(
        "/api/reader/explain",
        json={
            "text": "SparseSum computes a routing distribution over key blocks and attends only to the top-k.",
            "level": "simple",
            "paper_id": paper_id,
        },
    ).json()

    assert payload["status"] in ("completed", "partial")


def test_explain_rejects_a_trivial_selection(client):
    assert (
        client.post("/api/reader/explain", json={"text": "short"}).status_code == 422
    )
