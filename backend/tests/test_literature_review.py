"""The Literature Review Agent.

The thing worth testing hardest is the citation handling. A review whose
markers point at papers that were never supplied looks authoritative and
resolves to nothing — which is worse than a review with no citations at all,
because the reader has no reason to check.
"""

from __future__ import annotations

import pytest

from app.agents.literature_review import (
    GROUPS,
    SECTION_ORDER,
    build_markdown,
    format_reference,
    generate,
    numbered_context,
    resolve_citations,
)
from app.database import session_scope
from app.services.ingestion import process_paper, store_upload
from app.services.llm import set_llm
from tests.conftest import FakeGemini, seed_user_id


def ingest(path) -> str:
    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        return paper.id


SECTIONS = {
    "framing": {
        "introduction": "Speech models matter [S1]. This review covers two papers [S1][S2].",
        "evolution": "Dense attention came first [S1], then retrieval grounding [S2].",
        "approaches": "Two families appear: sparsity [S1] and retrieval [S2].",
    },
    "evidence": {
        "datasets": "arXiv and PubMed are shared [S1]; MedQA appears once [S2].",
        "comparison": "SparseSum reaches 44.1 ROUGE-L [S1]; ClinRAG reaches 78.4 accuracy [S2].",
        "conflicts": "The two do not conflict: they evaluate on different tasks [S1][S2].",
    },
    "forward": {
        "gaps": "Neither evaluates multilingual corpora [S1][S2].",
        "open_problems": "Faithfulness remains unmeasured at scale [S2].",
        "directions": "Replicate on a multilingual scientific corpus with human raters.",
    },
}


def review_llm(overrides: dict | None = None) -> FakeGemini:
    """A fake that answers each of the three group prompts in turn."""
    responses = {}
    for group in GROUPS:
        marker = group["sections"][0][0]          # a word unique to that prompt
        responses[f"`{marker}`"] = (overrides or {}).get(group["key"], SECTIONS[group["key"]])
    client = FakeGemini(responses)

    # The groups are told apart by the prompt, not the system instruction, so
    # the matcher has to look there.
    def generate_json(prompt, system_instruction=None, schema=None, **kwargs):
        client.calls.append({"prompt": prompt, "system": system_instruction, "schema": schema})
        for key, payload in responses.items():
            if key in prompt:
                return payload
        return {}

    client.generate_json = generate_json  # type: ignore[method-assign]
    set_llm(client)
    return client


# --- citations ----------------------------------------------------------------


def test_a_citation_to_a_paper_that_was_never_supplied_is_dropped():
    citations = [{"marker": "S1"}, {"marker": "S2"}]
    text, used = resolve_citations("True [S1]. Invented [S9]. Also true [S2].", citations)

    assert "[S9]" not in text
    assert "[S1]" in text and "[S2]" in text
    assert used == ["S1", "S2"]


def test_dropping_a_marker_does_not_leave_a_gap_before_the_full_stop():
    text, _ = resolve_citations("A claim [S7].", [{"marker": "S1"}])
    assert text == "A claim."


def test_the_same_marker_is_only_counted_once():
    _, used = resolve_citations("[S1] and again [S1] and [S2]", [{"marker": "S1"}, {"marker": "S2"}])
    assert used == ["S1", "S2"]


def test_text_without_citations_survives_untouched():
    text, used = resolve_citations("No citations here at all.", [{"marker": "S1"}])
    assert text == "No citations here at all."
    assert used == []


# --- context ------------------------------------------------------------------


def test_each_paper_gets_its_own_marker_and_a_key_back_to_it():
    documents = [
        {"id": "a", "title": "First", "authors": ["Ada"], "year": 2020, "text": "x" * 100},
        {"id": "b", "title": "Second", "authors": [], "year": None, "text": "y" * 100},
    ]
    context, citations = numbered_context(documents)

    assert "[S1] First" in context and "[S2] Second" in context
    assert [c["marker"] for c in citations] == ["S1", "S2"]
    assert citations[0]["paper_id"] == "a"


def test_references_are_built_from_metadata_without_a_model():
    line = format_reference(
        {"marker": "S1", "authors": ["Ada", "Alan", "Grace", "Barbara"],
         "year": 2021, "title": "A Paper", "venue": "NeurIPS", "doi": "10.1/x"}
    )
    assert "[S1]" in line
    assert "Ada et al." in line, "four authors should be abbreviated"
    assert "2021" in line and "A Paper" in line and "doi:10.1/x" in line


# --- generation ---------------------------------------------------------------


def test_it_writes_every_section_in_three_calls(sample_pdf, second_pdf):
    ingest(sample_pdf)
    ingest(second_pdf)
    client = review_llm()

    documents = [
        {"id": "a", "title": "SparseSum", "authors": ["Ada"], "year": 2023, "text": "x" * 500},
        {"id": "b", "title": "ClinRAG", "authors": ["Barbara"], "year": 2024, "text": "y" * 500},
    ]
    result = generate("speech models", documents)

    assert [s["key"] for s in result.sections] == SECTION_ORDER
    assert result.llm_calls == 3, "ten sections, three passes"
    assert len(client.calls) == 3
    assert result.status == "completed"


def test_the_markdown_carries_all_ten_sections(sample_pdf):
    review_llm()
    documents = [{"id": "a", "title": "SparseSum", "authors": ["Ada"], "year": 2023, "text": "x" * 500},
                 {"id": "b", "title": "ClinRAG", "authors": ["Bob"], "year": 2024, "text": "y" * 500}]
    result = generate("speech models", documents)

    for heading in ("1. Introduction", "5. Method Comparison", "9. Proposed Research Directions",
                    "10. References"):
        assert heading in result.markdown, heading
    assert result.markdown.startswith("# Literature Review: speech models")


def test_a_failed_group_does_not_lose_the_others(sample_pdf):
    """Two thirds of a review is worth returning; a note says what is missing."""
    client = review_llm()

    def half_broken(prompt, system_instruction=None, schema=None, **kwargs):
        client.calls.append({"prompt": prompt})
        if "`datasets`" in prompt:
            from app.services.llm import TransientLLMError

            raise TransientLLMError("503 UNAVAILABLE: overloaded")
        for key, payload in (("`introduction`", SECTIONS["framing"]), ("`gaps`", SECTIONS["forward"])):
            if key in prompt:
                return payload
        return {}

    client.generate_json = half_broken  # type: ignore[method-assign]

    documents = [{"id": "a", "title": "SparseSum", "authors": ["Ada"], "year": 2023, "text": "x" * 500}]
    result = generate("speech models", documents)

    keys = [s["key"] for s in result.sections]
    assert "introduction" in keys and "gaps" in keys
    assert "datasets" not in keys
    assert result.status == "partial"
    assert any("evidence" in error for error in result.errors)


def test_an_empty_corpus_is_reported_not_raised():
    review_llm()
    result = generate("anything", [])

    assert result.sections == []
    assert result.status == "failed"
    assert any("no indexed papers" in error for error in result.errors)


def test_invented_citations_are_stripped_from_the_finished_review():
    review_llm({"framing": {
        "introduction": "Real [S1] and invented [S8].",
        "evolution": "Fine [S1].",
        "approaches": "Fine [S1].",
    }})
    documents = [{"id": "a", "title": "Only Paper", "authors": [], "year": 2020, "text": "x" * 500}]
    result = generate("topic", documents)

    assert "[S8]" not in result.markdown
    assert "[S1]" in result.markdown


# --- HTTP ---------------------------------------------------------------------


def test_review_endpoint_writes_and_saves(client, sample_pdf, second_pdf):
    ingest(sample_pdf)
    ingest(second_pdf)
    review_llm()

    response = client.post("/api/review", json={"topic": "speech models"})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["id"]
    assert len(body["sections"]) == 9, "nine written sections; references are the tenth"
    assert body["llm_calls"] == 3
    assert body["citations"]
    assert body["status"] == "completed"

    listing = client.get("/api/review").json()
    assert [row["id"] for row in listing] == [body["id"]]
    assert listing[0]["topic"] == "speech models"


def test_a_saved_review_can_be_reopened_and_exported(client, sample_pdf):
    ingest(sample_pdf)
    review_llm()

    created = client.post("/api/review", json={"topic": "speech models"}).json()

    again = client.get(f"/api/review/{created['id']}").json()
    assert again["topic"] == "speech models"
    assert len(again["sections"]) == len(created["sections"])

    markdown = client.get(f"/api/review/{created['id']}/markdown")
    assert markdown.status_code == 200
    assert "10. References" in markdown.text

    pdf = client.get(f"/api/review/{created['id']}/pdf")
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"


def test_a_review_can_be_deleted(client, sample_pdf):
    ingest(sample_pdf)
    review_llm()

    created = client.post("/api/review", json={"topic": "speech models"}).json()
    assert client.delete(f"/api/review/{created['id']}").status_code == 204
    assert client.get(f"/api/review/{created['id']}").status_code == 404


def test_a_review_needs_at_least_one_paper(client):
    review_llm()
    assert client.post("/api/review", json={"topic": "speech models"}).status_code == 409


def test_a_short_topic_is_refused(client, sample_pdf):
    ingest(sample_pdf)
    review_llm()
    assert client.post("/api/review", json={"topic": "ai"}).status_code == 422


def test_another_account_cannot_read_your_review(client, anon_client, sample_pdf):
    ingest(sample_pdf)
    review_llm()
    created = client.post("/api/review", json={"topic": "speech models"}).json()

    from tests.test_auth import register_and_activate

    other = register_and_activate(anon_client, username="ada")
    assert anon_client.get(f"/api/review/{created['id']}", headers=other).status_code == 404
    assert anon_client.get("/api/review", headers=other).json() == []


def test_an_anonymous_caller_is_refused(anon_client):
    assert anon_client.post("/api/review", json={"topic": "speech models"}).status_code == 401
    assert anon_client.get("/api/review").status_code == 401
