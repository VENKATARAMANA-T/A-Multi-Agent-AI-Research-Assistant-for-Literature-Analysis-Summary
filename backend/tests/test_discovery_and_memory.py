"""Related-paper discovery and multi-turn conversation memory.

Discovery talks to OpenAlex, so every network call here is stubbed — the suite
must stay offline and deterministic.
"""

from __future__ import annotations

import pytest

from app.database import session_scope
from app.services import discovery
from app.services.discovery import Candidate, deduplicate
from app.services.ingestion import process_paper, store_upload


def ingest(path) -> str:
    from tests.conftest import seed_user_id

    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        return paper.id

WORK = {
    "id": "https://openalex.org/W123",
    "display_name": "Sparse Attention for Long Documents",
    "publication_year": 2023,
    "cited_by_count": 42,
    "doi": "https://doi.org/10.1000/abc",
    "authorships": [
        {"author": {"display_name": "Ada Lovelace"}},
        {"author": {"display_name": "Alan Turing"}},
    ],
    "primary_location": {"source": {"display_name": "NeurIPS"}},
    "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
    "abstract_inverted_index": {"We": [0], "introduce": [1], "SparseSum": [2]},
    "related_works": ["https://openalex.org/W456"],
    "referenced_works": ["https://openalex.org/W789"],
}


# --- parsing -----------------------------------------------------------------


def test_inverted_abstract_is_rebuilt_in_order():
    text = discovery._invert_abstract({"quick": [1], "The": [0], "fox": [2]})
    assert text == "The quick fox"


def test_missing_abstract_is_none():
    assert discovery._invert_abstract(None) is None
    assert discovery._invert_abstract({}) is None


def test_work_is_converted_to_a_candidate():
    candidate = discovery._to_candidate(WORK, "related")

    assert candidate.title == "Sparse Attention for Long Documents"
    assert candidate.authors == ["Ada Lovelace", "Alan Turing"]
    assert candidate.year == 2023
    assert candidate.venue == "NeurIPS"
    assert candidate.doi == "10.1000/abc"  # the URL prefix is stripped
    assert candidate.open_access_url == "https://example.org/paper.pdf"
    assert candidate.citations == 42
    assert candidate.external_id == "W123"


def test_a_work_without_a_title_is_skipped():
    assert discovery._to_candidate({"id": "x"}, "related") is None


# --- de-duplication ----------------------------------------------------------


def test_papers_already_in_the_corpus_are_removed():
    candidates = [
        Candidate(external_id="a", title="Sparse Attention", doi="10.1/x"),
        Candidate(external_id="b", title="Something Else", doi="10.2/y"),
    ]
    remaining = deduplicate(candidates, known_dois={"10.1/X"}, known_titles=set())

    assert [c.external_id for c in remaining] == ["b"]


def test_title_matching_ignores_punctuation_and_case():
    candidates = [Candidate(external_id="a", title="Sparse Attention: For Long Documents!")]
    remaining = deduplicate(candidates, set(), {"sparse attention for long documents"})
    assert remaining == []


def test_duplicate_candidates_are_collapsed():
    candidates = [
        Candidate(external_id="a", title="Same Paper"),
        Candidate(external_id="b", title="Same  Paper"),
    ]
    assert len(deduplicate(candidates, set(), set())) == 1


# --- HTTP --------------------------------------------------------------------


def upload(client, path):
    return client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (path.name, path.read_bytes(), "application/pdf"))],
    ).json()["results"][0]["paper_id"]


def test_related_returns_candidates(client, sample_pdf, monkeypatch):
    paper_id = upload(client, sample_pdf)
    monkeypatch.setattr(
        discovery,
        "related_to",
        lambda title=None, doi=None, limit=15: [
            Candidate(external_id="W9", title="A Neighbouring Paper", year=2022, citations=7)
        ],
    )

    payload = client.get("/api/discover/related", params={"paper_id": paper_id}).json()

    assert payload["source"] == "openalex"
    assert payload["candidates"][0]["title"] == "A Neighbouring Paper"


def test_related_excludes_what_you_already_have(client, sample_pdf, monkeypatch):
    """Suggesting a paper already in the corpus is noise."""
    paper_id = upload(client, sample_pdf)
    own_title = client.get(f"/api/papers/{paper_id}").json()["title"]

    monkeypatch.setattr(
        discovery,
        "related_to",
        lambda title=None, doi=None, limit=15: [
            Candidate(external_id="W1", title=own_title),
            Candidate(external_id="W2", title="Genuinely New Work"),
        ],
    )

    payload = client.get("/api/discover/related", params={"paper_id": paper_id}).json()
    titles = [c["title"] for c in payload["candidates"]]

    assert own_title not in titles
    assert "Genuinely New Work" in titles


def test_search_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        discovery,
        "search",
        lambda query, limit=15, year_from=None: [Candidate(external_id="W5", title="Found It")],
    )
    payload = client.get("/api/discover/search", params={"q": "dysarthria detection"}).json()
    assert payload["candidates"][0]["title"] == "Found It"


def test_search_validates_the_query(client):
    assert client.get("/api/discover/search", params={"q": "ab"}).status_code == 422


def test_unreachable_service_is_a_502_not_a_crash(client, sample_pdf, monkeypatch):
    paper_id = upload(client, sample_pdf)

    def boom(**kwargs):
        raise discovery.DiscoveryUnavailable("network down")

    monkeypatch.setattr(discovery, "related_to", boom)
    response = client.get("/api/discover/related", params={"paper_id": paper_id})

    assert response.status_code == 502
    assert "network down" in response.json()["detail"]


def test_discovery_can_be_disabled(client, sample_pdf, monkeypatch):
    from app.config import settings

    paper_id = upload(client, sample_pdf)
    monkeypatch.setattr(settings, "discovery_enabled", False)

    assert client.get("/api/discover/related", params={"paper_id": paper_id}).status_code == 503


def test_corpus_gaps_rank_by_how_often_work_recurs(client, sample_pdf, second_pdf, monkeypatch):
    upload(client, sample_pdf)
    upload(client, second_pdf)

    def fake_related(title=None, doi=None, limit=15):
        return [
            Candidate(external_id="W_common", title="Cited By Both", citations=5),
            Candidate(external_id=f"W_{(title or '')[:4]}", title=f"Only For {title}", citations=99),
        ]

    monkeypatch.setattr(discovery, "related_to", fake_related)
    payload = client.get("/api/discover/gaps").json()

    top = payload["candidates"][0]
    assert top["title"] == "Cited By Both"
    assert top["referenced_by_corpus"] == 2


# --- conversation memory -----------------------------------------------------


def test_a_conversation_records_both_turns(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)

    first = client.post(
        "/api/agents/ask",
        json={"question": "What datasets are used?", "start_conversation": True},
    ).json()

    conversation_id = first["conversation_id"]
    assert conversation_id

    detail = client.get(f"/api/agents/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"


def test_a_follow_up_receives_the_earlier_turns(client, sample_pdf, fake_llm):
    """Without history, a follow-up like "why?" has nothing to resolve against."""
    upload(client, sample_pdf)

    first = client.post(
        "/api/agents/ask",
        json={"question": "What datasets are used?", "start_conversation": True},
    ).json()

    client.post(
        "/api/agents/ask",
        json={"question": "Why those ones?", "conversation_id": first["conversation_id"]},
    )

    # The second call's prompt must contain the first exchange.
    last_prompt = fake_llm.calls[-1]["prompt"]
    assert "Earlier in this conversation" in last_prompt
    assert "What datasets are used?" in last_prompt


def test_history_is_bounded(client, sample_pdf, fake_llm, monkeypatch):
    """Every replayed turn costs prompt tokens, so the window is capped."""
    from app.config import settings

    monkeypatch.setattr(settings, "conversation_memory_turns", 1)
    upload(client, sample_pdf)

    conversation_id = client.post(
        "/api/agents/ask", json={"question": "Question one?", "start_conversation": True}
    ).json()["conversation_id"]

    for text in ("Question two?", "Question three?"):
        client.post("/api/agents/ask", json={"question": text, "conversation_id": conversation_id})

    last_prompt = fake_llm.calls[-1]["prompt"]
    assert "Question one?" not in last_prompt
    assert "Question two?" in last_prompt


def test_asking_without_a_conversation_creates_none(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    payload = client.post("/api/agents/ask", json={"question": "A one-off question?"}).json()

    assert payload.get("conversation_id") is None
    assert client.get("/api/agents/conversations").json() == []


def test_conversation_lifecycle(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    conversation_id = client.post(
        "/api/agents/ask", json={"question": "First?", "start_conversation": True}
    ).json()["conversation_id"]

    listing = client.get("/api/agents/conversations").json()
    assert listing[0]["id"] == conversation_id
    assert listing[0]["turns"] == 1

    assert client.delete(f"/api/agents/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/agents/conversations/{conversation_id}").status_code == 404


def test_unknown_conversation_is_rejected(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    response = client.post(
        "/api/agents/ask", json={"question": "Anything?", "conversation_id": "missing"}
    )
    assert response.status_code == 404


# --- scoping the gap search --------------------------------------------------


def test_gap_search_can_be_scoped_to_chosen_papers(client, sample_pdf, second_pdf, monkeypatch):
    """"Missing from your corpus" has to say which corpus it meant."""
    from app.services import discovery

    ingest(sample_pdf)
    ingest(second_pdf)

    asked: list[str] = []

    def fake_related(title, doi=None, limit=15):
        asked.append(title)
        return [
            discovery.Candidate(
                external_id="W1", title="A missing paper", relation="related",
                authors=["X"], year=2021, venue="V", citations=40,
            )
        ]

    monkeypatch.setattr(discovery, "related_to", fake_related)

    papers = client.get("/api/papers").json()
    one = papers[0]["id"]

    response = client.get("/api/discover/gaps", params={"paper_ids": [one]})
    assert response.status_code == 200

    body = response.json()
    assert len(asked) == 1, "only the chosen paper should be looked up"
    assert len(body["searched_papers"]) == 1
    assert body["candidates"]


def test_gap_search_without_paper_ids_uses_the_whole_corpus(client, sample_pdf, second_pdf, monkeypatch):
    from app.services import discovery

    ingest(sample_pdf)
    ingest(second_pdf)

    asked: list[str] = []
    monkeypatch.setattr(
        discovery,
        "related_to",
        lambda title, doi=None, limit=15: (asked.append(title) or []),
    )

    response = client.get("/api/discover/gaps")
    assert response.status_code == 200
    assert len(asked) == 2
    assert len(response.json()["searched_papers"]) == 2


def test_gap_search_rejects_an_unknown_paper(client, sample_pdf):
    ingest(sample_pdf)
    response = client.get("/api/discover/gaps", params={"paper_ids": ["nope"]})
    assert response.status_code == 404
