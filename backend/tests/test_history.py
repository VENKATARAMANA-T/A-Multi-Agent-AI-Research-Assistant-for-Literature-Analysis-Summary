"""Saved results.

Reopening a past run has to give back what the page originally showed, not a
summary of it. The test that matters most is that the evidence survives: an
answer whose citations are intact but whose passages are gone is less checkable
saved than it was live.
"""

from __future__ import annotations

from app.database import session_scope
from app.services.ingestion import process_paper, store_upload
from tests.conftest import seed_user_id


def ingest(path) -> str:
    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        return paper.id


# --- what gets listed ---------------------------------------------------------


def test_a_run_appears_in_history(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    client.post("/api/agents/ask", json={"question": "What datasets are used?"})

    items = client.get("/api/history", params={"intent": ["qa"]}).json()

    assert len(items) == 1
    assert items[0]["intent"] == "qa"
    assert items[0]["label"] == "What datasets are used?"
    assert items[0]["paper_count"] == 1
    assert items[0]["status"] == "completed"


def test_history_is_filtered_by_page(client, sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)

    client.post("/api/agents/ask", json={"question": "What is the method?"})
    client.post("/api/agents/extract", json={})
    client.post("/api/agents/gaps", json={})

    assert len(client.get("/api/history", params={"intent": ["qa"]}).json()) == 1
    assert len(client.get("/api/history", params={"intent": ["extract"]}).json()) == 1
    assert len(client.get("/api/history", params={"intent": ["gap"]}).json()) == 1
    # Summaries writes two different intents, so a page may ask for several.
    assert client.get("/api/history", params={"intent": ["summarize", "multi_summarize"]}).json() == []


def test_the_newest_run_is_first(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    client.post("/api/agents/ask", json={"question": "First question?"})
    client.post("/api/agents/ask", json={"question": "Second question?"})

    labels = [item["label"] for item in client.get("/api/history", params={"intent": ["qa"]}).json()]
    assert labels == ["Second question?", "First question?"]


def test_runs_with_nothing_to_reopen_are_left_out(client, sample_pdf, fake_llm):
    """A graph build writes to the graph database; there is no view to restore."""
    ingest(sample_pdf)
    client.post("/api/agents/graph/build", json={})

    assert client.get("/api/history").json() == []


def test_a_summary_is_labelled_by_its_paper(client, sample_pdf, fake_llm):
    paper_id = ingest(sample_pdf)
    client.post("/api/agents/summarize", json={"paper_ids": [paper_id], "scope": "single"})

    item = client.get("/api/history", params={"intent": ["summarize"]}).json()[0]
    # The fixture's title line is clipped at the page edge by PyMuPDF, which
    # does not wrap — so this matches the prefix rather than the whole title.
    assert item["label"].startswith("Sparse Attention Transformers for Long Document")


def test_a_run_with_no_question_falls_back_to_a_paper_count(client, sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    client.post("/api/agents/extract", json={})

    assert client.get("/api/history", params={"intent": ["extract"]}).json()[0]["label"] == "2 papers"


# --- reopening ----------------------------------------------------------------


def test_reopening_returns_the_answer_and_its_evidence(client, sample_pdf, fake_llm):
    """Citations without the passages behind them are not worth saving."""
    ingest(sample_pdf)
    live = client.post("/api/agents/ask", json={"question": "What datasets are used?"}).json()

    run_id = client.get("/api/history", params={"intent": ["qa"]}).json()[0]["id"]
    saved = client.get(f"/api/history/{run_id}").json()

    assert saved["answer"]["answer"] == live["answer"]["answer"]
    assert saved["answer"]["sources"] == live["answer"]["sources"]
    assert saved["retrieved"], "the retrieved passages must survive"
    assert len(saved["retrieved"]) == len(live["retrieved"])
    assert saved["question"] == "What datasets are used?"


def test_reopening_a_summary_gives_back_every_section(client, sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    live = client.post("/api/agents/summarize", json={"scope": "multi"}).json()

    run_id = client.get("/api/history", params={"intent": ["multi_summarize"]}).json()[0]["id"]
    saved = client.get(f"/api/history/{run_id}").json()

    assert saved["summary"] == live["summary"]
    assert saved["summary"]["themes"]


def test_reopening_an_extraction_keeps_the_aggregate(client, sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    live = client.post("/api/agents/extract", json={}).json()

    run_id = client.get("/api/history", params={"intent": ["extract"]}).json()[0]["id"]
    saved = client.get(f"/api/history/{run_id}").json()

    assert saved["extraction"]["aggregate"] == live["extraction"]["aggregate"]
    assert len(saved["extraction"]["papers"]) == 2


def test_reopening_gaps_keeps_their_order(client, sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    live = client.post("/api/agents/gaps", json={}).json()

    run_id = client.get("/api/history", params={"intent": ["gap"]}).json()[0]["id"]
    saved = client.get(f"/api/history/{run_id}").json()

    assert [g["severity"] for g in saved["gaps"]["gaps"]] == [
        g["severity"] for g in live["gaps"]["gaps"]
    ]


def test_a_failed_run_is_kept_and_says_so(client, sample_pdf):
    """No API key, so the agent fails — the attempt is still worth recording."""
    ingest(sample_pdf)
    client.post("/api/agents/gaps", json={})

    items = client.get("/api/history", params={"intent": ["gap"]}).json()
    assert items[0]["status"] == "partial"
    assert items[0]["has_error"] is True

    saved = client.get(f"/api/history/{items[0]['id']}").json()
    assert saved["errors"]


def test_an_unknown_run_is_404(client):
    assert client.get("/api/history/nope").status_code == 404


# --- removing -----------------------------------------------------------------


def test_one_entry_can_be_forgotten(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    client.post("/api/agents/ask", json={"question": "First?"})
    client.post("/api/agents/ask", json={"question": "Second?"})

    items = client.get("/api/history", params={"intent": ["qa"]}).json()
    assert client.delete(f"/api/history/{items[0]['id']}").status_code == 204

    remaining = client.get("/api/history", params={"intent": ["qa"]}).json()
    assert [i["label"] for i in remaining] == ["First?"]


def test_clearing_a_page_leaves_the_others(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    client.post("/api/agents/ask", json={"question": "A question?"})
    client.post("/api/agents/extract", json={})

    assert client.request("DELETE", "/api/history", params={"intent": ["qa"]}).status_code == 204

    assert client.get("/api/history", params={"intent": ["qa"]}).json() == []
    assert len(client.get("/api/history", params={"intent": ["extract"]}).json()) == 1


# --- isolation ----------------------------------------------------------------


def test_history_is_private_to_the_account(client, anon_client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    client.post("/api/agents/ask", json={"question": "My question?"})
    run_id = client.get("/api/history", params={"intent": ["qa"]}).json()[0]["id"]

    from tests.test_auth import register_and_activate

    other = register_and_activate(anon_client, username="ada")

    assert anon_client.get("/api/history", headers=other).json() == []
    assert anon_client.get(f"/api/history/{run_id}", headers=other).status_code == 404
    assert anon_client.delete(f"/api/history/{run_id}", headers=other).status_code == 404


def test_an_anonymous_caller_is_refused(anon_client):
    assert anon_client.get("/api/history").status_code == 401
