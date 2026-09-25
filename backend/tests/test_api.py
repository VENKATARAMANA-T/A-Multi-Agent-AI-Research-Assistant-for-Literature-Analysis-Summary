"""End-to-end HTTP tests over the FastAPI application."""

from __future__ import annotations

import io


def upload(client, path, filename=None):
    """Synchronous upload — `wait=true` returns the definitive per-file result."""
    with open(path, "rb") as handle:
        return client.post(
            "/api/papers/upload?wait=true",
            files=[("files", (filename or path.name, handle.read(), "application/pdf"))],
        )


# --- system ------------------------------------------------------------------


def test_health_reports_backends(client):
    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["llm"]["configured"] is False
    assert payload["embeddings"]["degraded"] is True  # offline fallback in tests
    assert payload["graph"]["backend"] == "in-memory"  # Neo4j disabled in tests


def test_root_and_docs(client):
    assert client.get("/").json()["name"] == "ResearchCompass API"
    assert client.get("/docs").status_code == 200


# --- upload ------------------------------------------------------------------


def test_upload_indexes_a_paper(client, sample_pdf):
    payload = upload(client, sample_pdf).json()

    assert payload["indexed"] == 1
    item = payload["results"][0]
    assert item["status"] == "indexed"
    assert "Sparse Attention" in item["paper"]["title"]
    assert item["paper"]["chunk_count"] > 0
    assert item["paper"]["status"] == "indexed"


def test_duplicate_upload_is_detected(client, sample_pdf):
    upload(client, sample_pdf)
    payload = upload(client, sample_pdf).json()

    assert payload["duplicates"] == 1
    assert payload["results"][0]["status"] == "duplicate"


def test_non_pdf_upload_is_rejected(client):
    response = client.post(
        "/api/papers/upload?wait=true",
        files=[("files", ("notes.pdf", io.BytesIO(b"this is not a pdf"), "application/pdf"))],
    )
    payload = response.json()

    assert payload["failed"] == 1
    assert "not a valid PDF" in payload["results"][0]["detail"]


def test_empty_upload_is_rejected(client):
    assert client.post("/api/papers/upload", files=[]).status_code == 422


# --- library -----------------------------------------------------------------


def test_list_get_and_filter_papers(client, sample_pdf, second_pdf):
    upload(client, sample_pdf)
    upload(client, second_pdf)

    papers = client.get("/api/papers").json()
    assert len(papers) == 2

    filtered = client.get("/api/papers", params={"q": "clinical"}).json()
    assert len(filtered) == 1
    assert "Clinical" in filtered[0]["title"]

    single = client.get(f"/api/papers/{papers[0]['id']}").json()
    assert single["id"] == papers[0]["id"]

    assert client.get("/api/papers/does-not-exist").status_code == 404


def test_chunks_endpoint_paginates(client, sample_pdf):
    paper_id = upload(client, sample_pdf).json()["results"][0]["paper_id"]
    payload = client.get(f"/api/papers/{paper_id}/chunks", params={"limit": 3}).json()

    assert len(payload["chunks"]) <= 3
    assert payload["total"] >= len(payload["chunks"])
    assert payload["chunks"][0]["index"] == 0


def test_download_original_pdf(client, sample_pdf):
    paper_id = upload(client, sample_pdf).json()["results"][0]["paper_id"]
    response = client.get(f"/api/papers/{paper_id}/file")

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")


def test_delete_removes_paper_and_vectors(client, sample_pdf):
    from app.services.vector_store import get_vector_store

    paper_id = upload(client, sample_pdf).json()["results"][0]["paper_id"]
    assert get_vector_store().count() > 0

    assert client.delete(f"/api/papers/{paper_id}").status_code == 204
    assert client.get(f"/api/papers/{paper_id}").status_code == 404
    assert get_vector_store().count() == 0
    assert client.delete(f"/api/papers/{paper_id}").status_code == 404


def test_reindex_paper(client, sample_pdf):
    paper_id = upload(client, sample_pdf).json()["results"][0]["paper_id"]
    payload = client.post(f"/api/papers/{paper_id}/reindex").json()

    assert payload["status"] == "indexed"
    assert payload["chunk_count"] > 0


# --- search ------------------------------------------------------------------


def test_semantic_search(client, sample_pdf):
    upload(client, sample_pdf)
    payload = client.post(
        "/api/papers/search", json={"query": "sparse attention routing", "top_k": 4}
    ).json()

    assert payload["query"] == "sparse attention routing"
    assert 0 < len(payload["hits"]) <= 4
    assert payload["hits"][0]["score"] >= payload["hits"][-1]["score"]


def test_search_validates_input(client):
    assert client.post("/api/papers/search", json={"query": ""}).status_code == 422
    assert client.post("/api/papers/search", json={"query": "x", "top_k": 99}).status_code == 422


# --- agents ------------------------------------------------------------------


def test_ask_endpoint(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    payload = client.post(
        "/api/agents/ask", json={"question": "What score does SparseSum reach?"}
    ).json()

    assert payload["intent"] == "qa"
    assert payload["answer"]["confidence"] == "high"
    assert payload["answer"]["sources"]
    assert payload["trace"]


def test_ask_requires_indexed_papers(client, fake_llm):
    response = client.post("/api/agents/ask", json={"question": "Anything here?"})
    assert response.status_code == 409


def test_unknown_paper_id_returns_404(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    response = client.post(
        "/api/agents/ask", json={"question": "What is this?", "paper_ids": ["nope"]}
    )
    assert response.status_code == 404


def test_summarize_single_requires_exactly_one_paper(client, sample_pdf, second_pdf, fake_llm):
    upload(client, sample_pdf)
    upload(client, second_pdf)

    response = client.post("/api/agents/summarize", json={"scope": "single"})
    assert response.status_code == 400


def test_summarize_multi_and_persists_record(client, sample_pdf, second_pdf, fake_llm):
    from sqlmodel import select

    from app.database import session_scope
    from app.models import SummaryRecord

    upload(client, sample_pdf)
    upload(client, second_pdf)

    payload = client.post("/api/agents/summarize", json={"scope": "multi"}).json()
    assert payload["summary"]["scope"] == "multi"

    with session_scope() as session:
        assert len(list(session.exec(select(SummaryRecord)).all())) == 1


def test_extract_endpoint_persists_records(client, sample_pdf, fake_llm):
    from sqlmodel import select

    from app.database import session_scope
    from app.models import ExtractionRecord

    upload(client, sample_pdf)
    payload = client.post("/api/agents/extract", json={}).json()

    assert payload["extraction"]["papers"]
    with session_scope() as session:
        assert len(list(session.exec(select(ExtractionRecord)).all())) == 1


def test_gaps_endpoint(client, sample_pdf, second_pdf, fake_llm):
    upload(client, sample_pdf)
    upload(client, second_pdf)

    payload = client.post("/api/agents/gaps", json={"focus": "evaluation"}).json()
    assert len(payload["gaps"]["gaps"]) == 2
    assert payload["gaps"]["gaps"][0]["severity"] == "high"


def test_agent_runs_audit_trail(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    client.post("/api/agents/ask", json={"question": "What is the method?"})

    runs = client.get("/api/agents/runs").json()
    assert len(runs) == 1
    assert runs[0]["intent"] == "qa"

    detail = client.get(f"/api/agents/runs/{runs[0]['id']}").json()
    assert detail["result"]["answer"]
    assert client.get("/api/agents/runs/missing").status_code == 404


# --- knowledge graph ---------------------------------------------------------


def test_build_and_fetch_graph(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    client.post("/api/agents/graph/build", json={})

    graph = client.get("/api/graph").json()
    assert graph["backend"] == "in-memory"
    assert graph["stats"]["node_count"] > 0
    assert {node["type"] for node in graph["nodes"]} & {"Paper", "Method", "Dataset"}

    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"]
    )


def test_graph_filters_by_node_type(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    client.post("/api/agents/graph/build", json={})

    filtered = client.get("/api/graph", params={"node_types": ["Paper"]}).json()
    assert {node["type"] for node in filtered["nodes"]} == {"Paper"}


def test_graph_neighbors_and_clear(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    client.post("/api/agents/graph/build", json={})

    graph = client.get("/api/graph").json()
    paper_node = next(node for node in graph["nodes"] if node["type"] == "Paper")

    neighborhood = client.get(f"/api/graph/neighbors/{paper_node['id']}").json()
    assert neighborhood["stats"]["node_count"] > 1

    assert client.get("/api/graph/neighbors/entity:nope").status_code == 404

    assert client.delete("/api/graph").status_code == 204
    assert client.get("/api/graph").json()["stats"]["node_count"] == 0


def test_deleting_a_paper_prunes_the_graph(client, sample_pdf, fake_llm):
    paper_id = upload(client, sample_pdf).json()["results"][0]["paper_id"]
    client.post("/api/agents/graph/build", json={})
    assert client.get("/api/graph").json()["stats"]["node_count"] > 0

    client.delete(f"/api/papers/{paper_id}")
    assert client.get("/api/graph").json()["stats"]["node_count"] == 0


# --- reports -----------------------------------------------------------------


def test_report_generation_produces_markdown_and_pdf(client, sample_pdf, second_pdf, fake_llm):
    upload(client, sample_pdf)
    upload(client, second_pdf)

    report = client.post("/api/reports", json={"title": "Efficiency Review"}).json()

    assert report["title"] == "Efficiency Review"
    assert len(report["paper_ids"]) == 2
    assert report["has_pdf"] is True

    markdown = report["markdown"]
    assert "# Efficiency Review" in markdown
    assert "## 1. Corpus Overview" in markdown
    assert "Research Gaps and Future Directions" in markdown
    assert "No multilingual evaluation" in markdown
    assert "References" in markdown

    pdf = client.get(f"/api/reports/{report['id']}/pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")

    raw = client.get(f"/api/reports/{report['id']}/markdown")
    assert raw.status_code == 200
    assert raw.text == markdown


def test_report_lifecycle(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    report_id = client.post("/api/reports", json={}).json()["id"]

    assert len(client.get("/api/reports").json()) == 1
    assert client.get(f"/api/reports/{report_id}").json()["id"] == report_id

    assert client.delete(f"/api/reports/{report_id}").status_code == 204
    assert client.get(f"/api/reports/{report_id}").status_code == 404


def test_report_requires_papers(client, fake_llm):
    assert client.post("/api/reports", json={}).status_code == 409


# --- stats -------------------------------------------------------------------


def test_stats_endpoint(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    client.post("/api/agents/ask", json={"question": "What is the method?"})

    payload = client.get("/api/stats").json()
    assert payload["papers"] == 1
    assert payload["indexed"] == 1
    assert payload["chunks"] > 0
    assert payload["vectors"] == payload["chunks"]
    assert payload["agent_runs"] == 1
