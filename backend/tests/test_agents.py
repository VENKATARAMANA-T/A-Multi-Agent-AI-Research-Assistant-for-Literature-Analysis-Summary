"""Step 6: the multi-agent LangGraph workflow, driven by a mocked Gemini client."""

from __future__ import annotations

from app.agents.extraction import aggregate_extractions, normalise_extraction
from app.agents.graph_builder import canonical_key
from app.agents.retrieval import expand_query
from app.agents.workflow import build_workflow, run_workflow
from app.database import session_scope
from app.services.ingestion import process_paper, store_upload
from app.services.llm import parse_json_object


def ingest(path) -> str:
    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes())
        process_paper(session, paper.id)
        return paper.id


# --- graph wiring ------------------------------------------------------------


def test_workflow_compiles_with_every_node():
    workflow = build_workflow()
    nodes = set(workflow.get_graph().nodes)

    for expected in ("router", "retrieve", "qa", "load", "summarize", "extract", "gap", "build_graph"):
        assert expected in nodes


def test_qa_intent_routes_through_retrieval(sample_pdf, fake_llm):
    ingest(sample_pdf)
    result = run_workflow("qa", question="What ROUGE-L score does SparseSum reach?")

    nodes = [event["node"] for event in result["trace"]]
    # Vector retrieval, then graph retrieval, then the answer.
    assert nodes[:4] == ["router", "retrieval", "graph_retrieval", "qa"]
    assert "load" not in nodes

    assert result["status"] == "completed"
    assert result["answer"]["confidence"] == "high"
    assert result["retrieved"], "retrieval should return grounding chunks"


def test_qa_resolves_citation_markers_to_real_chunks(sample_pdf, fake_llm):
    ingest(sample_pdf)
    result = run_workflow("qa", question="What datasets are used?")

    sources = result["answer"]["sources"]
    assert len(sources) == 1
    assert sources[0]["marker"] == "S1"
    assert sources[0]["chunk_id"] == result["retrieved"][0]["chunk_id"]
    assert sources[0]["excerpt"]


def test_single_paper_summary(sample_pdf, fake_llm):
    paper_id = ingest(sample_pdf)
    result = run_workflow("summarize", paper_ids=[paper_id])

    summary = result["summary"]
    assert summary["scope"] == "single"
    assert summary["paper_id"] == paper_id
    assert "linear-time" in summary["tldr"]
    assert summary["key_findings"]


def test_multi_paper_synthesis(sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    result = run_workflow("multi_summarize")

    summary = result["summary"]
    assert summary["scope"] == "multi"
    assert len(summary["paper_ids"]) == 2
    assert summary["themes"]


def test_extraction_runs_once_per_paper_and_aggregates(sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    result = run_workflow("extract")

    extraction = result["extraction"]
    assert len(extraction["papers"]) == 2
    assert result["llm_calls"] == 2

    datasets = {entry["name"] for entry in extraction["aggregate"]["datasets"]}
    assert {"arXiv", "PubMed"} <= datasets
    # arXiv appears in both papers' mocked responses, so it must aggregate.
    arxiv = next(e for e in extraction["aggregate"]["datasets"] if e["name"] == "arXiv")
    assert len(arxiv["papers"]) == 2


def test_gap_agent_sorts_by_severity_and_resolves_papers(sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    result = run_workflow("gap")

    gaps = result["gaps"]
    assert [gap["severity"] for gap in gaps["gaps"]] == ["high", "medium"]
    assert gaps["analysed_papers"] == 2
    # "S1"/"S2" markers must be resolved into the real paper titles.
    assert set(gaps["gaps"][0]["related_papers"]) <= set(gaps["paper_titles"])
    assert set(gaps["gaps"][1]["related_papers"]) <= set(gaps["paper_titles"])


def test_graph_agent_drops_hallucinated_endpoints(sample_pdf, fake_llm):
    paper_id = ingest(sample_pdf)
    result = run_workflow("graph", paper_ids=[paper_id])

    graph = result["graph"]
    names = {node["name"] for node in graph["nodes"]}
    assert "SparseSum" in names
    assert "Hallucinated Entity" not in names

    edge_types = {edge["type"] for edge in graph["edges"]}
    assert "EVALUATED_ON" in edge_types
    assert "MENTIONS" in edge_types  # paper -> entity links
    assert "AUTHORED_BY" in edge_types

    # Every edge endpoint must exist as a node.
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])


def test_review_intent_chains_all_agents(sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)
    result = run_workflow("review")

    nodes = [event["node"] for event in result["trace"]]
    assert "multi_summarize" in nodes
    assert "extract" in nodes
    assert "gap" in nodes
    assert "graph" in nodes

    assert result["summary"] and result["extraction"] and result["gaps"] and result["graph"]
    assert result["status"] == "completed"


# --- failure handling --------------------------------------------------------


def test_missing_api_key_is_reported_not_raised(sample_pdf):
    ingest(sample_pdf)
    result = run_workflow("qa", question="What is the method?")

    assert result["status"] == "partial"
    assert result["answer"] is None
    assert any("GOOGLE_API_KEY" in error for error in result["errors"])
    # Retrieval still worked — the failure is isolated to the LLM node.
    assert result["retrieved"]


def test_agent_failure_does_not_crash_the_workflow(sample_pdf):
    from tests.conftest import FakeGemini
    from app.services.llm import set_llm

    ingest(sample_pdf)
    set_llm(FakeGemini(fail=True))
    result = run_workflow("gap")

    assert result["status"] == "partial"
    assert result["gaps"] is None
    assert any("simulated model failure" in error for error in result["errors"])


def test_workflow_with_no_papers_reports_cleanly(fake_llm):
    result = run_workflow("gap")

    assert result["documents"] == []
    assert any("no indexed papers" in error for error in result["errors"])


def test_runs_are_persisted(sample_pdf, fake_llm):
    from sqlmodel import select

    from app.models import AgentRun

    ingest(sample_pdf)
    run_workflow("qa", question="What is SparseSum?")

    with session_scope() as session:
        runs = list(session.exec(select(AgentRun)).all())

    assert len(runs) == 1
    assert runs[0].intent == "qa"
    assert runs[0].trace
    assert runs[0].result["answer"]


# --- unit helpers ------------------------------------------------------------


def test_query_expansion_produces_variants():
    variants = expand_query("What is retrieval augmented generation?")
    assert variants[0] == "What is retrieval augmented generation?"
    assert len(variants) > 1
    assert len(set(variants)) == len(variants)


def test_canonical_key_normalises_entity_names():
    assert canonical_key("  The BERT ") == "bert"
    assert canonical_key("BERT") == canonical_key("bert")


def test_normalise_extraction_coerces_strings_and_drops_blanks():
    payload = normalise_extraction(
        {"datasets": ["SQuAD", {"name": ""}, {"name": "GLUE", "size": "9 tasks"}], "tasks": ["QA", "  "]}
    )
    assert [entry["name"] for entry in payload["datasets"]] == ["SQuAD", "GLUE"]
    assert payload["tasks"] == ["QA"]
    assert payload["methods"] == []


def test_aggregate_counts_papers_per_entity():
    aggregate = aggregate_extractions(
        [
            {"paper_title": "A", "datasets": [{"name": "GLUE"}], "methods": [], "metrics": [], "tasks": ["QA"]},
            {"paper_title": "B", "datasets": [{"name": "glue"}], "methods": [], "metrics": [], "tasks": ["QA"]},
        ]
    )
    assert len(aggregate["datasets"]) == 1
    assert aggregate["datasets"][0]["papers"] == ["A", "B"]
    assert aggregate["tasks"][0]["papers"] == ["A", "B"]


def test_json_parser_handles_fences_and_trailing_commas():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('prose then {"a": [1, 2,],}') == {"a": [1, 2]}
    assert parse_json_object("[1, 2]") == {"items": [1, 2]}
