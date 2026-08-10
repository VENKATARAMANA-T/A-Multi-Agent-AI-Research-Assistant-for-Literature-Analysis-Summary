"""Steps 7-8: the knowledge graph store and report rendering."""

from __future__ import annotations

from app.services.report import (
    build_comparison_table,
    build_gap_section,
    build_markdown_report,
    export_pdf,
    markdown_table,
)

FRAGMENT = {
    "nodes": [
        {"id": "paper:p1", "name": "Paper One", "type": "Paper", "paper_id": "p1", "year": 2023},
        {"id": "entity:bert", "name": "BERT", "type": "Method", "papers": ["p1"], "description": "Encoder"},
        {"id": "entity:glue", "name": "GLUE", "type": "Dataset", "papers": ["p1"], "description": ""},
    ],
    "edges": [
        {"source": "paper:p1", "target": "entity:bert", "type": "MENTIONS", "papers": ["p1"], "evidence": ""},
        {"source": "entity:bert", "target": "entity:glue", "type": "EVALUATED_ON", "papers": ["p1"], "evidence": "on GLUE"},
        {"source": "entity:bert", "target": "entity:ghost", "type": "USES", "papers": ["p1"], "evidence": ""},
    ],
}


# --- graph store -------------------------------------------------------------


def test_persist_and_fetch(memory_graph_store):
    assert memory_graph_store.persist(FRAGMENT) is True
    graph = memory_graph_store.fetch()

    assert len(graph["nodes"]) == 3
    # The edge pointing at a non-existent node must be dropped.
    assert len(graph["edges"]) == 2
    assert graph["backend"] == "in-memory"


def test_persist_is_idempotent_and_merges_papers(memory_graph_store):
    memory_graph_store.persist(FRAGMENT)
    memory_graph_store.persist(
        {
            "nodes": [
                {"id": "paper:p2", "name": "Paper Two", "type": "Paper", "paper_id": "p2"},
                {"id": "entity:bert", "name": "BERT", "type": "Method", "papers": ["p2"]},
            ],
            "edges": [
                {"source": "paper:p2", "target": "entity:bert", "type": "MENTIONS", "papers": ["p2"]}
            ],
        }
    )

    graph = memory_graph_store.fetch()
    bert = next(node for node in graph["nodes"] if node["id"] == "entity:bert")
    assert bert["papers"] == ["p1", "p2"]
    assert len(graph["nodes"]) == 4


def test_fetch_scoped_to_paper(memory_graph_store):
    memory_graph_store.persist(FRAGMENT)
    scoped = memory_graph_store.fetch(paper_ids=["p1"])
    assert len(scoped["nodes"]) == 3

    assert memory_graph_store.fetch(paper_ids=["unknown"])["nodes"] == []


def test_delete_paper_prunes_orphans(memory_graph_store):
    memory_graph_store.persist(FRAGMENT)
    memory_graph_store.delete_paper("p1")

    graph = memory_graph_store.fetch()
    assert graph["nodes"] == []
    assert graph["edges"] == []


def test_store_survives_a_restart(tmp_path):
    from app.services.graph_store import InMemoryGraphStore

    path = tmp_path / "graph.json"
    InMemoryGraphStore(path=path).persist(FRAGMENT)

    reopened = InMemoryGraphStore(path=path)
    assert len(reopened.fetch()["nodes"]) == 3


def test_stats(memory_graph_store):
    memory_graph_store.persist(FRAGMENT)
    stats = memory_graph_store.stats()

    assert stats["node_count"] == 3
    assert stats["edge_count"] == 2
    assert stats["node_types"]["Method"] == 1


def test_clear(memory_graph_store):
    memory_graph_store.persist(FRAGMENT)
    memory_graph_store.clear()
    assert memory_graph_store.fetch()["nodes"] == []


# --- report rendering --------------------------------------------------------


def test_markdown_table_escapes_pipes():
    table = markdown_table(["A"], [["value | with pipe"]])
    assert "value \\| with pipe" in table
    assert table.count("\n") == 2


def test_empty_table_renders_placeholder():
    assert "No data" in markdown_table(["A"], [])


def test_comparison_table_from_extraction():
    extraction = {
        "papers": [
            {
                "paper_title": "Paper One",
                "tasks": ["summarization"],
                "methods": [{"name": "SparseSum"}],
                "datasets": [{"name": "arXiv"}],
                "metrics": [{"name": "ROUGE-L", "value": "44.1"}],
            }
        ]
    }
    table = build_comparison_table(extraction)

    assert "Paper One" in table
    assert "SparseSum" in table
    assert "ROUGE-L: 44.1" in table


def test_comparison_table_without_extraction():
    assert "extraction agent" in build_comparison_table(None)


def test_gap_section_renders_severity_and_direction():
    section = build_gap_section(
        {
            "landscape_summary": "Summary text.",
            "gaps": [
                {
                    "title": "Missing multilingual work",
                    "description": "Only English.",
                    "category": "empirical",
                    "severity": "high",
                    "proposed_direction": "Run a multilingual replication.",
                    "evidence": ["English only [S1]"],
                    "related_papers": ["Paper One"],
                }
            ],
            "open_questions": ["Does it transfer?"],
        }
    )

    assert "Gap 1: Missing multilingual work" in section
    assert "**Severity:** HIGH" in section
    assert "Run a multilingual replication." in section
    assert "Open Questions" in section


def test_full_report_has_every_section():
    papers = [
        {
            "id": "p1",
            "title": "Paper One",
            "authors": ["Ada Lovelace"],
            "year": 2023,
            "venue": "NeurIPS",
            "doi": "10.1000/xyz",
            "page_count": 9,
        }
    ]
    markdown = build_markdown_report(
        title="My Review",
        papers=papers,
        summary={"overview": "Overview text.", "themes": []},
        extraction={"papers": [], "aggregate": {"datasets": [], "methods": [], "metrics": []}},
        gaps={"landscape_summary": "Landscape.", "gaps": []},
        graph_stats={"node_count": 5, "edge_count": 4, "backend": "in-memory", "node_types": {"Paper": 1}},
    )

    for heading in (
        "# My Review",
        "## 1. Corpus Overview",
        "## 2. Literature Synthesis",
        "## 3. Comparative Analysis",
        "## 4. Research Gaps",
        "Knowledge Graph Summary",
        "References",
    ):
        assert heading in markdown

    assert "Ada Lovelace (2023). Paper One." in markdown
    assert "https://doi.org/10.1000/xyz" in markdown


def test_pdf_export_renders_tables_and_headings(tmp_path):
    markdown = build_markdown_report(
        title="PDF Review",
        papers=[{"id": "p1", "title": "Paper One", "authors": ["Ada Lovelace"], "year": 2024, "page_count": 4}],
        summary={"overview": "Some **bold** overview.", "themes": []},
        extraction=None,
        gaps=None,
    )
    output = export_pdf(markdown, tmp_path / "report.pdf", "PDF Review")

    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 1500
