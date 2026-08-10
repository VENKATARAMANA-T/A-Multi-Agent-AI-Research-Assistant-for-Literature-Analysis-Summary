"""Regressions found during a live Gemini run.

1. Models append [S#] citation markers to structured *name* fields, splitting
   aggregation buckets and creating duplicate graph nodes.
2. A report whose agents failed rendered empty sections with no explanation.
"""

from __future__ import annotations

from app.agents.cleanup import clean_name, strip_markers
from app.agents.extraction import aggregate_extractions, normalise_extraction
from app.agents.graph_builder import canonical_key
from app.services.report import build_markdown_report, build_notes_block


def test_strip_markers_handles_every_shape():
    assert strip_markers("MedQA [S1]") == "MedQA"
    assert strip_markers("accuracy [S1, S3]") == "accuracy"
    assert strip_markers("ROUGE-L (S2)") == "ROUGE-L"
    assert strip_markers("PubMed abstracts [s1]") == "PubMed abstracts"
    assert strip_markers("plain name") == "plain name"
    assert strip_markers(None) == ""


def test_clean_name_trims_stray_brackets():
    assert clean_name("[BERT]") == "BERT"
    assert clean_name("  SparseSum [S1] ") == "SparseSum"


def test_extraction_names_are_cleaned_but_prose_keeps_citations():
    payload = normalise_extraction(
        {
            "datasets": [{"name": "MedQA [S1]"}, {"name": "PubMed abstracts [S1]", "domain": "biomedical [S1]"}],
            "metrics": [{"name": "accuracy [S1]", "value": "78.4 [S1]"}],
            "limitations": ["English MedQA only [S1]"],
        }
    )

    assert [d["name"] for d in payload["datasets"]] == ["MedQA", "PubMed abstracts"]
    assert payload["datasets"][1]["domain"] == "biomedical"
    assert payload["metrics"][0] == {"name": "accuracy", "value": "78.4"}
    # Free-text fields keep their grounding markers.
    assert payload["limitations"] == ["English MedQA only [S1]"]


def test_marked_and_unmarked_names_aggregate_together():
    aggregate = aggregate_extractions(
        [
            {"paper_title": "A", "datasets": normalise_extraction({"datasets": [{"name": "PubMed"}]})["datasets"],
             "methods": [], "metrics": [], "tasks": []},
            {"paper_title": "B", "datasets": normalise_extraction({"datasets": [{"name": "PubMed [S1]"}]})["datasets"],
             "methods": [], "metrics": [], "tasks": []},
        ]
    )

    assert len(aggregate["datasets"]) == 1
    assert aggregate["datasets"][0]["papers"] == ["A", "B"]


def test_graph_keys_collapse_marked_names():
    assert canonical_key("MedQA [S1]") == canonical_key("MedQA")
    assert canonical_key("The SparseSum [S2]") == "sparsesum"


def test_notes_block_is_empty_when_nothing_failed():
    assert build_notes_block(None) == []
    assert build_notes_block([]) == []


def test_report_explains_missing_sections():
    markdown = build_markdown_report(
        title="Partial Review",
        papers=[{"id": "p1", "title": "Paper One", "authors": ["Ada Lovelace"], "year": 2024, "page_count": 3}],
        notes=["gap: 429 RESOURCE_EXHAUSTED", "multi_summarize: 429 RESOURCE_EXHAUSTED"],
    )

    assert "**Incomplete report.**" in markdown
    assert "429 RESOURCE_EXHAUSTED" in markdown
    # The placeholders are still there, but now they are explained.
    assert "Run the research gap agent" in markdown


def test_report_has_no_notes_block_when_everything_succeeded():
    markdown = build_markdown_report(
        title="Clean Review",
        papers=[{"id": "p1", "title": "Paper One", "authors": ["Ada Lovelace"], "year": 2024, "page_count": 3}],
        notes=[],
    )
    assert "Incomplete report" not in markdown
