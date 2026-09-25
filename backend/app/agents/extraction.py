"""Information Extraction Agent — datasets, methods, metrics, tasks, limitations."""

from __future__ import annotations

import time
from typing import Any

from app.agents.base import llm_node, merge_deltas
from app.agents.cleanup import clean_name, strip_markers
from app.agents.parallel import map_parallel
from app.agents.prompts import EXTRACTION_SCHEMA, EXTRACTION_SYSTEM, extraction_prompt
from app.agents.state import AgentState, format_paper_context, trace_event

LIST_FIELDS = ("tasks", "research_questions", "limitations", "tools", "cited_works")
OBJECT_FIELDS = ("datasets", "methods", "metrics")


def extract_node(state: AgentState) -> dict[str, Any]:
    """Extract structured entities from every loaded document, one call per paper."""
    started = time.perf_counter()
    documents = state.get("documents") or []
    if not documents:
        return {
            "errors": ["extract: no document loaded"],
            "trace": [trace_event("extract", "skipped", started, reason="no document")],
            "llm_calls": 0,
        }

    def extract_one(document: dict) -> dict[str, Any]:
        title = document.get("title") or document.get("id") or "Untitled"
        # Fresh bookkeeping per paper, otherwise the parent's counters are
        # copied in and then added back, double-counting every call.
        single_state: AgentState = {
            **state,
            "context": format_paper_context([document]),
            "trace": [],
            "errors": [],
            "llm_calls": 0,
        }
        return llm_node(
            name=f"extract[{title[:40]}]",
            state=single_state,
            output_key="extraction",
            build_prompt=lambda s, t=title: extraction_prompt(t, s.get("context", "")),
            system_instruction=EXTRACTION_SYSTEM,
            schema=EXTRACTION_SCHEMA,
            temperature=0.0,
        )

    # The per-paper calls are independent, so run them concurrently. Results
    # come back in input order, keeping the aggregate deterministic.
    outcomes = map_parallel(extract_one, documents, label="extract")
    merged = merge_deltas(outcomes)

    per_paper: list[dict] = []
    for document, single in zip(documents, outcomes):
        payload = (single or {}).get("extraction")
        if payload:
            per_paper.append(
                {
                    "paper_id": document.get("id"),
                    "paper_title": document.get("title") or document.get("id") or "Untitled",
                    **normalise_extraction(payload),
                }
            )

    return {
        "extraction": {"papers": per_paper, "aggregate": aggregate_extractions(per_paper)},
        "errors": merged["errors"],
        "llm_calls": merged["llm_calls"],
        "trace": [
            *merged["trace"],
            trace_event("extract", "ok" if per_paper else "error", started, papers=len(per_paper)),
        ],
    }


def normalise_extraction(payload: dict) -> dict:
    """Coerce the model's output into the shape the rest of the system expects."""
    normalised: dict = {}

    for field in OBJECT_FIELDS:
        items = payload.get(field) or []
        cleaned = []
        for item in items:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict):
                continue
            name = clean_name(item.get("name"))
            if not name:
                continue
            attributes = {
                key: strip_markers(value) if isinstance(value, str) else value
                for key, value in item.items()
                if value and key != "name"
            }
            cleaned.append({**attributes, "name": name})
        normalised[field] = cleaned

    for field in LIST_FIELDS:
        items = payload.get(field) or []
        # Prose fields keep their citations; only names are stripped above.
        normalised[field] = [str(item).strip() for item in items if str(item).strip()]

    return normalised


def aggregate_extractions(per_paper: list[dict]) -> dict:
    """Roll per-paper entities up into corpus-level counts for the comparison view."""
    aggregate: dict[str, dict[str, dict]] = {field: {} for field in OBJECT_FIELDS}
    tasks: dict[str, set[str]] = {}

    for record in per_paper:
        paper_title = record.get("paper_title") or record.get("paper_id")
        for field in OBJECT_FIELDS:
            for item in record.get(field, []):
                key = item["name"].strip().lower()
                bucket = aggregate[field].setdefault(
                    key, {"name": item["name"], "papers": [], "details": []}
                )
                if paper_title not in bucket["papers"]:
                    bucket["papers"].append(paper_title)
                detail = {k: v for k, v in item.items() if k != "name"}
                if detail:
                    bucket["details"].append({"paper": paper_title, **detail})
        for task in record.get("tasks", []):
            tasks.setdefault(task.strip().lower(), set()).add(paper_title)

    def ranked(field: str) -> list[dict]:
        return sorted(
            aggregate[field].values(),
            key=lambda entry: (-len(entry["papers"]), entry["name"].lower()),
        )

    return {
        "datasets": ranked("datasets"),
        "methods": ranked("methods"),
        "metrics": ranked("metrics"),
        "tasks": sorted(
            ({"name": name.title(), "papers": sorted(papers)} for name, papers in tasks.items()),
            key=lambda entry: (-len(entry["papers"]), entry["name"]),
        ),
    }
