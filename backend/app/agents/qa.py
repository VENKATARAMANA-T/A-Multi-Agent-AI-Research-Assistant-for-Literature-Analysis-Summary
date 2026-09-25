"""Question Answering Agent — RAG over retrieved chunks."""

from __future__ import annotations

from typing import Any

from app.agents.base import llm_node
from app.agents.prompts import QA_SCHEMA, QA_SYSTEM, qa_prompt
from app.agents.state import AgentState


def qa_node(state: AgentState) -> dict[str, Any]:
    delta = llm_node(
        name="qa",
        state=state,
        output_key="answer",
        build_prompt=lambda s: qa_prompt(
            s.get("question", ""),
            s.get("context", ""),
            (s.get("options") or {}).get("history"),
            s.get("graph_context", ""),
        ),
        # In graph-only mode there are no text excerpts, but the graph facts
        # are context in their own right.
        require_context=not (state.get("graph_context") or "").strip(),
        system_instruction=QA_SYSTEM,
        schema=QA_SCHEMA,
    )

    answer = delta.get("answer")
    if answer:
        delta["answer"] = attach_sources(
            answer,
            state.get("retrieved", []),
            state.get("graph_facts", []),
        )
    return delta


def attach_sources(
    answer: dict,
    retrieved: list[dict],
    graph_facts: list[dict] | None = None,
) -> dict:
    """Resolve the [S#] and [G#] markers the model cited back to real records.

    A marker that resolves to nothing is dropped rather than shown: a citation
    the reader cannot follow is worse than no citation.
    """
    markers = answer.get("supporting_sources") or []
    resolved = []
    graph_facts = graph_facts or []
    graph_sources = []

    for marker in markers:
        text = str(marker).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        idx = int(digits) - 1

        # A "G" marker points at a relationship, not a passage.
        if text.upper().lstrip("[").startswith("G"):
            if 0 <= idx < len(graph_facts):
                fact = graph_facts[idx]
                graph_sources.append(
                    {
                        "marker": f"G{idx + 1}",
                        "kind": "graph",
                        "sentence": fact.get("sentence"),
                        "relation": fact.get("relation"),
                        "source": fact.get("source"),
                        "target": fact.get("target"),
                        "paper_ids": fact.get("papers") or [],
                        "paper_titles": fact.get("paper_titles") or [],
                        "evidence": fact.get("evidence"),
                    }
                )
            continue

        if 0 <= idx < len(retrieved):
            chunk = retrieved[idx]
            resolved.append(
                {
                    "marker": f"S{idx + 1}",
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_id": chunk.get("paper_id"),
                    "paper_title": chunk.get("paper_title"),
                    "page": chunk.get("page_start"),
                    "section": chunk.get("section"),
                    "score": chunk.get("score"),
                    "excerpt": (chunk.get("text") or "")[:400],
                    # Carried through so the UI can show the chart itself next
                    # to an answer that was drawn from one.
                    "kind": chunk.get("kind", "text"),
                    "figure_id": chunk.get("figure_id"),
                    "label": chunk.get("label"),
                }
            )

    answer["sources"] = resolved
    answer["graph_sources"] = graph_sources
    return answer
