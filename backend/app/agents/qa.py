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
        ),
        system_instruction=QA_SYSTEM,
        schema=QA_SCHEMA,
    )

    answer = delta.get("answer")
    if answer:
        delta["answer"] = attach_sources(answer, state.get("retrieved", []))
    return delta


def attach_sources(answer: dict, retrieved: list[dict]) -> dict:
    """Resolve the [S#] markers the model cited back to real chunk records."""
    markers = answer.get("supporting_sources") or []
    resolved = []
    for marker in markers:
        digits = "".join(ch for ch in str(marker) if ch.isdigit())
        if not digits:
            continue
        idx = int(digits) - 1
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
    return answer
