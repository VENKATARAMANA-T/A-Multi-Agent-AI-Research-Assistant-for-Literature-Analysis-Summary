"""Shared LangGraph state and helpers for building grounded prompt context."""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from app.services.vector_store import RetrievedChunk

Intent = Literal["qa", "summarize", "multi_summarize", "extract", "gap", "graph", "review"]


class AgentState(TypedDict, total=False):
    """State threaded through the LangGraph workflow.

    Every node reads what it needs and writes only its own keys, so the graph
    stays a set of independent actors over one shared blackboard.
    """

    # --- inputs --------------------------------------------------------------
    intent: Intent
    question: str
    paper_ids: list[str]
    top_k: int
    focus: str | None
    options: dict[str, Any]

    # --- working memory ------------------------------------------------------
    retrieved: list[dict[str, Any]]
    context: str
    documents: list[dict[str, Any]]

    # --- outputs -------------------------------------------------------------
    answer: dict[str, Any]
    summary: dict[str, Any]
    extraction: dict[str, Any]
    gaps: dict[str, Any]
    graph: dict[str, Any]
    report: dict[str, Any]

    # --- bookkeeping ---------------------------------------------------------
    # These three are written by *every* node, including branches that LangGraph
    # runs concurrently in the `review` pipeline. Without reducers, parallel
    # branches would each write a whole list and the last writer would silently
    # discard the others' entries. `operator.add` merges them instead.
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]


def new_state(intent: Intent, **kwargs: Any) -> AgentState:
    state: AgentState = {
        "intent": intent,
        "question": kwargs.pop("question", "") or "",
        "paper_ids": list(kwargs.pop("paper_ids", []) or []),
        "top_k": int(kwargs.pop("top_k", 0) or 0),
        "focus": kwargs.pop("focus", None),
        "options": kwargs.pop("options", {}) or {},
        "retrieved": [],
        "context": "",
        "documents": [],
        "trace": [],
        "errors": [],
        "llm_calls": 0,
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


def trace_event(node: str, status: str, started: float, **details: Any) -> dict[str, Any]:
    return {
        "node": node,
        "status": status,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        **details,
    }


# --- context construction ----------------------------------------------------

MAX_CONTEXT_CHARS = 220_000  # generous; Gemini 2.0 Flash has a 1M-token window


def format_chunk_context(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Number retrieved chunks as [S1], [S2], ... so the model can cite them."""
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        location = f"p.{chunk.page_start}" if chunk.page_start else "n/a"
        section = f", section: {chunk.section}" if chunk.section else ""
        title = chunk.paper_title or chunk.paper_id[:8]

        if getattr(chunk, "kind", "text") == "figure":
            # Label figures explicitly: an answer citing a chart should say so,
            # and the model needs to know it is reading a description of an
            # image rather than the paper's prose.
            header = f"[S{i}] {title} — {chunk.label or 'Figure'} ({location}) [FIGURE]"
        else:
            header = f"[S{i}] {title} ({location}{section})"

        body = chunk.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def format_paper_context(
    documents: list[dict[str, Any]],
    max_chars_per_paper: int = 60_000,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Render whole papers with [S1..Sn] markers, one marker per paper."""
    parts: list[str] = []
    used = 0
    for i, doc in enumerate(documents, start=1):
        meta_bits = []
        if doc.get("authors"):
            meta_bits.append("Authors: " + ", ".join(doc["authors"][:8]))
        if doc.get("year"):
            meta_bits.append(f"Year: {doc['year']}")
        if doc.get("venue"):
            meta_bits.append(f"Venue: {doc['venue']}")
        meta = " | ".join(meta_bits)

        body = (doc.get("text") or "").strip()[:max_chars_per_paper]
        block = f"[S{i}] TITLE: {doc.get('title') or doc.get('id')}\n{meta}\n\n{body}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n========\n\n".join(parts)


def condense_paper_text(text: str, sections: dict[str, Any] | None, budget: int) -> str:
    """Fit a paper into `budget` characters, prioritising the informative sections.

    Full text is used when it fits. Otherwise we assemble abstract, introduction,
    method, results, limitations and conclusion — dropping references, which are
    long and rarely useful for synthesis.
    """
    text = text or ""
    if len(text) <= budget:
        return text

    if sections:
        ordered = [
            "abstract",
            "introduction",
            "method",
            "experiments",
            "results",
            "discussion",
            "limitations",
            "conclusion",
        ]
        per_section = max(1200, budget // max(1, len([s for s in ordered if sections.get(s)])))
        parts = []
        for name in ordered:
            body = (sections.get(name) or "").strip()
            if body:
                parts.append(f"### {name.replace('_', ' ').title()}\n{body[:per_section]}")
        assembled = "\n\n".join(parts)
        if assembled.strip():
            return assembled[:budget]

    # No section map: keep the head (framing) and the tail before references.
    head = text[: int(budget * 0.65)]
    tail = text[-int(budget * 0.35) :]
    return f"{head}\n\n[... middle omitted ...]\n\n{tail}"
