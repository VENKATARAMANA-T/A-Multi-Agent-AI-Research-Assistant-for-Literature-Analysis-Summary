"""Retrieval Agent — semantic search over the vector store, plus document loading.

Two nodes live here:
  * `retrieve`      — top-k semantic retrieval for RAG question answering,
  * `load_documents`— whole-paper loading for summarisation / extraction / gaps.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlmodel import select

from app.agents.state import (
    AgentState,
    condense_paper_text,
    format_chunk_context,
    format_paper_context,
    trace_event,
)
from app.config import settings
from app.database import session_scope
from app.models import Paper, PaperStatus
from app.services.vector_store import RetrievedChunk, get_vector_store

logger = logging.getLogger(__name__)

# Per-paper character budget shrinks as the corpus grows, so a 20-paper gap
# analysis still fits comfortably in one request.
PAPER_BUDGETS = [(1, 90_000), (3, 45_000), (6, 25_000), (12, 14_000)]
MIN_PAPER_BUDGET = 8_000


def _budget_for(count: int) -> int:
    for threshold, budget in PAPER_BUDGETS:
        if count <= threshold:
            return budget
    return MIN_PAPER_BUDGET


def expand_query(question: str) -> list[str]:
    """Cheap multi-query expansion — no LLM call, just structural variants.

    Retrieving with a couple of phrasings and fusing the results measurably
    improves recall on short questions without adding latency.
    """
    question = question.strip()
    variants = [question]
    lowered = question.lower()
    stripped = lowered.rstrip("?").strip()
    for prefix in ("what is ", "what are ", "how does ", "how do ", "why is ", "why do ", "which "):
        if stripped.startswith(prefix):
            variants.append(stripped[len(prefix) :])
            break
    keywords = [w for w in stripped.split() if len(w) > 3][:12]
    if len(keywords) >= 3:
        variants.append(" ".join(keywords))
    seen: set[str] = set()
    unique = []
    for variant in variants:
        key = variant.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(variant.strip())
    return unique[:3]


def search(question: str, paper_ids: list[str] | None, top_k: int) -> list[RetrievedChunk]:
    """Reciprocal-rank fusion over the expanded query variants."""
    store = get_vector_store()
    variants = expand_query(question)

    fused: dict[str, tuple[float, RetrievedChunk]] = {}
    for variant in variants:
        hits = store.search(variant, top_k=top_k, paper_ids=paper_ids)
        for rank, hit in enumerate(hits):
            weight = 1.0 / (60 + rank)  # RRF with the standard k=60
            previous = fused.get(hit.chunk_id)
            if previous:
                fused[hit.chunk_id] = (previous[0] + weight, previous[1])
            else:
                fused[hit.chunk_id] = (weight, hit)

    ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


def retrieve_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    question = state.get("question", "").strip()
    top_k = state.get("top_k") or settings.retrieval_top_k
    paper_ids = state.get("paper_ids") or None
    mode = (state.get("options") or {}).get("retrieval_mode", "hybrid")

    if mode == "graph":
        # Graph-only: the answer must rest on relationships alone.
        return {
            "retrieved": [],
            "context": "",
            "trace": [trace_event("retrieval", "skipped", started, reason="graph-only mode")],
        }

    if not question:
        return {
            "errors": ["retrieval: no question supplied"],
            "trace": [trace_event("retrieval", "skipped", started, reason="no question")],
        }

    try:
        chunks = search(question, paper_ids, top_k)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return {
            "errors": [f"retrieval: {exc}"],
            "trace": [trace_event("retrieval", "error", started, error=str(exc))],
        }

    return {
        "retrieved": [chunk.to_dict() for chunk in chunks],
        "context": format_chunk_context(chunks),
        "errors": [],
        "trace": [
            trace_event(
                "retrieval",
                "ok",
                started,
                hits=len(chunks),
                queries=expand_query(question),
                top_score=round(chunks[0].score, 4) if chunks else None,
            )
        ],
    }


def graph_retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve facts by walking the knowledge graph.

    Runs after vector retrieval and writes separate keys, so the two forms of
    context complement rather than overwrite each other. A question with no
    matching entity yields nothing and the answer rests on the text alone —
    inventing a subgraph would be worse than having none.
    """
    started = time.perf_counter()
    mode = (state.get("options") or {}).get("retrieval_mode", "hybrid")

    if mode == "vector":
        return {"graph_facts": [], "graph_context": "", "graph_matches": []}

    question = state.get("question", "").strip()
    if not question:
        return {
            "graph_facts": [],
            "graph_context": "",
            "graph_matches": [],
            "trace": [trace_event("graph_retrieval", "skipped", started, reason="no question")],
        }

    try:
        from app.services.graph_retrieval import format_facts, retrieve

        context = retrieve(question, paper_ids=state.get("paper_ids") or None)
    except Exception as exc:
        logger.exception("Graph retrieval failed")
        return {
            "graph_facts": [],
            "graph_context": "",
            "graph_matches": [],
            "errors": [f"graph_retrieval: {exc}"],
            "trace": [trace_event("graph_retrieval", "error", started, error=str(exc))],
        }

    if not context.found:
        return {
            "graph_facts": [],
            "graph_context": "",
            "graph_matches": [match.to_dict() for match in context.matches],
            "trace": [
                trace_event(
                    "graph_retrieval",
                    "empty",
                    started,
                    reason=context.reason,
                    matched=len(context.matches),
                )
            ],
        }

    return {
        "graph_facts": [fact.to_dict() for fact in context.facts],
        "graph_context": format_facts(context.facts),
        "graph_matches": [match.to_dict() for match in context.matches],
        "errors": [],
        "trace": [
            trace_event(
                "graph_retrieval",
                "ok",
                started,
                matched=[match.name for match in context.matches],
                facts=len(context.facts),
                papers=len(context.paper_ids),
            )
        ],
    }


def load_documents_node(state: AgentState) -> dict[str, Any]:
    """Load full paper text for the selected papers (all indexed papers if none given)."""
    started = time.perf_counter()
    paper_ids = state.get("paper_ids") or []

    with session_scope() as session:
        if paper_ids:
            statement = select(Paper).where(Paper.id.in_(paper_ids))  # type: ignore[attr-defined]
        else:
            statement = select(Paper).where(Paper.status == PaperStatus.INDEXED)
        papers = list(session.exec(statement).all())

    papers = [p for p in papers if p.status == PaperStatus.INDEXED]
    if not papers:
        message = "no indexed papers matched the request"
        return {
            "documents": [],
            "context": "",
            "errors": [f"loader: {message}"],
            "trace": [trace_event("loader", "empty", started, reason=message)],
        }

    papers.sort(key=lambda p: (p.year or 0, p.title or ""))
    budget = _budget_for(len(papers))

    documents = [
        {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "venue": paper.venue,
            "abstract": paper.abstract,
            "text": condense_paper_text(paper.full_text or "", paper.sections, budget),
        }
        for paper in papers
    ]

    context = format_paper_context(documents, max_chars_per_paper=budget)
    return {
        "documents": documents,
        "paper_ids": [paper.id for paper in papers],
        "context": context,
        "errors": [],
        "trace": [
            trace_event(
                "loader",
                "ok",
                started,
                papers=len(documents),
                budget_per_paper=budget,
                context_chars=len(context),
            )
        ],
    }
