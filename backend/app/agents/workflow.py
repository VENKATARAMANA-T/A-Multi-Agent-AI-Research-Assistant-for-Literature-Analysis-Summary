"""Step 6 — LangGraph orchestration of the specialised agents.

    router ─┬─ retrieve ─ qa ─────────────────────────────── END
            └─ load ─┬─ summarize ───────────────────────── END
                     ├─ multi_summarize ──┐
                     ├─ extract ──────────┤
                     ├─ gap ──────────────┼──────────────── END
                     └─ build_graph ──────┘

A single-intent run enters exactly one branch. The `review` intent fans out to
all four at once: they read the same loaded documents and write disjoint result
keys, so running them serially only added latency. Reducers on `trace`,
`errors` and `llm_calls` merge the concurrent branches' bookkeeping, and a
global semaphore in the LLM client keeps the combined fan-out inside the
provider's rate limit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.extraction import extract_node
from app.agents.gap import gap_node
from app.agents.graph_builder import build_graph_node
from app.agents.qa import qa_node
from app.agents.retrieval import graph_retrieve_node, load_documents_node, retrieve_node
from app.agents.state import AgentState, Intent, new_state, trace_event
from app.agents.summarization import multi_summarize_node, summarize_node

logger = logging.getLogger(__name__)

REVIEW_INTENT: Intent = "review"


REVIEW_BRANCHES = ["multi_summarize", "extract", "gap", "build_graph"]


def router_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    return {
        "trace": [
            trace_event(
                "router",
                "ok",
                started,
                intent=state.get("intent"),
                papers=len(state.get("paper_ids") or []),
            )
        ]
    }


def route_after_router(state: AgentState) -> str:
    return "retrieve" if state.get("intent") == "qa" else "load"


def route_after_load(state: AgentState) -> str | list[str]:
    """Dispatch to one agent, or — for `review` — fan out to all of them.

    The four review agents read the same loaded documents and write disjoint
    keys, so there is no reason to run them one after another. Returning a list
    tells LangGraph to execute them concurrently; the reducers on `trace`,
    `errors` and `llm_calls` merge their bookkeeping on the way back in.
    """
    intent = state.get("intent")
    if not state.get("documents"):
        return END
    if intent == REVIEW_INTENT:
        return REVIEW_BRANCHES
    return {
        "summarize": "summarize",
        "multi_summarize": "multi_summarize",
        "extract": "extract",
        "gap": "gap",
        "graph": "build_graph",
    }.get(str(intent), END)


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("graph_retrieve", graph_retrieve_node)
    graph.add_node("qa", qa_node)
    graph.add_node("load", load_documents_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("multi_summarize", multi_summarize_node)
    graph.add_node("extract", extract_node)
    graph.add_node("gap", gap_node)
    graph.add_node("build_graph", build_graph_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {"retrieve": "retrieve", "load": "load"})
    # Vector retrieval then graph retrieval, in series: they write different
    # keys and graph traversal is local, so there is nothing to gain from
    # running them concurrently and a simpler graph to reason about.
    graph.add_edge("retrieve", "graph_retrieve")
    graph.add_edge("graph_retrieve", "qa")
    graph.add_edge("qa", END)

    graph.add_conditional_edges(
        "load",
        route_after_load,
        {
            "summarize": "summarize",
            "multi_summarize": "multi_summarize",
            "extract": "extract",
            "gap": "gap",
            "build_graph": "build_graph",
            END: END,
        },
    )

    # Every branch terminates. For a single-intent run only one of them was
    # entered; for `review` all four ran concurrently and LangGraph joins them.
    for node in ("summarize", *REVIEW_BRANCHES):
        graph.add_edge(node, END)

    return graph.compile()


_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_workflow(
    intent: Intent,
    question: str = "",
    paper_ids: list[str] | None = None,
    top_k: int | None = None,
    focus: str | None = None,
    persist: bool = True,
    owner_id: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Execute the workflow and return a serialisable result envelope."""
    started = time.perf_counter()
    state = new_state(
        intent,
        question=question,
        paper_ids=paper_ids or [],
        top_k=top_k or 0,
        focus=focus,
        options=options,
    )

    error: str | None = None
    try:
        final: dict[str, Any] = get_workflow().invoke(state)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Workflow failed for intent=%s", intent)
        error = str(exc)
        final = {**state, "errors": [*state.get("errors", []), str(exc)]}

    duration_ms = int((time.perf_counter() - started) * 1000)
    errors = list(final.get("errors") or [])

    result = {
        "intent": intent,
        "question": question or None,
        "paper_ids": final.get("paper_ids") or [],
        "answer": final.get("answer"),
        "summary": final.get("summary"),
        "extraction": final.get("extraction"),
        "gaps": final.get("gaps"),
        "graph": final.get("graph"),
        "retrieved": final.get("retrieved") or [],
        "graph_facts": final.get("graph_facts") or [],
        "graph_matches": final.get("graph_matches") or [],
        "documents": [
            {"id": doc.get("id"), "title": doc.get("title"), "year": doc.get("year")}
            for doc in (final.get("documents") or [])
        ],
        "trace": final.get("trace") or [],
        "errors": errors,
        "llm_calls": final.get("llm_calls", 0),
        "duration_ms": duration_ms,
        "status": "failed" if error else ("partial" if errors else "completed"),
    }

    if persist:
        # The id goes back to the caller so a result can be handed to the
        # Verification Agent later without re-running the whole workflow.
        result["run_id"] = _persist_run(result, error, owner_id)
    return result


def _persist_run(
    result: dict[str, Any], error: str | None, owner_id: str | None = None
) -> str | None:
    """Write an audit record; never let bookkeeping break the request."""
    try:
        from app.database import session_scope
        from app.models import AgentRun

        # The retrieved passages are kept alongside the answer so a run reopened
        # from history renders exactly as it did when it ran. Without them the
        # citations survive but the evidence behind them does not, and a saved
        # answer becomes less checkable than a fresh one.
        payload = {
            key: result.get(key)
            for key in (
                "answer", "summary", "extraction", "gaps", "graph",
                "retrieved", "graph_facts", "graph_matches", "documents",
            )
            if result.get(key)
        }
        with session_scope() as session:
            run = AgentRun(
                intent=str(result["intent"]),
                question=result.get("question"),
                paper_ids=result.get("paper_ids") or [],
                status=str(result["status"]),
                trace=result.get("trace") or [],
                result=payload,
                error=error or ("; ".join(result["errors"]) if result["errors"] else None),
                duration_ms=int(result["duration_ms"]),
                owner_id=owner_id,
            )
            session.add(run)
            session.commit()
            return run.id
    except Exception:  # pragma: no cover
        logger.warning("Could not persist agent run", exc_info=True)
        return None
