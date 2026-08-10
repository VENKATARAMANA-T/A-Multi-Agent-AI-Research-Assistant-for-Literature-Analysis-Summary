"""Step 6 — LangGraph orchestration of the specialised agents.

    router ─┬─ retrieve ─ qa ─────────────────────────────── END
            └─ load ─┬─ summarize ───────────────────────── END
                     ├─ multi_summarize ─┐                 END
                     ├─ extract ─────────┤                 END
                     ├─ gap ─────────────┤                 END
                     └─ build_graph ─────┘                 END

The `review` intent chains the branches together
(load → multi_summarize → extract → gap → build_graph) to produce everything a
literature review report needs in a single stateful run.
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
from app.agents.retrieval import load_documents_node, retrieve_node
from app.agents.state import AgentState, Intent, new_state, trace_event
from app.agents.summarization import multi_summarize_node, summarize_node

logger = logging.getLogger(__name__)

REVIEW_INTENT: Intent = "review"


def router_node(state: AgentState) -> AgentState:
    started = time.perf_counter()
    state["trace"] = [
        *state.get("trace", []),
        trace_event(
            "router",
            "ok",
            started,
            intent=state.get("intent"),
            papers=len(state.get("paper_ids") or []),
        ),
    ]
    return state


def route_after_router(state: AgentState) -> str:
    return "retrieve" if state.get("intent") == "qa" else "load"


def route_after_load(state: AgentState) -> str:
    intent = state.get("intent")
    if not state.get("documents"):
        return END
    if intent == REVIEW_INTENT:
        return "multi_summarize"
    return {
        "summarize": "summarize",
        "multi_summarize": "multi_summarize",
        "extract": "extract",
        "gap": "gap",
        "graph": "build_graph",
    }.get(str(intent), END)


def _continue_review(next_node: str):
    def _route(state: AgentState) -> str:
        return next_node if state.get("intent") == REVIEW_INTENT else END

    return _route


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qa", qa_node)
    graph.add_node("load", load_documents_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("multi_summarize", multi_summarize_node)
    graph.add_node("extract", extract_node)
    graph.add_node("gap", gap_node)
    graph.add_node("build_graph", build_graph_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {"retrieve": "retrieve", "load": "load"})
    graph.add_edge("retrieve", "qa")
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

    graph.add_edge("summarize", END)
    graph.add_conditional_edges(
        "multi_summarize", _continue_review("extract"), {"extract": "extract", END: END}
    )
    graph.add_conditional_edges("extract", _continue_review("gap"), {"gap": "gap", END: END})
    graph.add_conditional_edges(
        "gap", _continue_review("build_graph"), {"build_graph": "build_graph", END: END}
    )
    graph.add_edge("build_graph", END)

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
        _persist_run(result, error)
    return result


def _persist_run(result: dict[str, Any], error: str | None) -> None:
    """Write an audit record; never let bookkeeping break the request."""
    try:
        from app.database import session_scope
        from app.models import AgentRun

        payload = {
            key: result.get(key)
            for key in ("answer", "summary", "extraction", "gaps", "graph")
            if result.get(key)
        }
        with session_scope() as session:
            session.add(
                AgentRun(
                    intent=str(result["intent"]),
                    question=result.get("question"),
                    paper_ids=result.get("paper_ids") or [],
                    status=str(result["status"]),
                    trace=result.get("trace") or [],
                    result=payload,
                    error=error or ("; ".join(result["errors"]) if result["errors"] else None),
                    duration_ms=int(result["duration_ms"]),
                )
            )
            session.commit()
    except Exception:  # pragma: no cover
        logger.warning("Could not persist agent run", exc_info=True)
