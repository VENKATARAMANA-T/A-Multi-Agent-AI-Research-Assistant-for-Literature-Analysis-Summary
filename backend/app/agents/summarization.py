"""Summarization Agent — single-paper and cross-paper synthesis."""

from __future__ import annotations

import time

from app.agents.base import llm_node
from app.agents.prompts import (
    MULTI_SUMMARY_SCHEMA,
    MULTI_SUMMARY_SYSTEM,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM,
    multi_summary_prompt,
    summary_prompt,
)
from app.agents.state import AgentState, trace_event


def summarize_node(state: AgentState) -> AgentState:
    """Summarise a single paper (the first loaded document)."""
    documents = state.get("documents") or []
    if not documents:
        started = time.perf_counter()
        state["errors"] = [*state.get("errors", []), "summarize: no document loaded"]
        state["trace"] = [*state.get("trace", []), trace_event("summarize", "skipped", started, reason="no document")]
        return state

    title = documents[0].get("title") or documents[0].get("id") or "Untitled"

    state = llm_node(
        name="summarize",
        state=state,
        output_key="summary",
        build_prompt=lambda s: summary_prompt(title, s.get("context", "")),
        system_instruction=SUMMARY_SYSTEM,
        schema=SUMMARY_SCHEMA,
    )

    if state.get("summary") is not None:
        state["summary"] = {
            **state["summary"],
            "scope": "single",
            "paper_id": documents[0].get("id"),
            "paper_title": title,
        }
    return state


def multi_summarize_node(state: AgentState) -> AgentState:
    """Synthesise several papers into a comparative overview."""
    documents = state.get("documents") or []

    state = llm_node(
        name="multi_summarize",
        state=state,
        output_key="summary",
        build_prompt=lambda s: multi_summary_prompt(s.get("context", "")),
        system_instruction=MULTI_SUMMARY_SYSTEM,
        schema=MULTI_SUMMARY_SCHEMA,
    )

    if state.get("summary") is not None:
        state["summary"] = {
            **state["summary"],
            "scope": "multi",
            "paper_ids": [doc.get("id") for doc in documents],
            "paper_titles": [doc.get("title") for doc in documents],
        }
    return state
