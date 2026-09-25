"""Summarization Agent — single-paper and cross-paper synthesis."""

from __future__ import annotations

import time
from typing import Any

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


def summarize_node(state: AgentState) -> dict[str, Any]:
    """Summarise a single paper (the first loaded document)."""
    documents = state.get("documents") or []
    if not documents:
        started = time.perf_counter()
        return {
            "errors": ["summarize: no document loaded"],
            "trace": [trace_event("summarize", "skipped", started, reason="no document")],
            "llm_calls": 0,
        }

    title = documents[0].get("title") or documents[0].get("id") or "Untitled"

    delta = llm_node(
        name="summarize",
        state=state,
        output_key="summary",
        build_prompt=lambda s: summary_prompt(title, s.get("context", "")),
        system_instruction=SUMMARY_SYSTEM,
        schema=SUMMARY_SCHEMA,
    )

    if delta.get("summary") is not None:
        delta["summary"] = {
            **delta["summary"],
            "scope": "single",
            "paper_id": documents[0].get("id"),
            "paper_title": title,
        }
    return delta


def multi_summarize_node(state: AgentState) -> dict[str, Any]:
    """Synthesise several papers into a comparative overview."""
    documents = state.get("documents") or []

    delta = llm_node(
        name="multi_summarize",
        state=state,
        output_key="summary",
        build_prompt=lambda s: multi_summary_prompt(s.get("context", "")),
        system_instruction=MULTI_SUMMARY_SYSTEM,
        schema=MULTI_SUMMARY_SCHEMA,
    )

    if delta.get("summary") is not None:
        delta["summary"] = {
            **delta["summary"],
            "scope": "multi",
            "paper_ids": [doc.get("id") for doc in documents],
            "paper_titles": [doc.get("title") for doc in documents],
        }
    return delta
