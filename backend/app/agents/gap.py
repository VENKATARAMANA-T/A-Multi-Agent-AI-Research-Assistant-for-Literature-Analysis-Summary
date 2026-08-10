"""Research Gap Agent — reads across the corpus to find what is missing."""

from __future__ import annotations

import time

from app.agents.base import llm_node
from app.agents.prompts import GAP_SCHEMA, GAP_SYSTEM, gap_prompt
from app.agents.state import AgentState, trace_event

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def gap_node(state: AgentState) -> AgentState:
    started = time.perf_counter()
    documents = state.get("documents") or []

    if len(documents) < 2:
        state["trace"] = [
            *state.get("trace", []),
            trace_event("gap", "warning", started, note="gap analysis is most reliable with 2+ papers"),
        ]

    state = llm_node(
        name="gap",
        state=state,
        output_key="gaps",
        build_prompt=lambda s: gap_prompt(s.get("context", "")),
        system_instruction=GAP_SYSTEM,
        schema=GAP_SCHEMA,
        temperature=0.35,
    )

    gaps = state.get("gaps")
    if gaps:
        state["gaps"] = _postprocess(gaps, documents)
    return state


def _postprocess(gaps: dict, documents: list[dict]) -> dict:
    """Sort by severity and map the model's [S#] markers to real paper titles."""
    titles = [doc.get("title") or doc.get("id") for doc in documents]

    items = []
    for gap in gaps.get("gaps") or []:
        if not isinstance(gap, dict):
            continue
        related = gap.get("related_papers") or []
        resolved: list[str] = []
        for entry in related:
            entry = str(entry).strip()
            digits = "".join(ch for ch in entry if ch.isdigit())
            if entry.upper().startswith("S") and digits:
                idx = int(digits) - 1
                if 0 <= idx < len(titles):
                    resolved.append(titles[idx])
                    continue
            resolved.append(entry)
        items.append({**gap, "related_papers": resolved})

    items.sort(key=lambda gap: SEVERITY_ORDER.get(str(gap.get("severity", "medium")).lower(), 1))

    return {
        **gaps,
        "gaps": items,
        "paper_ids": [doc.get("id") for doc in documents],
        "paper_titles": titles,
        "analysed_papers": len(documents),
    }
