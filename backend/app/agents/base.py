"""Shared plumbing for LLM-backed agent nodes.

Nodes return a *state delta*, not the whole state. The `trace`, `errors` and
`llm_calls` channels carry `operator.add` reducers, so LangGraph merges the
deltas itself — which is what lets the review pipeline fan out into concurrent
branches without one branch's bookkeeping overwriting another's.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from app.agents.state import AgentState, trace_event
from app.services.llm import (
    GeminiClient,
    LLMError,
    LLMUnavailable,
    get_llm,
    summarise_provider_error,
)

logger = logging.getLogger(__name__)


def llm_node(
    name: str,
    state: AgentState,
    output_key: str,
    build_prompt: Callable[[AgentState], str],
    system_instruction: str,
    schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    require_context: bool = True,
    client: GeminiClient | None = None,
) -> dict[str, Any]:
    """Run one agent node: build prompt -> call Gemini -> return the parsed JSON.

    Returns a delta containing `output_key` (on success) plus `trace`, `errors`
    and `llm_calls`. Failures are recorded rather than raised, so a single
    failing agent never takes down a multi-agent run.
    """
    started = time.perf_counter()

    if require_context and not (state.get("context") or "").strip():
        message = f"{name}: no context available"
        return {
            "errors": [message],
            "trace": [trace_event(name, "skipped", started, reason="empty context")],
            "llm_calls": 0,
        }

    llm = client or get_llm()
    try:
        prompt = build_prompt(state)
        payload = llm.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            schema=schema,
            temperature=temperature,
        )
        return {
            output_key: payload,
            "errors": [],
            "trace": [
                trace_event(name, "ok", started, prompt_chars=len(prompt), keys=sorted(payload.keys()))
            ],
            "llm_calls": 1,
        }
    except LLMUnavailable as exc:
        return {
            "errors": [f"{name}: {exc}"],
            "trace": [trace_event(name, "unavailable", started, error=str(exc))],
            "llm_calls": 0,
        }
    except LLMError as exc:
        # Provider errors are multi-kilobyte JSON blobs; condense before they
        # reach a report, an API response, or a UI banner.
        message = summarise_provider_error(str(exc))
        logger.warning("%s agent failed: %s", name, message)
        return {
            "errors": [f"{name}: {message}"],
            "trace": [trace_event(name, "error", started, error=message)],
            "llm_calls": 0,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("%s agent crashed", name)
        return {
            "errors": [f"{name}: {exc}"],
            "trace": [trace_event(name, "error", started, error=str(exc))],
            "llm_calls": 0,
        }


def merge_deltas(deltas: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Combine sibling deltas produced by parallel per-paper calls."""
    trace: list[dict[str, Any]] = []
    errors: list[str] = []
    calls = 0
    for delta in deltas:
        if not delta:
            continue
        trace.extend(delta.get("trace") or [])
        errors.extend(delta.get("errors") or [])
        calls += int(delta.get("llm_calls") or 0)
    return {"trace": trace, "errors": errors, "llm_calls": calls}
