"""Shared plumbing for LLM-backed agent nodes."""

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
) -> AgentState:
    """Run one agent node: build prompt -> call Gemini -> store parsed JSON.

    Failures are recorded in `errors`/`trace` and the workflow continues, so a
    single failing agent never takes down a multi-agent run.
    """
    started = time.perf_counter()

    if require_context and not (state.get("context") or "").strip():
        message = f"{name}: no context available"
        state["errors"] = [*state.get("errors", []), message]
        state["trace"] = [*state.get("trace", []), trace_event(name, "skipped", started, reason="empty context")]
        return state

    llm = client or get_llm()
    try:
        prompt = build_prompt(state)
        payload = llm.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            schema=schema,
            temperature=temperature,
        )
        state[output_key] = payload  # type: ignore[literal-required]
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        state["trace"] = [
            *state.get("trace", []),
            trace_event(name, "ok", started, prompt_chars=len(prompt), keys=sorted(payload.keys())),
        ]
    except LLMUnavailable as exc:
        state["errors"] = [*state.get("errors", []), f"{name}: {exc}"]
        state["trace"] = [*state.get("trace", []), trace_event(name, "unavailable", started, error=str(exc))]
    except LLMError as exc:
        # Provider errors are multi-kilobyte JSON blobs; condense before they
        # reach a report, an API response, or a UI banner.
        message = summarise_provider_error(str(exc))
        logger.warning("%s agent failed: %s", name, message)
        state["errors"] = [*state.get("errors", []), f"{name}: {message}"]
        state["trace"] = [*state.get("trace", []), trace_event(name, "error", started, error=message)]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("%s agent crashed", name)
        state["errors"] = [*state.get("errors", []), f"{name}: {exc}"]
        state["trace"] = [*state.get("trace", []), trace_event(name, "error", started, error=str(exc))]

    return state
