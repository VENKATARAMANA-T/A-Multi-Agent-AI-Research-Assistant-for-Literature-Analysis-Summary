"""Explain Agent — plain-English explanation of a selected passage.

Selecting a dense sentence in the PDF and asking "what does this mean?" is the
single most common thing a reader wants. The passage is explained in the
context of the paper it came from, at a level the reader chooses, and the agent
is explicit about the difference between what the passage says and what it is
adding as background.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.state import trace_event
from app.services.llm import LLMError, get_llm, parse_json_object, summarise_provider_error

LEVELS = {
    "simple": (
        "Explain it to a capable person from outside this field. Expand every "
        "acronym, avoid jargon, and use an everyday analogy where one genuinely fits."
    ),
    "standard": (
        "Explain it to a graduate student in an adjacent field. Assume general "
        "research literacy but not familiarity with this subfield's terminology."
    ),
    "technical": (
        "Explain it to a specialist. Be precise about the method, the assumptions "
        "it rests on, and what the result does and does not establish."
    ),
}

EXPLAIN_SYSTEM = (
    "You are the Explain Agent of ResearchCompass. You take a passage a reader "
    "has selected in a paper and make it clear.\n"
    "Explain what the passage itself says. Where you add background the passage "
    "does not contain, put it in `background` so the reader can tell the two "
    "apart. If the passage is ambiguous or depends on context that is not "
    "supplied, say so rather than resolving it by guessing."
)

EXPLAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["term", "meaning"],
            },
        },
        "background": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation"],
}


def build_prompt(passage: str, level: str, paper_title: str | None, surrounding: str | None) -> str:
    instruction = LEVELS.get(level, LEVELS["standard"])
    context = f"\n## The paper\n{paper_title}\n" if paper_title else ""
    nearby = f"\n## Surrounding text (for context only)\n{surrounding}\n" if surrounding else ""

    return f"""A reader has selected this passage and asked what it means.
{context}{nearby}
## Selected passage
{passage}

## Instructions
{instruction}

- `explanation`: what the passage is saying, in 2-5 sentences.
- `terms`: any technical term or acronym in the passage, each with a one-line meaning.
- `background`: context the passage assumes but does not state. Leave empty if none is needed.
- `why_it_matters`: why this appears in the paper — what it supports or enables.
- `caveats`: anything the passage is ambiguous about, or that cannot be determined
  from the text supplied.

Do not restate the passage verbatim. Do not invent numbers or findings.

Return JSON only."""


def explain(
    passage: str,
    level: str = "standard",
    paper_title: str | None = None,
    surrounding: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Explain a passage. Returns (result, trace delta)."""
    started = time.perf_counter()
    passage = (passage or "").strip()

    if len(passage) < 10:
        return None, {
            "errors": ["explain: the selected passage is too short to explain."],
            "trace": [trace_event("explain", "skipped", started, reason="passage too short")],
            "llm_calls": 0,
        }

    try:
        text = get_llm().generate_json(
            prompt=build_prompt(passage, level, paper_title, surrounding),
            system_instruction=EXPLAIN_SYSTEM,
            schema=EXPLAIN_SCHEMA,
            temperature=0.2,
        )
    except LLMError as exc:
        message = summarise_provider_error(str(exc))
        return None, {
            "errors": [f"explain: {message}"],
            "trace": [trace_event("explain", "error", started, error=message)],
            "llm_calls": 0,
        }

    payload = text if isinstance(text, dict) else parse_json_object(str(text))
    return payload, {
        "errors": [],
        "trace": [trace_event("explain", "ok", started, level=level, chars=len(passage))],
        "llm_calls": 1,
    }
