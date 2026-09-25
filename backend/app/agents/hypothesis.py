"""Hypothesis Agent — turns an ABC chain into something a researcher can test.

Graph traversal produces a ranked list of A→C pairs and the intermediate terms
that imply them. That is an artefact, not a hypothesis: it says two things are
two hops apart, which is often trivially true. This agent reads the chain and
judges whether the implied connection is *worth* testing, states it as a
falsifiable claim, and proposes how to test it.

It is explicitly allowed — expected — to reject candidates. A traversal that
returns twenty pairs and an agent that calls all twenty promising is useless;
the value is in the filtering.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agents.parallel import map_parallel
from app.agents.state import trace_event
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

HYPOTHESIS_SYSTEM = (
    "You are the Hypothesis Agent of ResearchCompass. You assess candidate "
    "connections produced by literature-based discovery: pairs of concepts that "
    "the literature links only indirectly, through a shared intermediate term.\n"
    "Most such pairs are uninteresting — two hops apart in a graph usually means "
    "nothing. Say so. Only call a candidate promising when the indirect link "
    "suggests something a researcher could actually investigate and that the "
    "source papers do not already state.\n"
    "Judge the connection on the evidence given. Do not invent findings, and do "
    "not assert the connection is true — the whole point is that nobody has "
    "tested it."
)

HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["promising", "plausible", "trivial", "implausible"],
        },
        "hypothesis": {"type": "string"},
        "reasoning": {"type": "string"},
        "mechanism": {"type": "string"},
        "proposed_test": {"type": "string"},
        "novelty": {"type": "string", "enum": ["high", "medium", "low"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "why_not": {"type": "string"},
    },
    "required": ["verdict", "hypothesis", "reasoning"],
}


def format_chains(candidate: dict[str, Any], limit: int = 5) -> str:
    lines = []
    for chain in (candidate.get("chains") or [])[:limit]:
        lines.append(
            f"- {candidate['a_name']} —{chain['a_to_b']}→ {chain['b_name']} "
            f"({chain['b_type']}) —{chain['b_to_c']}→ {candidate['c_name']}"
        )
    return "\n".join(lines) or "- (no chains)"


def build_prompt(candidate: dict[str, Any], corpus_note: str = "") -> str:
    a, c = candidate["a_name"], candidate["c_name"]
    return f"""Two concepts in this corpus are connected only indirectly.

## A
{a}

## C
{c} ({candidate.get('c_type', 'Concept')})

## The paths linking them
{format_chains(candidate)}

## What the graph says
- They are linked through {candidate.get('support', 0)} distinct intermediate term(s).
- No paper states a direct relationship between them.
- {a} appears in {candidate.get('a_paper_count', 0)} paper(s); {c} in {candidate.get('c_paper_count', 0)}.
{corpus_note}

## Your task
- `verdict`: is this worth investigating?
  - "promising" — the indirect link suggests a real, testable connection
  - "plausible" — could be worth a look, but weakly supported
  - "trivial" — true but obvious, or an artefact of how the graph was built
    (for example two methods linked merely because both are neural networks)
  - "implausible" — the concepts do not meaningfully relate
- `hypothesis`: one falsifiable sentence. Phrase it as a claim to be tested,
  not as a fact. Leave empty for a trivial or implausible verdict.
- `reasoning`: why the intermediate term does or does not justify the link.
- `mechanism`: the plausible reason A would relate to C, if there is one.
- `proposed_test`: a concrete study or experiment that would settle it —
  data, comparison and outcome measure. Not "more research is needed".
- `novelty`: would this be new, or is it already well known in the field?
- `why_not`: for a trivial or implausible verdict, say plainly what is wrong.

Be willing to reject. A list where everything is promising is worthless.

Return JSON only."""


def assess(candidate: dict[str, Any], corpus_note: str = "") -> dict[str, Any] | None:
    """One LLM call for one candidate."""
    payload = get_llm().generate_json(
        prompt=build_prompt(candidate, corpus_note),
        system_instruction=HYPOTHESIS_SYSTEM,
        schema=HYPOTHESIS_SCHEMA,
        temperature=0.3,
    )
    return normalise(payload)


def normalise(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "plausible").lower()
    if verdict not in ("promising", "plausible", "trivial", "implausible"):
        verdict = "plausible"
    return {
        "verdict": verdict,
        "hypothesis": str(payload.get("hypothesis") or "").strip() or None,
        "reasoning": str(payload.get("reasoning") or "").strip() or None,
        "mechanism": str(payload.get("mechanism") or "").strip() or None,
        "proposed_test": str(payload.get("proposed_test") or "").strip() or None,
        "novelty": str(payload.get("novelty") or "medium").lower(),
        "confidence": str(payload.get("confidence") or "medium").lower(),
        "why_not": str(payload.get("why_not") or "").strip() or None,
    }


VERDICT_ORDER = {"promising": 0, "plausible": 1, "trivial": 2, "implausible": 3}


def assess_many(
    candidates: list[dict[str, Any]],
    corpus_note: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assess several candidates concurrently. Returns (assessed, trace delta)."""
    started = time.perf_counter()
    if not candidates:
        return [], {
            "trace": [trace_event("hypothesis", "skipped", started, reason="no candidates")],
            "errors": [],
            "llm_calls": 0,
        }

    errors: list[str] = []

    def one(candidate: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return assess(candidate, corpus_note)
        except LLMError as exc:
            message = summarise_provider_error(str(exc))
            errors.append(f"{candidate.get('a_name')} -> {candidate.get('c_name')}: {message}")
            return None
        except Exception as exc:  # pragma: no cover
            logger.exception("Hypothesis assessment crashed")
            errors.append(f"{candidate.get('a_name')} -> {candidate.get('c_name')}: {exc}")
            return None

    results = map_parallel(one, candidates, label="hypothesis")

    assessed: list[dict[str, Any]] = []
    for candidate, judgement in zip(candidates, results):
        assessed.append({**candidate, "assessment": judgement})

    # Surface the interesting ones first; the agent is meant to reject most.
    assessed.sort(
        key=lambda item: (
            VERDICT_ORDER.get((item.get("assessment") or {}).get("verdict", "plausible"), 1),
            -float(item.get("score") or 0),
        )
    )

    succeeded = sum(1 for item in assessed if item.get("assessment"))
    return assessed, {
        "trace": [
            trace_event(
                "hypothesis",
                "ok" if succeeded else "error",
                started,
                assessed=succeeded,
                promising=sum(
                    1
                    for item in assessed
                    if (item.get("assessment") or {}).get("verdict") == "promising"
                ),
            )
        ],
        "errors": errors,
        "llm_calls": succeeded,
    }
