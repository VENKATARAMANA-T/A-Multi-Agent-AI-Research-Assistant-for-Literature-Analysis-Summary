"""Verification Agent — fact-checks the other agents against the source text.

Every other agent in this system produces claims. This one checks them.

The design decision that makes it worth having: **the verifier retrieves its
own evidence**. Handed the same context the original agent used, a model
reliably agrees with itself — the output looks like an audit and carries none
of the value. Each claim is therefore used as a fresh retrieval query against
the corpus, so the evidence is gathered independently of whatever the first
agent happened to see. A claim the original context supported but the corpus
does not is exactly the failure this is built to catch.

Cost is two calls regardless of length: one to split the text into atomic
claims, one to judge them all together. Judging claims individually would be
more accurate and far too expensive to ever be switched on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.state import trace_event
from app.config import settings
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

VERDICTS = ("supported", "partially_supported", "unsupported", "contradicted", "unverifiable")

# A claim scores 1.0 when fully supported; partial credit reflects that a
# half-right claim is not as bad as an invented one, but is not fine either.
VERDICT_SCORES = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
    "contradicted": 0.0,
    "unverifiable": 0.0,
}

PROBLEM_VERDICTS = frozenset({"unsupported", "contradicted"})

MAX_CLAIMS = 25
EVIDENCE_PER_CLAIM = 4
EVIDENCE_CHARS = 700


# --- claim extraction --------------------------------------------------------

CLAIM_SYSTEM = (
    "You are the Verification Agent of ResearchCompass. Your first job is to "
    "break a piece of generated text into the individual factual claims it "
    "makes, so each can be checked separately.\n"
    "Split faithfully. Do not add claims the text does not make, do not merge "
    "two assertions into one, and do not soften a strong claim into a weaker "
    "one — the point is to check what was actually said."
)

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["factual", "numeric", "causal", "comparative", "general"],
                    },
                    "checkable": {"type": "boolean"},
                },
                "required": ["text"],
            },
        }
    },
    "required": ["claims"],
}


def claim_prompt(text: str) -> str:
    return f"""Break the text below into the separate factual claims it makes.

## Text
{text}

## Rules
- One assertion per claim, stated so it can be checked on its own. Resolve
  pronouns: "it reaches 91.2 F1" becomes "SparseSum reaches 91.2 F1".
- Keep numbers, dataset names and comparisons exactly as written. A claim that
  drops the number cannot be checked.
- `type`: "numeric" for a reported figure, "causal" for a cause-and-effect
  claim, "comparative" for one thing beating another, "factual" otherwise.
- `checkable`: false for hedges, opinions and generalities ("further work is
  needed"), which cannot be confirmed or refuted from the papers.
- Do not include citation markers like [S1] in the claim text.
- At most {MAX_CLAIMS} claims. If the text makes more, keep the most substantive.

Return JSON only."""


def extract_claims(text: str) -> list[dict[str, Any]]:
    payload = get_llm().generate_json(
        prompt=claim_prompt(text),
        system_instruction=CLAIM_SYSTEM,
        schema=CLAIM_SCHEMA,
        temperature=0.0,
    )

    claims: list[dict[str, Any]] = []
    for item in payload.get("claims") or []:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        if len(body) < 8:
            continue
        claims.append(
            {
                "text": body,
                "type": str(item.get("type") or "factual"),
                "checkable": bool(item.get("checkable", True)),
            }
        )
    return claims[:MAX_CLAIMS]


# --- independent evidence ----------------------------------------------------


def gather_evidence(
    claim: str,
    paper_ids: list[str] | None,
    top_k: int = EVIDENCE_PER_CLAIM,
) -> list[dict[str, Any]]:
    """Retrieve evidence for a claim, from scratch.

    Deliberately does *not* reuse the original agent's context: searching the
    corpus with the claim itself is what makes this a check rather than a
    restatement.
    """
    from app.agents.retrieval import search

    try:
        hits = search(claim, paper_ids or None, top_k)
    except Exception:  # pragma: no cover - retrieval problems surface elsewhere
        logger.warning("Evidence retrieval failed for a claim", exc_info=True)
        return []

    return [
        {
            "chunk_id": hit.chunk_id,
            "paper_id": hit.paper_id,
            "paper_title": hit.paper_title,
            "page": hit.page_start,
            "score": hit.score,
            "text": hit.text[:EVIDENCE_CHARS],
        }
        for hit in hits
    ]


# --- judgement ---------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are the Verification Agent of ResearchCompass. You decide whether each "
    "claim is supported by the evidence passages retrieved for it.\n"
    "Judge only against the evidence shown. Do not use outside knowledge: a "
    "claim that is true in general but absent from these papers is unsupported, "
    "and saying so is the whole point.\n"
    "Be exacting with numbers. A claim of 91.2 against evidence of 89.7 is "
    "contradicted, not supported. A claim about all papers, backed by evidence "
    "from one, is partially supported at best."
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence_quote": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["claim_index", "verdict"],
            },
        }
    },
    "required": ["judgements"],
}


def judge_prompt(claims: list[dict[str, Any]]) -> str:
    blocks = []
    for index, claim in enumerate(claims):
        evidence = claim.get("evidence") or []
        if evidence:
            passages = "\n".join(
                f"  - ({item['paper_title'] or item['paper_id']}, p.{item['page'] or '?'}) "
                f"{item['text']}"
                for item in evidence
            )
        else:
            passages = "  - (nothing relevant was found in the corpus)"
        blocks.append(f"### Claim {index}\n{claim['text']}\n\nEvidence retrieved for it:\n{passages}")

    body = "\n\n".join(blocks)

    return f"""Judge each claim against the evidence retrieved for it.

{body}

## Verdicts
- `supported` — the evidence states this claim.
- `partially_supported` — some of it holds, or it is broader than the evidence
  warrants (for example "all papers" where only one is shown).
- `unsupported` — the evidence does not address it. Use this freely; it is the
  common case for an invented claim.
- `contradicted` — the evidence says something different. Numbers that do not
  match belong here, not in "unsupported".
- `unverifiable` — the claim is an opinion or a hedge that no evidence could settle.

For each judgement give `evidence_quote`: the exact words from the evidence that
decided it, or empty when nothing was found. `explanation`: one sentence.

Return one judgement per claim, with `claim_index` matching the numbers above.

Return JSON only."""


def judge_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = get_llm().generate_json(
        prompt=judge_prompt(claims),
        system_instruction=JUDGE_SYSTEM,
        schema=JUDGE_SCHEMA,
        temperature=0.0,
    )

    by_index: dict[int, dict[str, Any]] = {}
    for item in payload.get("judgements") or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("claim_index"))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict") or "unsupported").lower()
        if verdict not in VERDICTS:
            verdict = "unsupported"
        by_index[index] = {
            "verdict": verdict,
            "confidence": str(item.get("confidence") or "medium").lower(),
            "evidence_quote": str(item.get("evidence_quote") or "").strip() or None,
            "explanation": str(item.get("explanation") or "").strip() or None,
        }

    judged = []
    for index, claim in enumerate(claims):
        judgement = by_index.get(index)
        if judgement is None:
            # A claim the model skipped must not silently pass.
            judgement = {
                "verdict": "unverifiable",
                "confidence": "low",
                "evidence_quote": None,
                "explanation": "The verifier returned no judgement for this claim.",
            }
        judged.append({**claim, **judgement})
    return judged


# --- orchestration -----------------------------------------------------------


@dataclass
class VerificationResult:
    claims: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    problems: list[dict[str, Any]] = field(default_factory=list)
    checked: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors and not self.claims:
            return "failed"
        return "partial" if self.errors else "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "score": round(self.score, 3),
            "counts": self.counts,
            "problems": self.problems,
            "checked": self.checked,
            "llm_calls": self.llm_calls,
            "errors": self.errors,
            "trace": self.trace,
            "status": self.status,
        }


def verify(
    text: str,
    paper_ids: list[str] | None = None,
    evidence_per_claim: int = EVIDENCE_PER_CLAIM,
) -> VerificationResult:
    """Extract claims, gather evidence independently, and judge each one."""
    started = time.perf_counter()
    result = VerificationResult()

    text = (text or "").strip()
    if len(text) < 20:
        result.errors.append("verification: the text is too short to contain checkable claims.")
        result.trace.append(trace_event("verify", "skipped", started, reason="text too short"))
        return result

    try:
        claims = extract_claims(text)
        result.llm_calls += 1
    except LLMError as exc:
        message = summarise_provider_error(str(exc))
        result.errors.append(f"verification: {message}")
        result.trace.append(trace_event("verify", "error", started, error=message))
        return result

    if not claims:
        result.trace.append(trace_event("verify", "empty", started, reason="no checkable claims"))
        return result

    checkable = [claim for claim in claims if claim.get("checkable", True)]
    for claim in checkable:
        claim["evidence"] = gather_evidence(claim["text"], paper_ids, evidence_per_claim)

    if not checkable:
        result.claims = [{**claim, "verdict": "unverifiable"} for claim in claims]
        result.trace.append(
            trace_event("verify", "ok", started, claims=len(claims), checkable=0)
        )
        return result

    try:
        judged = judge_claims(checkable)
        result.llm_calls += 1
    except LLMError as exc:
        message = summarise_provider_error(str(exc))
        result.errors.append(f"verification: {message}")
        result.trace.append(trace_event("verify", "error", started, error=message))
        return result

    unchecked = [
        {**claim, "verdict": "unverifiable", "explanation": "Not a checkable factual claim."}
        for claim in claims
        if not claim.get("checkable", True)
    ]

    result.claims = judged + unchecked
    result.checked = len(judged)
    result.counts = {
        verdict: sum(1 for claim in result.claims if claim.get("verdict") == verdict)
        for verdict in VERDICTS
    }
    result.counts = {key: value for key, value in result.counts.items() if value}
    result.score = (
        sum(VERDICT_SCORES.get(claim.get("verdict", "unsupported"), 0.0) for claim in judged)
        / len(judged)
        if judged
        else 0.0
    )
    result.problems = [
        claim for claim in result.claims if claim.get("verdict") in PROBLEM_VERDICTS
    ]
    result.trace.append(
        trace_event(
            "verify",
            "ok",
            started,
            claims=len(result.claims),
            checked=result.checked,
            score=round(result.score, 3),
            problems=len(result.problems),
        )
    )
    return result


def extract_text(payload: dict[str, Any]) -> str:
    """Pull the checkable prose out of a stored agent result.

    An agent run holds structured output; verification needs the assertions it
    contains, wherever in that structure they live.
    """
    parts: list[str] = []

    answer = payload.get("answer") or {}
    if isinstance(answer, dict):
        if answer.get("answer"):
            parts.append(str(answer["answer"]))
        parts.extend(str(item) for item in (answer.get("caveats") or []))

    summary = payload.get("summary") or {}
    if isinstance(summary, dict):
        for key in ("tldr", "overview", "problem", "approach"):
            if summary.get(key):
                parts.append(str(summary[key]))
        for key in ("key_findings", "contributions", "limitations", "agreements", "disagreements"):
            parts.extend(str(item) for item in (summary.get(key) or []))
        for theme in summary.get("themes") or []:
            if isinstance(theme, dict) and theme.get("description"):
                parts.append(str(theme["description"]))

    gaps = payload.get("gaps") or {}
    if isinstance(gaps, dict):
        if gaps.get("landscape_summary"):
            parts.append(str(gaps["landscape_summary"]))
        for gap in gaps.get("gaps") or []:
            if isinstance(gap, dict):
                for key in ("title", "description"):
                    if gap.get(key):
                        parts.append(str(gap[key]))

    return "\n".join(part.strip() for part in parts if part and part.strip())
