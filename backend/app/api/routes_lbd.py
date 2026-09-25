"""Literature-Based Discovery endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session, desc, select

from app.agents.hypothesis import assess_many
from app.database import get_session
from app.models import AgentRun, Hypothesis
from app.schemas import ClosedLbdResponse, HypothesisSummary, LbdRequest, LbdResponse
from app.services.graph_store import get_graph_store
from app.services.lbd import closed_discovery, open_discovery, rankable_terms

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lbd", tags=["discovery"])


@router.get("/terms", summary="Entities worth starting a search from")
def terms(limit: int = Query(default=40, ge=1, le=200)) -> list[dict]:
    return rankable_terms(get_graph_store().fetch(), limit=limit)


@router.post("", response_model=LbdResponse, summary="Find implied connections (ABC model)")
def discover(payload: LbdRequest, session: Session = Depends(get_session)) -> LbdResponse:
    """Open discovery: given A, propose every C the literature implies.

    `strict` applies Swanson's criterion — A and C must share no paper at all,
    so the connection spans genuinely separate literatures. `unstated` only
    requires that no paper states the link directly, which is what a small
    corpus can realistically produce.
    """
    graph = get_graph_store().fetch()

    candidates, diagnostics = open_discovery(
        graph,
        payload.source,
        limit=payload.limit,
        min_support=payload.min_support,
        require_disjoint=(payload.mode == "strict"),
    )

    if diagnostics.get("reason"):
        raise HTTPException(status_code=404, detail=diagnostics["reason"])

    items = [candidate.to_dict() for candidate in candidates]
    errors: list[str] = []
    llm_calls = 0
    saved_ids: list[str] = []

    if payload.assess and items:
        note = _corpus_note(diagnostics, payload.mode)
        assessed, delta = assess_many(items[: payload.assess_limit], corpus_note=note)
        errors = delta.get("errors") or []
        llm_calls = int(delta.get("llm_calls") or 0)

        # Assessed candidates replace their unassessed originals, in order.
        remaining = items[payload.assess_limit :]
        items = assessed + remaining

        for item in assessed:
            judgement = item.get("assessment")
            if not judgement:
                continue
            saved_ids.append(_save(session, item, judgement, payload.mode))

        session.add(
            AgentRun(
                intent="lbd",
                question=payload.source,
                paper_ids=[],
                status="completed" if llm_calls and not errors else "partial",
                trace=delta.get("trace") or [],
                result={"candidates": len(items), "assessed": llm_calls},
                error="; ".join(errors) or None,
                duration_ms=int((delta.get("trace") or [{}])[0].get("duration_ms", 0)),
            )
        )
        session.commit()

    return LbdResponse(
        source=diagnostics.get("source_resolved") or payload.source,
        mode=payload.mode,
        candidates=items,
        diagnostics=diagnostics,
        errors=errors,
        llm_calls=llm_calls,
        saved_ids=saved_ids,
    )


def _corpus_note(diagnostics: dict, mode: str) -> str:
    if mode == "strict":
        return "- The two concepts appear in entirely separate papers."
    return (
        "- They may appear in the same paper, but no paper states a direct "
        "relationship. Weigh whether the link is genuinely unstated or merely "
        "too obvious to write down."
    )


def _save(session: Session, item: dict, judgement: dict, mode: str) -> str:
    paper_ids = sorted(
        {
            paper
            for chain in (item.get("chains") or [])
            for paper in (chain.get("a_papers") or []) + (chain.get("c_papers") or [])
        }
    )
    record = Hypothesis(
        source_term=item.get("a_name", ""),
        target_term=item.get("c_name", ""),
        target_type=item.get("c_type", "Concept"),
        mode=mode,
        support=int(item.get("support") or 0),
        score=float(item.get("score") or 0),
        chains=item.get("chains") or [],
        paper_ids=paper_ids,
        verdict=judgement.get("verdict"),
        statement=judgement.get("hypothesis"),
        reasoning=judgement.get("reasoning"),
        mechanism=judgement.get("mechanism"),
        proposed_test=judgement.get("proposed_test"),
        novelty=judgement.get("novelty"),
        confidence=judgement.get("confidence"),
        why_not=judgement.get("why_not"),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record.id


@router.post("/closed", response_model=ClosedLbdResponse, summary="Why are A and C linked?")
def closed(payload: LbdRequest) -> ClosedLbdResponse:
    """Closed discovery: given both ends, show the intermediate terms."""
    if not payload.target:
        raise HTTPException(status_code=400, detail="Closed discovery needs a target term.")

    chains, diagnostics = closed_discovery(get_graph_store().fetch(), payload.source, payload.target)
    if diagnostics.get("reason"):
        raise HTTPException(status_code=404, detail=diagnostics["reason"])

    return ClosedLbdResponse(
        source=diagnostics.get("source_resolved") or payload.source,
        target=diagnostics.get("target_resolved") or payload.target,
        chains=[chain.to_dict() for chain in chains],
        diagnostics=diagnostics,
    )


@router.get("/hypotheses", response_model=list[HypothesisSummary], summary="Saved hypotheses")
def list_hypotheses(
    verdict: str | None = Query(default=None),
    starred: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[HypothesisSummary]:
    statement = select(Hypothesis).order_by(desc(Hypothesis.created_at)).limit(limit)
    if verdict:
        statement = statement.where(Hypothesis.verdict == verdict)
    if starred is not None:
        statement = statement.where(Hypothesis.starred == starred)
    return [HypothesisSummary(**row.model_dump()) for row in session.exec(statement).all()]


@router.post("/hypotheses/{hypothesis_id}/star", response_model=HypothesisSummary, summary="Star a hypothesis")
def star(hypothesis_id: str, starred: bool = Query(default=True),
         session: Session = Depends(get_session)) -> HypothesisSummary:
    record = session.get(Hypothesis, hypothesis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found.")
    record.starred = starred
    session.add(record)
    session.commit()
    session.refresh(record)
    return HypothesisSummary(**record.model_dump())


@router.delete("/hypotheses/{hypothesis_id}", status_code=204, summary="Delete a hypothesis")
def delete_hypothesis(hypothesis_id: str, session: Session = Depends(get_session)) -> Response:
    record = session.get(Hypothesis, hypothesis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found.")
    session.delete(record)
    session.commit()
    return Response(status_code=204)
