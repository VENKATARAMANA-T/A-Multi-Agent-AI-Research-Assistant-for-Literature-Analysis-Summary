"""Verification endpoints — fact-check generated text against the corpus."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session, desc, select

from app.agents.verification import extract_text, verify
from app.api.deps import current_user, owned_paper_ids
from app.database import get_session
from app.models import AgentRun, User, VerificationRecord
from app.schemas import (
    VerificationDetail,
    VerificationSummary,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/verify", tags=["verification"])


@router.post("", response_model=VerifyResponse, summary="Fact-check generated text")
def run_verification(
    payload: VerifyRequest,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> VerifyResponse:
    """Check each claim against evidence retrieved independently for it.

    Costs two model calls — one to split the text into claims, one to judge
    them all — regardless of how long the text is.
    """
    started = time.perf_counter()

    text = (payload.text or "").strip()
    paper_ids = list(payload.paper_ids)
    source = payload.source
    subject = payload.subject

    if payload.agent_run_id:
        run = session.get(AgentRun, payload.agent_run_id)
        if run is None or run.owner_id != user.id:
            raise HTTPException(status_code=404, detail="Agent run not found.")
        if not text:
            text = extract_text(run.result or {})
        if not paper_ids:
            paper_ids = list(run.paper_ids or [])
        if source == "text":
            source = run.intent
        subject = subject or run.question

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Nothing to verify — supply text, or an agent run that produced some.",
        )

    # Evidence is retrieved afresh, so the scope has to be this account's
    # papers — otherwise a verification could quote somebody else's corpus.
    scope = owned_paper_ids(session, user, paper_ids or None)
    result = verify(text, scope or None, payload.evidence_per_claim)
    duration_ms = int((time.perf_counter() - started) * 1000)

    record_id = None
    if payload.save and result.claims:
        record = VerificationRecord(
            owner_id=user.id,
            agent_run_id=payload.agent_run_id,
            source=source,
            subject=subject,
            text=text,
            paper_ids=paper_ids,
            score=result.score,
            claims=result.claims,
            counts=result.counts,
            checked=result.checked,
            status=result.status,
            errors=result.errors,
            llm_calls=result.llm_calls,
            duration_ms=duration_ms,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        record_id = record.id

    return VerifyResponse(
        id=record_id,
        source=source,
        subject=subject,
        status=result.status,
        score=round(result.score, 3),
        checked=result.checked,
        counts=result.counts,
        claims=result.claims,
        problems=result.problems,
        errors=result.errors,
        llm_calls=result.llm_calls,
        duration_ms=duration_ms,
        trace=result.trace,
    )


@router.get("", response_model=list[VerificationSummary], summary="Past verifications")
def list_verifications(
    agent_run_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> list[VerificationSummary]:
    statement = (
        select(VerificationRecord)
        .where(VerificationRecord.owner_id == user.id)
        .order_by(desc(VerificationRecord.created_at))
        .limit(limit)
    )
    if agent_run_id:
        statement = statement.where(VerificationRecord.agent_run_id == agent_run_id)
    if source:
        statement = statement.where(VerificationRecord.source == source)
    return [VerificationSummary(**row.model_dump()) for row in session.exec(statement).all()]


@router.get("/{verification_id}", response_model=VerificationDetail, summary="One verification")
def get_verification(
    verification_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> VerificationDetail:
    record = session.get(VerificationRecord, verification_id)
    if record is None or record.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Verification not found.")
    return VerificationDetail(**record.model_dump())


@router.delete("/{verification_id}", status_code=204, summary="Delete a verification")
def delete_verification(
    verification_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    record = session.get(VerificationRecord, verification_id)
    if record is None or record.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Verification not found.")
    session.delete(record)
    session.commit()
    return Response(status_code=204)
