"""Multi-agent endpoints — every one of these runs the LangGraph workflow."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.agents.workflow import run_workflow
from app.database import get_session
from app.models import AgentRun, ExtractionRecord, Paper, PaperStatus, SummaryRecord
from app.schemas import (
    AgentRunResponse,
    AskRequest,
    GapRequest,
    PaperIdsRequest,
    SummarizeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])


def _validate_papers(session: Session, paper_ids: list[str], minimum: int = 1) -> list[str]:
    """Resolve the requested papers, defaulting to the whole indexed corpus."""
    if paper_ids:
        found = list(session.exec(select(Paper).where(Paper.id.in_(paper_ids))).all())  # type: ignore[attr-defined]
        missing = set(paper_ids) - {paper.id for paper in found}
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown paper id(s): {', '.join(sorted(missing))}")
        unindexed = [p.id for p in found if p.status != PaperStatus.INDEXED]
        if unindexed:
            raise HTTPException(
                status_code=409,
                detail=f"These papers are not indexed yet: {', '.join(unindexed)}",
            )
        resolved = [paper.id for paper in found]
    else:
        resolved = [
            paper.id for paper in session.exec(select(Paper).where(Paper.status == PaperStatus.INDEXED)).all()
        ]

    if len(resolved) < minimum:
        raise HTTPException(
            status_code=409,
            detail=f"This analysis needs at least {minimum} indexed paper(s); found {len(resolved)}.",
        )
    return resolved


@router.post("/ask", response_model=AgentRunResponse, summary="Question Answering Agent (RAG)")
def ask(payload: AskRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    paper_ids = _validate_papers(session, payload.paper_ids)
    result = run_workflow(
        "qa",
        question=payload.question,
        paper_ids=paper_ids,
        top_k=payload.top_k,
    )
    return AgentRunResponse(**result)


@router.post("/summarize", response_model=AgentRunResponse, summary="Summarization Agent")
def summarize(payload: SummarizeRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    paper_ids = _validate_papers(session, payload.paper_ids)

    if payload.scope == "single":
        if len(paper_ids) != 1:
            raise HTTPException(status_code=400, detail="Single-paper summarisation needs exactly one paper id.")
        result = run_workflow("summarize", paper_ids=paper_ids)
    else:
        result = run_workflow("multi_summarize", paper_ids=paper_ids)

    if result.get("summary"):
        session.add(
            SummaryRecord(
                paper_id=paper_ids[0] if payload.scope == "single" else None,
                scope=payload.scope,
                paper_ids=paper_ids,
                payload=result["summary"],
            )
        )
        session.commit()

    return AgentRunResponse(**result)


@router.post("/extract", response_model=AgentRunResponse, summary="Information Extraction Agent")
def extract(payload: PaperIdsRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    paper_ids = _validate_papers(session, payload.paper_ids)
    result = run_workflow("extract", paper_ids=paper_ids)

    for record in (result.get("extraction") or {}).get("papers", []):
        if record.get("paper_id"):
            session.add(ExtractionRecord(paper_id=record["paper_id"], payload=record))
    session.commit()

    return AgentRunResponse(**result)


@router.post("/gaps", response_model=AgentRunResponse, summary="Research Gap Agent")
def find_gaps(payload: GapRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    paper_ids = _validate_papers(session, payload.paper_ids)
    result = run_workflow("gap", paper_ids=paper_ids, focus=payload.focus)
    return AgentRunResponse(**result)


@router.post("/graph/build", response_model=AgentRunResponse, summary="Knowledge Graph Agent")
def build_graph(payload: PaperIdsRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    paper_ids = _validate_papers(session, payload.paper_ids)
    result = run_workflow("graph", paper_ids=paper_ids)
    return AgentRunResponse(**result)


@router.post("/review", response_model=AgentRunResponse, summary="Full pipeline (summary + extraction + gaps + graph)")
def full_review(payload: GapRequest, session: Session = Depends(get_session)) -> AgentRunResponse:
    """Runs every agent in one stateful LangGraph execution."""
    paper_ids = _validate_papers(session, payload.paper_ids)
    result = run_workflow("review", paper_ids=paper_ids, focus=payload.focus)

    if result.get("summary"):
        session.add(
            SummaryRecord(scope="multi", paper_ids=paper_ids, payload=result["summary"])
        )
    for record in (result.get("extraction") or {}).get("papers", []):
        if record.get("paper_id"):
            session.add(ExtractionRecord(paper_id=record["paper_id"], payload=record))
    session.commit()

    return AgentRunResponse(**result)


@router.get("/runs", summary="Recent agent runs (audit trail)")
def list_runs(
    limit: int = Query(default=25, ge=1, le=200),
    intent: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[dict]:
    statement = select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
    if intent:
        statement = statement.where(AgentRun.intent == intent)
    return [
        {
            "id": run.id,
            "intent": run.intent,
            "question": run.question,
            "paper_ids": run.paper_ids,
            "status": run.status,
            "duration_ms": run.duration_ms,
            "error": run.error,
            "trace": run.trace,
            "created_at": run.created_at,
        }
        for run in session.exec(statement).all()
    ]


@router.get("/runs/{run_id}", summary="One agent run with its full result payload")
def get_run(run_id: str, session: Session = Depends(get_session)) -> dict:
    run = session.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found.")
    return {
        "id": run.id,
        "intent": run.intent,
        "question": run.question,
        "paper_ids": run.paper_ids,
        "status": run.status,
        "duration_ms": run.duration_ms,
        "error": run.error,
        "trace": run.trace,
        "result": run.result,
        "created_at": run.created_at,
    }
