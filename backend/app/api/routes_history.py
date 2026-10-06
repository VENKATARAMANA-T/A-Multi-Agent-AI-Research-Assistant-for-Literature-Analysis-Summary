"""Saved results — reopening what an agent produced earlier.

Every agent run was already being written to `agent_runs` as an audit trail.
This turns that record into something the user can actually use: a list per
page, and a way to load a past result back into the view that produced it.

Nothing new is stored. The run already holds its own output, so a saved result
costs no extra model calls to look at again — which is the point, on a tier
that allows twenty requests a day.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session, desc, select

from app.api.deps import current_user
from app.database import get_session
from app.models import AgentRun, User
from app.schemas import HistoryDetail, HistoryItem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/history", tags=["history"])

# Only the intents that have a page to reopen them in. Figure analysis and
# graph builds are recorded too, but there is nothing to restore — their output
# already lives in the figures table and the graph database.
REOPENABLE = {"qa", "summarize", "multi_summarize", "extract", "gap", "review"}


def describe(run: AgentRun) -> str:
    """A line identifying the run in a list.

    Derived rather than stored: a title taken at write time goes stale when the
    paper behind it is renamed or deleted.
    """
    if run.question:
        return run.question

    result = run.result or {}
    summary = result.get("summary") or {}
    if summary.get("paper_title"):
        return summary["paper_title"]
    if summary.get("tldr"):
        return summary["tldr"][:120]

    gaps = (result.get("gaps") or {}).get("gaps") or []
    if gaps and isinstance(gaps[0], dict) and gaps[0].get("title"):
        count = len(gaps)
        return f"{count} gap{'s' if count != 1 else ''} — {gaps[0]['title']}"

    papers = len(run.paper_ids or [])
    return f"{papers} paper{'s' if papers != 1 else ''}"


def to_item(run: AgentRun) -> HistoryItem:
    return HistoryItem(
        id=run.id,
        intent=run.intent,
        label=describe(run),
        paper_count=len(run.paper_ids or []),
        status=run.status,
        duration_ms=run.duration_ms,
        has_error=bool(run.error),
        created_at=run.created_at,
    )


@router.get("", response_model=list[HistoryItem], summary="Past results for a page")
def list_history(
    intent: list[str] | None = Query(
        default=None, description="Restrict to these intents, e.g. summarize & multi_summarize."
    ),
    limit: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> list[HistoryItem]:
    wanted = [i for i in (intent or []) if i in REOPENABLE] or sorted(REOPENABLE)

    statement = (
        select(AgentRun)
        .where(AgentRun.owner_id == user.id, AgentRun.intent.in_(wanted))  # type: ignore[attr-defined]
        .order_by(desc(AgentRun.created_at))
        .limit(limit)
    )
    return [to_item(run) for run in session.exec(statement).all()]


@router.get("/{run_id}", response_model=HistoryDetail, summary="Reopen a saved result")
def get_history(
    run_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> HistoryDetail:
    run = session.get(AgentRun, run_id)
    if run is None or run.owner_id != user.id:
        raise HTTPException(status_code=404, detail="That result is no longer saved.")

    result = run.result or {}
    return HistoryDetail(
        id=run.id,
        run_id=run.id,
        intent=run.intent,
        label=describe(run),
        question=run.question,
        paper_ids=run.paper_ids or [],
        status=run.status,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        # Returned under the same keys the live endpoints use, so a page can
        # render a saved result through exactly the same path as a fresh one.
        answer=result.get("answer"),
        summary=result.get("summary"),
        extraction=result.get("extraction"),
        gaps=result.get("gaps"),
        graph=result.get("graph"),
        retrieved=result.get("retrieved") or [],
        graph_facts=result.get("graph_facts") or [],
        graph_matches=result.get("graph_matches") or [],
        documents=result.get("documents") or [],
        trace=run.trace or [],
        errors=[run.error] if run.error else [],
    )


@router.delete("/{run_id}", status_code=204, response_class=Response, summary="Forget a result")
def delete_history(
    run_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    run = session.get(AgentRun, run_id)
    if run is None or run.owner_id != user.id:
        raise HTTPException(status_code=404, detail="That result is no longer saved.")
    session.delete(run)
    session.commit()
    return Response(status_code=204)


@router.delete("", status_code=204, response_class=Response, summary="Clear a page's history")
def clear_history(
    intent: list[str] | None = Query(default=None),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    """Clears one page's saved results, or all of them when no intent is given."""
    wanted = [i for i in (intent or []) if i in REOPENABLE] or sorted(REOPENABLE)

    runs = session.exec(
        select(AgentRun).where(
            AgentRun.owner_id == user.id,
            AgentRun.intent.in_(wanted),  # type: ignore[attr-defined]
        )
    ).all()
    for run in runs:
        session.delete(run)
    session.commit()
    return Response(status_code=204)
