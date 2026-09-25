"""Figures, charts and tables — listing, images, and the costed vision pass."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.agents.figure import analyse_many
from app.config import settings
from app.database import get_session
from app.models import AgentRun, Figure, FigureStatus, Paper, PaperStatus
from app.schemas import (
    FigureAnalysisRequest,
    FigureAnalysisResponse,
    FigureCostEstimate,
    FigureSummary,
)
from app.services.figure_store import apply_analysis, index_figures, pending_figures

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/figures", tags=["figures"])


def _resolve_papers(session: Session, paper_ids: list[str]) -> list[str]:
    if paper_ids:
        found = list(session.exec(select(Paper).where(Paper.id.in_(paper_ids))).all())  # type: ignore[attr-defined]
        missing = set(paper_ids) - {paper.id for paper in found}
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown paper id(s): {', '.join(sorted(missing))}")
        return [paper.id for paper in found]
    return [
        paper.id
        for paper in session.exec(select(Paper).where(Paper.status == PaperStatus.INDEXED)).all()
    ]


@router.get("", response_model=list[FigureSummary], summary="List extracted figures")
def list_figures(
    paper_ids: list[str] | None = Query(default=None),
    kind: str | None = Query(default=None, description="figure | table | chart | algorithm"),
    status: FigureStatus | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[FigureSummary]:
    statement = select(Figure)
    if paper_ids:
        statement = statement.where(Figure.paper_id.in_(paper_ids))  # type: ignore[attr-defined]
    if kind:
        statement = statement.where(Figure.kind == kind)
    if status:
        statement = statement.where(Figure.status == status)

    rows = list(session.exec(statement).all())
    titles = {
        paper.id: paper.title for paper in session.exec(select(Paper)).all()
    }
    rows.sort(key=lambda f: (titles.get(f.paper_id) or "", f.page, f.label))
    return [FigureSummary.from_model(row, titles.get(row.paper_id)) for row in rows]


@router.get("/estimate", response_model=FigureCostEstimate, summary="Cost of analysing figures")
def estimate(
    paper_ids: list[str] | None = Query(default=None),
    session: Session = Depends(get_session),
) -> FigureCostEstimate:
    """How many vision requests a full analysis would spend.

    Shown before the action because on the free tier this is the difference
    between a working day and an exhausted quota.
    """
    resolved = _resolve_papers(session, paper_ids or [])
    outstanding = pending_figures(session, resolved)
    tables_free = list(
        session.exec(
            select(Figure).where(
                Figure.paper_id.in_(resolved),  # type: ignore[attr-defined]
                Figure.status == FigureStatus.SKIPPED,
            )
        ).all()
    ) if resolved else []

    return FigureCostEstimate(
        paper_ids=resolved,
        pending=len(outstanding),
        requests_required=min(len(outstanding), settings.figure_max_analysis_batch),
        capped_at=settings.figure_max_analysis_batch,
        already_readable=len(tables_free),
    )


@router.post("/analyse", response_model=FigureAnalysisResponse, summary="Read figures with the vision model")
def analyse(
    payload: FigureAnalysisRequest,
    session: Session = Depends(get_session),
) -> FigureAnalysisResponse:
    """Run the Figure Agent over pending figures. One request per figure."""
    resolved = _resolve_papers(session, payload.paper_ids)
    outstanding = pending_figures(session, resolved)

    if payload.figure_ids:
        wanted = set(payload.figure_ids)
        outstanding = [figure for figure in outstanding if figure.id in wanted]

    limit = min(payload.limit or settings.figure_max_analysis_batch, settings.figure_max_analysis_batch)
    selected = outstanding[:limit]

    if not selected:
        return FigureAnalysisResponse(
            analysed=0, failed=0, remaining=0, llm_calls=0, duration_ms=0,
            errors=[], figures=[],
            detail="Nothing to analyse — every figure is already read or has no image.",
        )

    titles = {paper.id: paper.title for paper in session.exec(select(Paper)).all()}
    items = []
    for figure in selected:
        items.append(
            {
                "image_png": Path(figure.image_path).read_bytes(),
                "label": figure.label or "Figure",
                "caption": figure.caption or "",
                "paper_title": titles.get(figure.paper_id),
            }
        )

    results, delta = analyse_many(items)

    updated: list[Figure] = []
    for figure, analysis in zip(selected, results):
        updated.append(apply_analysis(session, figure, analysis))

    # Re-index so the new descriptions are searchable immediately.
    for paper_id in {figure.paper_id for figure in updated}:
        index_figures([f for f in updated if f.paper_id == paper_id], titles.get(paper_id))

    analysed = sum(1 for figure in updated if figure.status == FigureStatus.ANALYSED)
    duration = int(delta["trace"][0]["duration_ms"]) if delta.get("trace") else 0

    session.add(
        AgentRun(
            intent="figures",
            paper_ids=resolved,
            status="completed" if analysed and not delta.get("errors") else "partial",
            trace=delta.get("trace") or [],
            result={"analysed": analysed},
            error="; ".join(delta.get("errors") or []) or None,
            duration_ms=duration,
        )
    )
    session.commit()

    return FigureAnalysisResponse(
        analysed=analysed,
        failed=len(updated) - analysed,
        remaining=max(0, len(outstanding) - len(selected)),
        llm_calls=int(delta.get("llm_calls") or 0),
        duration_ms=duration,
        errors=delta.get("errors") or [],
        figures=[FigureSummary.from_model(f, titles.get(f.paper_id)) for f in updated],
    )


@router.get("/{figure_id}", response_model=FigureSummary, summary="Get one figure")
def get_figure(figure_id: str, session: Session = Depends(get_session)) -> FigureSummary:
    figure = session.get(Figure, figure_id)
    if figure is None:
        raise HTTPException(status_code=404, detail="Figure not found.")
    paper = session.get(Paper, figure.paper_id)
    return FigureSummary.from_model(figure, paper.title if paper else None)


@router.get("/{figure_id}/image", summary="The rendered figure image")
def get_image(figure_id: str, session: Session = Depends(get_session)) -> Response:
    figure = session.get(Figure, figure_id)
    if figure is None:
        raise HTTPException(status_code=404, detail="Figure not found.")
    if not figure.image_path or not Path(figure.image_path).exists():
        raise HTTPException(status_code=404, detail="No image was rendered for this figure.")
    return FileResponse(
        figure.image_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
