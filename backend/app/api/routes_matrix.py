"""The AI matrix — user-defined comparison columns across papers."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, desc, select

from app.agents.matrix import run_matrix, to_csv
from app.agents.retrieval import load_documents_node
from app.agents.state import new_state
from app.database import get_session
from app.models import MatrixRun, Paper, PaperStatus
from app.schemas import MatrixRequest, MatrixResponse, MatrixRunSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/matrix", tags=["matrix"])

MAX_COLUMNS = 12


@router.post("", response_model=MatrixResponse, summary="Fill custom columns across papers")
def create_matrix(payload: MatrixRequest, session: Session = Depends(get_session)) -> MatrixResponse:
    """One LLM call per paper. Columns are defined by the caller."""
    if not payload.columns:
        raise HTTPException(status_code=400, detail="Define at least one column.")
    if len(payload.columns) > MAX_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_COLUMNS} columns per run; more makes each answer less reliable.",
        )

    if payload.paper_ids:
        found = list(session.exec(select(Paper).where(Paper.id.in_(payload.paper_ids))).all())  # type: ignore[attr-defined]
        missing = set(payload.paper_ids) - {paper.id for paper in found}
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown paper id(s): {', '.join(sorted(missing))}")
        resolved = [paper.id for paper in found if paper.status == PaperStatus.INDEXED]
    else:
        resolved = [
            paper.id
            for paper in session.exec(select(Paper).where(Paper.status == PaperStatus.INDEXED)).all()
        ]

    if not resolved:
        raise HTTPException(status_code=409, detail="No indexed papers to compare.")

    started = time.perf_counter()
    loaded = load_documents_node(new_state("extract", paper_ids=resolved))
    documents = loaded.get("documents") or []
    if not documents:
        raise HTTPException(status_code=409, detail="Could not load the selected papers.")

    result = run_matrix(documents, [column.model_dump() for column in payload.columns])
    duration = int((time.perf_counter() - started) * 1000)

    run = MatrixRun(
        name=payload.name or f"Comparison of {len(documents)} papers",
        paper_ids=resolved,
        columns=result["columns"],
        rows=result["rows"],
        errors=result["errors"],
        llm_calls=result["llm_calls"],
        duration_ms=duration,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    return MatrixResponse(
        id=run.id,
        name=run.name,
        columns=result["columns"],
        rows=result["rows"],
        errors=result["errors"],
        llm_calls=result["llm_calls"],
        duration_ms=duration,
        status="completed" if result["rows"] and not result["errors"] else "partial",
    )


@router.get("", response_model=list[MatrixRunSummary], summary="List saved comparisons")
def list_runs(
    limit: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[MatrixRunSummary]:
    runs = session.exec(select(MatrixRun).order_by(desc(MatrixRun.created_at)).limit(limit)).all()
    return [
        MatrixRunSummary(
            id=run.id,
            name=run.name,
            paper_count=len(run.paper_ids or []),
            column_count=len(run.columns or []),
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=MatrixResponse, summary="Get a saved comparison")
def get_run(run_id: str, session: Session = Depends(get_session)) -> MatrixResponse:
    run = session.get(MatrixRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    return MatrixResponse(
        id=run.id,
        name=run.name,
        columns=run.columns or [],
        rows=run.rows or [],
        errors=run.errors or [],
        llm_calls=run.llm_calls,
        duration_ms=run.duration_ms,
        status="completed",
    )


@router.get("/{run_id}/csv", response_class=PlainTextResponse, summary="Download as CSV")
def download_csv(run_id: str, session: Session = Depends(get_session)):
    run = session.get(MatrixRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    body = to_csv(run.columns or [], run.rows or [])
    filename = (run.name or "comparison")[:50].replace(" ", "_")
    return PlainTextResponse(
        body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
    )


@router.delete("/{run_id}", status_code=204, summary="Delete a comparison")
def delete_run(run_id: str, session: Session = Depends(get_session)):
    from fastapi import Response

    run = session.get(MatrixRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    session.delete(run)
    session.commit()
    return Response(status_code=204)
