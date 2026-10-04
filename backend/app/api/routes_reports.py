"""Literature review report generation, listing, and PDF export."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, PlainTextResponse
from sqlmodel import Session, desc, select

from app.agents.state import condense_paper_text, format_paper_context
from app.agents.workflow import run_workflow
from app.api.deps import current_user, owned_paper_ids
from app.database import get_session
from app.models import Paper, PaperStatus, Report, User
from app.schemas import ReportDetail, ReportRequest, ReportSummary
from app.services.graph_store import get_graph_store
from app.services.report import (
    build_markdown_report,
    export_pdf,
    generate_narrative,
    report_path,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("", response_model=ReportDetail, summary="Generate a literature review report")
def create_report(
    payload: ReportRequest,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> ReportDetail:
    """Runs the full review pipeline, then renders Markdown + PDF."""
    statement = select(Paper).where(
        Paper.status == PaperStatus.INDEXED, Paper.owner_id == user.id
    )
    if payload.paper_ids:
        statement = statement.where(Paper.id.in_(payload.paper_ids))  # type: ignore[attr-defined]
    papers = list(session.exec(statement).all())

    if not papers:
        raise HTTPException(status_code=409, detail="No indexed papers available for this report.")

    papers.sort(key=lambda p: (p.year or 0, p.title or ""))
    paper_ids = [paper.id for paper in papers]

    result = run_workflow("review", paper_ids=paper_ids, focus=payload.focus)

    paper_dicts = [
        {
            "id": paper.id,
            "title": paper.title,
            "filename": paper.filename,
            "authors": paper.authors or [],
            "year": paper.year,
            "venue": paper.venue,
            "doi": paper.doi,
            "page_count": paper.page_count,
        }
        for paper in papers
    ]

    notes: list[str] = list(result.get("errors") or [])

    narrative = None
    if payload.include_narrative:
        context = format_paper_context(
            [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors or [],
                    "year": paper.year,
                    "venue": paper.venue,
                    "text": condense_paper_text(paper.full_text or "", paper.sections, 18_000),
                }
                for paper in papers
            ]
        )
        narrative = generate_narrative(context, payload.focus)
        if narrative is None:
            notes.append("narrative: the prose synthesis could not be generated.")

    graph_stats = None
    if payload.include_graph:
        try:
            graph_stats = get_graph_store().stats()
        except Exception:  # pragma: no cover
            logger.warning("Could not read graph stats for the report", exc_info=True)

    title = payload.title or f"Literature Review: {len(papers)} Papers"
    markdown = build_markdown_report(
        title=title,
        papers=paper_dicts,
        summary=result.get("summary"),
        extraction=result.get("extraction"),
        gaps=result.get("gaps"),
        narrative=narrative,
        graph_stats=graph_stats,
        notes=notes,
    )

    report = Report(
        owner_id=user.id,
        title=title,
        kind="literature_review",
        paper_ids=paper_ids,
        markdown=markdown,
        agent_status="partial" if notes else str(result.get("status", "completed")),
        agent_errors=notes,
    )
    session.add(report)
    session.commit()
    session.refresh(report)

    try:
        pdf_file = export_pdf(markdown, report_path(report.id), title)
        report.pdf_path = str(pdf_file)
        session.add(report)
        session.commit()
        session.refresh(report)
    except Exception:  # pragma: no cover - PDF is a nice-to-have
        logger.warning("PDF export failed for report %s", report.id, exc_info=True)

    return _detail(report)


@router.get("", response_model=list[ReportSummary], summary="List reports")
def list_reports(
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> list[ReportSummary]:
    reports = session.exec(
        select(Report)
        .where(Report.owner_id == user.id)
        .order_by(desc(Report.created_at))
        .limit(limit)
    ).all()
    return [
        ReportSummary(
            id=report.id,
            title=report.title,
            kind=report.kind,
            paper_ids=report.paper_ids,
            has_pdf=bool(report.pdf_path and Path(report.pdf_path).exists()),
            agent_status=report.agent_status,
            agent_errors=report.agent_errors or [],
            created_at=report.created_at,
        )
        for report in reports
    ]


@router.get("/{report_id}", response_model=ReportDetail, summary="Get a report")
def get_report(
    report_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> ReportDetail:
    report = session.get(Report, report_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found.")
    return _detail(report)


@router.get("/{report_id}/markdown", response_class=PlainTextResponse, summary="Download Markdown")
def get_report_markdown(
    report_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> PlainTextResponse:
    report = session.get(Report, report_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found.")
    filename = f"{report.title[:60].replace(' ', '_')}.md"
    return PlainTextResponse(
        report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{report_id}/pdf", summary="Download PDF")
def get_report_pdf(
    report_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> FileResponse:
    report = session.get(Report, report_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found.")

    path = Path(report.pdf_path) if report.pdf_path else report_path(report.id)
    if not path.exists():
        try:
            path = export_pdf(report.markdown, report_path(report.id), report.title)
            report.pdf_path = str(path)
            session.add(report)
            session.commit()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{report.title[:60].replace(' ', '_')}.pdf",
    )


@router.delete("/{report_id}", status_code=204, response_class=Response, summary="Delete a report")
def delete_report(
    report_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    report = session.get(Report, report_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.pdf_path:
        Path(report.pdf_path).unlink(missing_ok=True)
    session.delete(report)
    session.commit()
    return Response(status_code=204)


def _detail(report: Report) -> ReportDetail:
    return ReportDetail(
        id=report.id,
        title=report.title,
        kind=report.kind,
        paper_ids=report.paper_ids,
        has_pdf=bool(report.pdf_path and Path(report.pdf_path).exists()),
        agent_status=report.agent_status,
        agent_errors=report.agent_errors or [],
        created_at=report.created_at,
        markdown=report.markdown,
    )
