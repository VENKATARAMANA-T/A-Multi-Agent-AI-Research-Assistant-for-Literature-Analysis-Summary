"""Literature Review Generator — a written review, not a table of results."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, PlainTextResponse
from sqlmodel import Session, desc, select

from app.agents.literature_review import generate
from app.api.deps import current_user, owned_paper_ids
from app.database import get_session
from app.models import Paper, PaperStatus, Report, User
from app.schemas import (
    LiteratureReviewRequest,
    LiteratureReviewResponse,
    LiteratureReviewSummary,
)
from app.services.report import export_pdf, report_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/review", tags=["literature review"])

KIND = "narrative_review"


def _summary(report: Report) -> LiteratureReviewSummary:
    return LiteratureReviewSummary(
        id=report.id,
        topic=report.topic or report.title,
        title=report.title,
        paper_ids=report.paper_ids or [],
        section_count=len(report.sections or []),
        citation_count=len(report.citations or []),
        agent_status=report.agent_status,
        has_pdf=bool(report.pdf_path and Path(report.pdf_path).exists()),
        created_at=report.created_at,
    )


def _owned(session: Session, user: User, review_id: str) -> Report:
    report = session.get(Report, review_id)
    if report is None or report.owner_id != user.id or report.kind != KIND:
        raise HTTPException(status_code=404, detail="Review not found.")
    return report


@router.post("", response_model=LiteratureReviewResponse, summary="Write a literature review")
def create_review(
    payload: LiteratureReviewRequest,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> LiteratureReviewResponse:
    """Ten sections of prose on a topic, cited back to the papers.

    Costs three model calls whatever the corpus size: the sections are written
    in three groups that need each other's context, and the references come
    from stored metadata for nothing.
    """
    started = time.perf_counter()
    paper_ids = owned_paper_ids(session, user, payload.paper_ids or None, minimum=1)

    papers = list(
        session.exec(
            select(Paper).where(
                Paper.id.in_(paper_ids),  # type: ignore[attr-defined]
                Paper.status == PaperStatus.INDEXED,
            )
        ).all()
    )
    # Chronological, because the "Research Evolution" section is a story about
    # time and the model should meet the papers in that order.
    papers.sort(key=lambda p: (p.year or 0, p.title or ""))

    documents = [
        {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "venue": paper.venue,
            "doi": paper.doi,
            "sections": paper.sections,
            "text": paper.full_text or "",
        }
        for paper in papers
    ]

    result = generate(payload.topic.strip(), documents, payload.focus)
    duration_ms = int((time.perf_counter() - started) * 1000)

    if not result.sections:
        raise HTTPException(
            status_code=502,
            detail="; ".join(result.errors) or "The review could not be generated.",
        )

    report_id = None
    created_at = None
    if payload.save:
        report = Report(
            owner_id=user.id,
            title=f"Literature Review: {result.topic}",
            kind=KIND,
            topic=result.topic,
            paper_ids=paper_ids,
            markdown=result.markdown,
            sections=result.sections,
            citations=result.citations,
            llm_calls=result.llm_calls,
            agent_status=result.status,
            agent_errors=result.errors,
        )
        session.add(report)
        session.commit()
        session.refresh(report)
        report_id = report.id
        created_at = report.created_at

        # The PDF is written eagerly: a review that takes three model calls
        # should not need a fourth request before it can be handed to anyone.
        try:
            path = export_pdf(report.markdown, report_path(report.id), report.title)
            report.pdf_path = str(path)
            session.add(report)
            session.commit()
        except Exception:  # pragma: no cover - reportlab edge cases
            logger.exception("Could not render the review to PDF")

    return LiteratureReviewResponse(
        id=report_id,
        topic=result.topic,
        status=result.status,
        sections=result.sections,
        citations=result.citations,
        markdown=result.markdown,
        paper_ids=paper_ids,
        llm_calls=result.llm_calls,
        duration_ms=duration_ms,
        errors=result.errors,
        trace=result.trace,
        created_at=created_at,
    )


@router.get("", response_model=list[LiteratureReviewSummary], summary="Saved reviews")
def list_reviews(
    limit: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> list[LiteratureReviewSummary]:
    rows = session.exec(
        select(Report)
        .where(Report.owner_id == user.id, Report.kind == KIND)
        .order_by(desc(Report.created_at))
        .limit(limit)
    ).all()
    return [_summary(row) for row in rows]


@router.get("/{review_id}", response_model=LiteratureReviewResponse, summary="One saved review")
def get_review(
    review_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> LiteratureReviewResponse:
    report = _owned(session, user, review_id)
    return LiteratureReviewResponse(
        id=report.id,
        topic=report.topic or report.title,
        status=report.agent_status,
        sections=report.sections or [],
        citations=report.citations or [],
        markdown=report.markdown,
        paper_ids=report.paper_ids or [],
        llm_calls=report.llm_calls,
        errors=report.agent_errors or [],
        created_at=report.created_at,
    )


@router.get("/{review_id}/markdown", response_class=PlainTextResponse, summary="Download Markdown")
def get_markdown(
    review_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> PlainTextResponse:
    report = _owned(session, user, review_id)
    return PlainTextResponse(
        report.markdown,
        headers={"Content-Disposition": f'attachment; filename="review-{report.id}.md"'},
    )


@router.get("/{review_id}/pdf", summary="Download PDF")
def get_pdf(
    review_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> FileResponse:
    report = _owned(session, user, review_id)

    path = Path(report.pdf_path) if report.pdf_path else report_path(report.id)
    if not path.exists():
        # Rendered on demand when the eager write failed or the file was swept up.
        try:
            path = export_pdf(report.markdown, path, report.title)
            report.pdf_path = str(path)
            session.add(report)
            session.commit()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Could not render the PDF: {exc}") from exc

    return FileResponse(path, media_type="application/pdf", filename=f"review-{report.id}.pdf")


@router.delete("/{review_id}", status_code=204, response_class=Response, summary="Delete a review")
def delete_review(
    review_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    report = _owned(session, user, review_id)
    if report.pdf_path:
        Path(report.pdf_path).unlink(missing_ok=True)
    session.delete(report)
    session.commit()
    return Response(status_code=204)
