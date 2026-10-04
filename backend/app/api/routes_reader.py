"""Reader features: citation highlighting, explain-selection, citation export."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, select

from app.agents.explain import explain as run_explain
from app.api.deps import current_user, owned_paper, owned_paper_ids
from app.database import get_session
from app.models import Chunk, Paper, User
from app.schemas import (
    CitationResponse,
    ExplainRequest,
    ExplainResponse,
    HighlightResponse,
)
from app.services import citations
from app.services.highlight import find_highlight

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reader", tags=["reader"])


@router.get("/highlight", response_model=HighlightResponse, summary="Locate a passage in its PDF")
def highlight(
    paper_id: str = Query(...),
    chunk_id: str | None = Query(default=None),
    figure_id: str | None = Query(default=None),
    text: str | None = Query(default=None, max_length=4000),
    page: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> HighlightResponse:
    """Return the rectangles a passage occupies, so a citation can be shown in place.

    Search happens server-side with PyMuPDF rather than against the browser's
    text layer: the stored chunk text differs from the page's raw text after
    ligature and hyphenation normalisation, so matching needs the same engine
    that produced it.
    """
    paper = owned_paper(session, user, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if not paper.file_path or not Path(paper.file_path).exists():
        raise HTTPException(status_code=404, detail="The stored PDF file is missing.")

    needle = text
    hint = page

    if chunk_id:
        chunk = session.get(Chunk, chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found.")
        needle = chunk.text
        hint = hint or chunk.page_start
    elif figure_id:
        from app.models import Figure

        figure = session.get(Figure, figure_id)
        if figure is None:
            raise HTTPException(status_code=404, detail="Figure not found.")
        # A figure has coordinates already — no text search needed.
        return HighlightResponse(
            page=figure.page,
            rects=[figure.bbox] if figure.bbox else [],
            matched_phrase=figure.label,
            found=bool(figure.bbox),
            page_width=0,
            page_height=0,
        )

    if not needle:
        raise HTTPException(status_code=400, detail="Provide chunk_id, figure_id or text.")

    try:
        result = find_highlight(paper.file_path, needle, page_number=hint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return HighlightResponse(**result.to_dict())


@router.post("/explain", response_model=ExplainResponse, summary="Explain a selected passage")
def explain_passage(
    payload: ExplainRequest,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> ExplainResponse:
    paper_title = None
    if payload.paper_id:
        paper = owned_paper(session, user, payload.paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found.")
        paper_title = paper.title

    result, delta = run_explain(
        passage=payload.text,
        level=payload.level,
        paper_title=paper_title,
        surrounding=payload.surrounding,
    )

    return ExplainResponse(
        explanation=(result or {}).get("explanation"),
        terms=(result or {}).get("terms") or [],
        background=(result or {}).get("background"),
        why_it_matters=(result or {}).get("why_it_matters"),
        caveats=(result or {}).get("caveats") or [],
        errors=delta.get("errors") or [],
        status="completed" if result else "partial",
    )


@router.get("/citation/{paper_id}", response_model=CitationResponse, summary="Cite one paper")
def citation(
    paper_id: str,
    style: str = Query(default="bibtex"),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> CitationResponse:
    paper = owned_paper(session, user, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    try:
        text = citations.format_citation(paper, style)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CitationResponse(style=style.lower(), text=text, count=1)


@router.get("/citations", response_class=PlainTextResponse, summary="Export a bibliography")
def bibliography(
    style: str = Query(default="bibtex"),
    paper_ids: list[str] | None = Query(default=None),
    download: bool = Query(default=False),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
):
    if style.lower() not in citations.STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown citation style '{style}'. Choose from: {', '.join(citations.STYLES)}",
        )

    statement = select(Paper).where(Paper.owner_id == user.id)
    if paper_ids:
        statement = statement.where(Paper.id.in_(paper_ids))  # type: ignore[attr-defined]
    papers = list(session.exec(statement).all())
    if not papers:
        raise HTTPException(status_code=404, detail="No papers matched.")

    papers.sort(key=lambda p: ((p.authors or [""])[0], p.year or 0))
    body = citations.format_many(papers, style)

    headers = {}
    if download:
        extension = citations.file_extension(style)
        headers["Content-Disposition"] = f'attachment; filename="references.{extension}"'

    return PlainTextResponse(body, media_type=citations.media_type(style), headers=headers)
