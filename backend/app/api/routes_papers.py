"""Paper upload, listing, retrieval, deletion, and semantic search."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, select

from app.api.deps import current_user, owned_paper, owned_paper_ids
from app.database import get_session, session_scope
from app.services import jobs
from app.models import Chunk, Paper, PaperStatus, User
from app.schemas import (
    PaperSummary,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UploadResponse,
    UploadResultItem,
)
from app.services.ingestion import IngestionError, delete_paper, process_paper, store_upload
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/papers", tags=["papers"])

MAX_FILES_PER_REQUEST = 20


@router.post("/upload", summary="Upload and index PDFs")
async def upload_papers(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    wait: bool = Query(
        default=False,
        description="Block until indexing finishes instead of returning a job to poll.",
    ),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Any:
    """Steps 1-5: store the PDF, extract text, chunk, embed and index it.

    By default the work is queued and a job is returned immediately, because a
    twenty-file upload otherwise holds the request open for minutes with no
    feedback. Poll `/api/jobs/{id}` or subscribe to `/api/jobs/{id}/stream` for
    per-file progress. Pass `wait=true` for the synchronous result instead,
    which is convenient for scripting and tests.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {MAX_FILES_PER_REQUEST} files per request.",
        )

    # Read the uploads here either way: the request body is gone once this
    # handler returns, so a background worker cannot stream from it.
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "paper.pdf", await upload.read()))
        await upload.close()

    if not wait:
        job = jobs.create_job(
            "ingest",
            [name for name, _ in payloads],
            message=f"Indexing {len(payloads)} file(s)",
        )
        jobs.submit(job.id, lambda job_id: _ingest_in_background(job_id, payloads, user.id))
        return JSONResponse(status_code=202, content=job.to_dict())

    return _ingest_sync(session, payloads, user.id)


def _ingest_sync(
    session: Session, payloads: list[tuple[str, bytes]], owner_id: str
) -> UploadResponse:
    """Run the pipeline inline and return a definitive per-file result."""
    results: list[UploadResultItem] = []

    for filename, data in payloads:
        try:
            paper, created = store_upload(session, filename, data, owner_id)

            if not created and paper.status == PaperStatus.INDEXED:
                results.append(
                    UploadResultItem(
                        filename=filename,
                        paper_id=paper.id,
                        status="duplicate",
                        detail="This PDF is already indexed.",
                        paper=PaperSummary.from_model(paper),
                    )
                )
                continue

            paper = process_paper(session, paper.id)
            results.append(
                UploadResultItem(
                    filename=filename,
                    paper_id=paper.id,
                    status="indexed",
                    detail=f"Indexed {paper.chunk_count} chunks from {paper.page_count} pages.",
                    paper=PaperSummary.from_model(paper),
                )
            )
        except IngestionError as exc:
            results.append(UploadResultItem(filename=filename, status="failed", detail=str(exc)))
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Upload failed for %s", filename)
            results.append(UploadResultItem(filename=filename, status="failed", detail=str(exc)))

    return UploadResponse(
        uploaded=len(results),
        indexed=sum(1 for r in results if r.status == "indexed"),
        duplicates=sum(1 for r in results if r.status == "duplicate"),
        failed=sum(1 for r in results if r.status == "failed"),
        results=results,
    )


def _ingest_in_background(
    job_id: str, payloads: list[tuple[str, bytes]], owner_id: str
) -> None:
    """Worker body: index each file, reporting progress as it goes.

    Runs off the request thread, so it opens its own database session.
    """
    for index, (filename, data) in enumerate(payloads):
        try:
            jobs.update_item(job_id, index, state="running", stage="storing")
            with session_scope() as session:
                paper, created = store_upload(session, filename, data, owner_id)
                paper_id = paper.id

                if not created and paper.status == PaperStatus.INDEXED:
                    jobs.record_result(
                        job_id,
                        index,
                        ok=True,
                        state="duplicate",
                        detail="This PDF is already indexed.",
                        paper_id=paper_id,
                    )
                    continue

                jobs.update_item(job_id, index, stage="extracting", paper_id=paper_id)
                paper = process_paper(session, paper_id)
                jobs.record_result(
                    job_id,
                    index,
                    ok=True,
                    state="indexed",
                    detail=f"Indexed {paper.chunk_count} chunks from {paper.page_count} pages.",
                    paper_id=paper_id,
                )
        except IngestionError as exc:
            jobs.record_result(job_id, index, ok=False, state="failed", detail=str(exc))
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Background ingestion failed for %s", filename)
            jobs.record_result(job_id, index, ok=False, state="failed", detail=str(exc))


@router.get("", response_model=list[PaperSummary], summary="List papers")
def list_papers(
    status: PaperStatus | None = Query(default=None),
    q: str | None = Query(default=None, description="Filter by title/author substring"),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> list[PaperSummary]:
    statement = select(Paper).where(Paper.owner_id == user.id)
    if status:
        statement = statement.where(Paper.status == status)
    papers = list(session.exec(statement).all())

    if q:
        needle = q.lower()
        papers = [
            paper
            for paper in papers
            if needle in (paper.title or "").lower()
            or needle in (paper.filename or "").lower()
            or any(needle in author.lower() for author in (paper.authors or []))
        ]

    papers.sort(key=lambda p: p.created_at, reverse=True)
    return [PaperSummary.from_model(paper) for paper in papers]


@router.get("/{paper_id}", response_model=PaperSummary, summary="Get one paper")
def get_paper(
    paper_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> PaperSummary:
    return PaperSummary.from_model(owned_paper(session, user, paper_id))


@router.get("/{paper_id}/chunks", summary="List a paper's chunks")
def get_paper_chunks(
    paper_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict:
    owned_paper(session, user, paper_id)

    rows = list(
        session.exec(
            select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.index).offset(offset).limit(limit)
        ).all()
    )
    total = len(list(session.exec(select(Chunk.id).where(Chunk.paper_id == paper_id)).all()))
    return {
        "paper_id": paper_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "chunks": [
            {
                "id": row.id,
                "index": row.index,
                "text": row.text,
                "page_start": row.page_start,
                "page_end": row.page_end,
                "section": row.section,
                "token_estimate": row.token_estimate,
            }
            for row in rows
        ],
    }


@router.get("/{paper_id}/file", summary="Download the original PDF")
def download_paper(
    paper_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> FileResponse:
    paper = owned_paper(session, user, paper_id)
    path = Path(paper.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="The stored PDF file is missing.")
    return FileResponse(path, media_type="application/pdf", filename=paper.filename)


@router.post("/{paper_id}/reindex", response_model=PaperSummary, summary="Re-run ingestion")
def reindex_paper(
    paper_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> PaperSummary:
    owned_paper(session, user, paper_id)
    try:
        paper = process_paper(session, paper_id)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PaperSummary.from_model(paper)


@router.delete(
    "/{paper_id}",
    status_code=204,
    response_class=Response,
    summary="Delete a paper and its artefacts",
)
def remove_paper(
    paper_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    owned_paper(session, user, paper_id)
    if not delete_paper(session, paper_id):
        raise HTTPException(status_code=404, detail="Paper not found.")
    return Response(status_code=204)


@router.post("/search", response_model=SearchResponse, summary="Semantic search over your chunks")
def search_papers(
    payload: SearchRequest,
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> SearchResponse:
    started = time.perf_counter()

    # The vector store holds every account's chunks in one collection, so the
    # scope has to be passed explicitly. Searching with `paper_ids=None` would
    # return the whole installation's text.
    scope = owned_paper_ids(session, user, payload.paper_ids or None)
    if not scope:
        return SearchResponse(query=payload.query, hits=[], took_ms=0)

    hits = get_vector_store().search(payload.query, top_k=payload.top_k, paper_ids=scope)
    return SearchResponse(
        query=payload.query,
        hits=[SearchHit(**hit.to_dict()) for hit in hits],
        took_ms=int((time.perf_counter() - started) * 1000),
    )
