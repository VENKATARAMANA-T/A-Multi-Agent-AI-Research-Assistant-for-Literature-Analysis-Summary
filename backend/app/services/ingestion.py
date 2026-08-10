"""Ingestion pipeline: upload -> extract -> chunk -> embed -> index (steps 1-5)."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from sqlmodel import Session, delete, select

from app.config import settings
from app.models import Chunk, ExtractionRecord, Paper, PaperStatus, SummaryRecord, utcnow
from app.services.chunking import chunk_document
from app.services.pdf_extract import extract_pdf
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"


class IngestionError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "paper.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:120]


def validate_pdf(data: bytes, filename: str) -> None:
    if not data:
        raise IngestionError("Uploaded file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise IngestionError(f"'{filename}' exceeds the {settings.max_upload_mb} MB upload limit.")
    if not data.startswith(PDF_MAGIC):
        raise IngestionError(f"'{filename}' is not a valid PDF (missing %PDF header).")


def store_upload(session: Session, filename: str, data: bytes) -> tuple[Paper, bool]:
    """Persist the bytes and create (or return the existing) Paper row.

    Returns (paper, created). Re-uploading identical bytes is a no-op.
    """
    validate_pdf(data, filename)
    content_hash = sha256_bytes(data)

    existing = session.exec(select(Paper).where(Paper.content_hash == content_hash)).first()
    if existing:
        logger.info("Duplicate upload ignored: %s matches paper %s", filename, existing.id)
        return existing, False

    paper = Paper(
        filename=safe_filename(filename),
        file_path="",
        content_hash=content_hash,
        size_bytes=len(data),
        status=PaperStatus.UPLOADED,
    )
    session.add(paper)
    session.commit()
    session.refresh(paper)

    destination = Path(settings.upload_dir) / f"{paper.id}.pdf"
    destination.write_bytes(data)
    paper.file_path = str(destination)
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper, True


def process_paper(session: Session, paper_id: str) -> Paper:
    """Run extraction, chunking, embedding and indexing for one paper."""
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise IngestionError(f"Unknown paper {paper_id}")

    try:
        paper.status = PaperStatus.EXTRACTING
        paper.error = None
        paper.updated_at = utcnow()
        session.add(paper)
        session.commit()

        document = extract_pdf(paper.file_path)
        if not document.text.strip():
            raise IngestionError(
                "No text layer found in this PDF. Scanned documents need OCR before ingestion."
            )

        paper.title = document.title or paper.filename
        paper.authors = document.authors
        paper.abstract = document.abstract
        paper.year = document.year
        paper.doi = document.doi
        paper.venue = document.venue
        paper.keywords = document.keywords
        paper.page_count = document.page_count
        paper.char_count = document.char_count
        paper.full_text = document.text
        paper.sections = document.sections

        paper.status = PaperStatus.CHUNKING
        session.add(paper)
        session.commit()

        chunks = chunk_document(document)
        if not chunks:
            raise IngestionError("Chunking produced no text segments.")

        # Replace any previous chunks for this paper (re-ingest is idempotent).
        session.exec(delete(Chunk).where(Chunk.paper_id == paper.id))
        session.commit()

        rows = [
            Chunk(
                paper_id=paper.id,
                index=chunk.index,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                token_estimate=chunk.token_estimate,
            )
            for chunk in chunks
        ]
        session.add_all(rows)
        session.commit()
        for row in rows:
            session.refresh(row)

        paper.status = PaperStatus.EMBEDDING
        session.add(paper)
        session.commit()

        store = get_vector_store()
        store.delete_paper(paper.id)
        store.add_chunks(paper.id, chunks, [row.id for row in rows], paper_title=paper.title)

        paper.chunk_count = len(chunks)
        paper.status = PaperStatus.INDEXED
        paper.updated_at = utcnow()
        session.add(paper)
        session.commit()
        session.refresh(paper)

        logger.info("Indexed paper %s ('%s') with %d chunks", paper.id, paper.title, len(chunks))
        return paper

    except Exception as exc:
        logger.exception("Ingestion failed for paper %s", paper_id)
        session.rollback()
        paper = session.get(Paper, paper_id)
        if paper is not None:
            paper.status = PaperStatus.FAILED
            paper.error = str(exc)
            paper.updated_at = utcnow()
            session.add(paper)
            session.commit()
            session.refresh(paper)
        raise


def delete_paper(session: Session, paper_id: str) -> bool:
    """Remove a paper and every derived artefact (rows, vectors, file, graph)."""
    paper = session.get(Paper, paper_id)
    if paper is None:
        return False

    session.exec(delete(Chunk).where(Chunk.paper_id == paper_id))
    session.exec(delete(ExtractionRecord).where(ExtractionRecord.paper_id == paper_id))
    session.exec(delete(SummaryRecord).where(SummaryRecord.paper_id == paper_id))
    session.commit()

    try:
        get_vector_store().delete_paper(paper_id)
    except Exception:  # pragma: no cover - store may be unavailable
        logger.warning("Could not delete vectors for paper %s", paper_id, exc_info=True)

    try:
        from app.services.graph_store import get_graph_store

        get_graph_store().delete_paper(paper_id)
    except Exception:  # pragma: no cover
        logger.warning("Could not delete graph nodes for paper %s", paper_id, exc_info=True)

    if paper.file_path:
        Path(paper.file_path).unlink(missing_ok=True)

    session.delete(paper)
    session.commit()
    return True
