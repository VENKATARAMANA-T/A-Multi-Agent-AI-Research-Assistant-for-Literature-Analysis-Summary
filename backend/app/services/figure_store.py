"""Persistence and indexing for extracted figures.

Extraction runs during ingestion and is free. Vision analysis is a separate,
costed step. Both feed the same record, and whatever is known about a figure —
caption alone, or caption plus a full reading of the chart — is embedded into
the vector store so figures are findable by meaning alongside the text.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, delete, select

from app.config import settings
from app.models import Figure, FigureStatus, Paper
from app.services.figures import ExtractedFigure, extract_figures

logger = logging.getLogger(__name__)

# Figures are indexed in the same collection as text, tagged so a search can
# tell an excerpt from a chart.
FIGURE_ID_PREFIX = "figure:"


def figure_dir(paper_id: str) -> Path:
    path = Path(settings.figure_dir) / paper_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_figures(
    session: Session,
    paper: Paper,
    extracted: list[ExtractedFigure],
) -> list[Figure]:
    """Replace this paper's figures with a freshly extracted set."""
    clear_figures(session, paper.id, remove_files=True)

    rows: list[Figure] = []
    for item in extracted:
        row = Figure(
            paper_id=paper.id,
            kind=item.kind,
            label=item.label,
            caption=item.caption,
            page=item.page,
            bbox=list(item.bbox),
            detector=item.detector,
            table_markdown=item.table_markdown,
            table_rows=item.table_rows,
            table_cols=item.table_cols,
            # A table whose contents we already read needs no vision call.
            status=FigureStatus.SKIPPED if item.table_markdown else FigureStatus.PENDING,
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        if item.image_png:
            destination = figure_dir(paper.id) / f"{row.id}.png"
            destination.write_bytes(item.image_png)
            row.image_path = str(destination)
            session.add(row)
            session.commit()
            session.refresh(row)

        rows.append(row)

    logger.info("Stored %d figure(s) for paper %s", len(rows), paper.id)
    return rows


def clear_figures(session: Session, paper_id: str, remove_files: bool = True) -> int:
    existing = list(session.exec(select(Figure).where(Figure.paper_id == paper_id)).all())
    if remove_files:
        for row in existing:
            if row.image_path:
                Path(row.image_path).unlink(missing_ok=True)
    session.exec(delete(Figure).where(Figure.paper_id == paper_id))
    session.commit()

    try:
        get_index().delete_paper_figures(paper_id)
    except Exception:  # pragma: no cover
        logger.warning("Could not unindex figures for %s", paper_id, exc_info=True)

    return len(existing)


def apply_analysis(session: Session, figure: Figure, analysis: dict | None, error: str | None = None) -> Figure:
    """Write a vision result (or a failure) onto the figure row."""
    if analysis is None:
        figure.status = FigureStatus.FAILED
        figure.analysis_error = error or "Analysis produced no result."
    else:
        figure.chart_type = analysis.get("chart_type")
        figure.description = analysis.get("description")
        figure.takeaway = analysis.get("takeaway")
        figure.axes = analysis.get("axes") or {}
        figure.series = analysis.get("series") or []
        figure.findings = analysis.get("findings") or []
        figure.entities = analysis.get("entities") or {}
        figure.analysis_error = None
        figure.status = FigureStatus.ANALYSED
    figure.analysed_at = datetime.now(timezone.utc)

    session.add(figure)
    session.commit()
    session.refresh(figure)
    return figure


def pending_figures(session: Session, paper_ids: list[str] | None = None) -> list[Figure]:
    """Figures that a vision pass would actually act on."""
    statement = select(Figure).where(Figure.status == FigureStatus.PENDING)
    if paper_ids:
        statement = statement.where(Figure.paper_id.in_(paper_ids))  # type: ignore[attr-defined]
    rows = list(session.exec(statement).all())
    # Without an image there is nothing to look at.
    return [row for row in rows if row.image_path and Path(row.image_path).exists()]


def extract_and_store(session: Session, paper: Paper) -> list[Figure]:
    """Full extraction pass for one paper. Never raises — figures are a bonus."""
    if not settings.figures_enabled:
        return []
    try:
        extracted = extract_figures(
            paper.file_path,
            dpi=settings.figure_dpi,
            max_figures=settings.figure_max_per_document,
        )
    except Exception:
        logger.warning("Figure extraction failed for paper %s", paper.id, exc_info=True)
        return []

    rows = store_figures(session, paper, extracted)
    index_figures(rows, paper.title)
    return rows


# --- vector indexing ---------------------------------------------------------


def index_figures(figures: list[Figure], paper_title: str | None) -> int:
    """Embed figures so semantic search can return them alongside text."""
    indexed = [figure for figure in figures if figure.search_text.strip()]
    if not indexed:
        return 0
    try:
        return get_index().add_figures(indexed, paper_title)
    except Exception:  # pragma: no cover
        logger.warning("Could not index figures", exc_info=True)
        return 0


def get_index():
    from app.services.vector_store import get_vector_store

    return get_vector_store()
