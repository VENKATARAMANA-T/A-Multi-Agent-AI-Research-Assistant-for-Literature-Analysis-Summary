"""SQLModel tables — relational metadata for papers, chunks, agent runs, reports."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, Text
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class Paper(SQLModel, table=True):
    __tablename__ = "papers"

    id: str = Field(default_factory=_uuid, primary_key=True)
    filename: str
    file_path: str
    content_hash: str = Field(index=True)
    size_bytes: int = 0

    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    abstract: Optional[str] = Field(default=None, sa_column=Column(Text))
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    page_count: int = 0
    char_count: int = 0
    chunk_count: int = 0
    figure_count: int = 0

    # --- OCR provenance ------------------------------------------------------
    # "native" (text layer), "ocr" (every page recognised) or "mixed".
    text_source: str = Field(default="native")
    ocr_pages: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    ocr_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None

    status: PaperStatus = Field(default=PaperStatus.UPLOADED, index=True)
    error: Optional[str] = Field(default=None, sa_column=Column(Text))

    full_text: Optional[str] = Field(default=None, sa_column=Column(Text))
    sections: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: str = Field(default_factory=_uuid, primary_key=True)
    paper_id: str = Field(index=True, foreign_key="papers.id")
    index: int = 0
    text: str = Field(sa_column=Column(Text))
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    token_estimate: int = 0
    # "native" or "ocr" — lets a search result flag text that may be imperfect.
    source: str = Field(default="native")
    created_at: datetime = Field(default_factory=utcnow)


class FigureStatus(str, Enum):
    PENDING = "pending"      # extracted, not yet analysed by the vision model
    ANALYSED = "analysed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Figure(SQLModel, table=True):
    """A figure, chart, diagram or table lifted out of a paper.

    Extraction is free and happens during ingestion. Analysis costs one vision
    request per figure, so it is a separate, opt-in step — see FigureStatus.
    """

    __tablename__ = "figures"

    id: str = Field(default_factory=_uuid, primary_key=True)
    paper_id: str = Field(index=True, foreign_key="papers.id")

    kind: str = Field(default="figure", index=True)  # figure | table | chart | algorithm
    label: str = ""                                   # "Figure 3"
    caption: str = Field(default="", sa_column=Column(Text))
    page: int = 0
    bbox: list[float] = Field(default_factory=list, sa_column=Column(JSON))
    image_path: Optional[str] = None
    detector: str = "caption"

    # Tables carry their content directly; no vision call needed to read them.
    table_markdown: Optional[str] = Field(default=None, sa_column=Column(Text))
    table_rows: int = 0
    table_cols: int = 0

    # --- vision analysis -----------------------------------------------------
    status: FigureStatus = Field(default=FigureStatus.PENDING, index=True)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    chart_type: Optional[str] = None
    axes: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    series: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    findings: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    takeaway: Optional[str] = Field(default=None, sa_column=Column(Text))
    entities: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    analysis_error: Optional[str] = Field(default=None, sa_column=Column(Text))
    analysed_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=utcnow)

    @property
    def search_text(self) -> str:
        """What gets embedded so this figure is findable by meaning."""
        parts = [self.label, self.caption]
        if self.description:
            parts.append(self.description)
        if self.findings:
            parts.extend(self.findings)
        if self.takeaway:
            parts.append(self.takeaway)
        if self.table_markdown:
            parts.append(self.table_markdown)
        return "\n".join(part for part in parts if part)


class ExtractionRecord(SQLModel, table=True):
    """Structured entities pulled out of a paper by the extraction agent."""

    __tablename__ = "extractions"

    id: str = Field(default_factory=_uuid, primary_key=True)
    paper_id: str = Field(index=True, foreign_key="papers.id")
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class SummaryRecord(SQLModel, table=True):
    __tablename__ = "summaries"

    id: str = Field(default_factory=_uuid, primary_key=True)
    paper_id: Optional[str] = Field(default=None, index=True)
    scope: str = "single"  # single | multi
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class Conversation(SQLModel, table=True):
    """A multi-turn question-and-answer thread over a set of papers.

    Without this, every question is asked cold and a follow-up like "why?" has
    nothing to refer back to.
    """

    __tablename__ = "conversations"

    id: str = Field(default_factory=_uuid, primary_key=True)
    title: str = ""
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # [{role: "user"|"assistant", content: str, sources: [...], at: iso}]
    messages: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def turn_count(self) -> int:
        return sum(1 for message in (self.messages or []) if message.get("role") == "user")


class Hypothesis(SQLModel, table=True):
    """A candidate connection found by literature-based discovery.

    Saved so a promising lead survives the session it was found in — the point
    of the technique is to surface things worth following up later.
    """

    __tablename__ = "hypotheses"

    id: str = Field(default_factory=_uuid, primary_key=True)

    source_term: str = ""
    target_term: str = ""
    target_type: str = "Concept"
    mode: str = "strict"          # strict (disjoint literatures) | unstated

    support: int = 0
    score: float = 0.0
    chains: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # --- assessment ----------------------------------------------------------
    verdict: Optional[str] = Field(default=None, index=True)
    statement: Optional[str] = Field(default=None, sa_column=Column(Text))
    reasoning: Optional[str] = Field(default=None, sa_column=Column(Text))
    mechanism: Optional[str] = Field(default=None, sa_column=Column(Text))
    proposed_test: Optional[str] = Field(default=None, sa_column=Column(Text))
    novelty: Optional[str] = None
    confidence: Optional[str] = None
    why_not: Optional[str] = Field(default=None, sa_column=Column(Text))

    starred: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class MatrixRun(SQLModel, table=True):
    """A saved custom-column comparison across papers."""

    __tablename__ = "matrix_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str = ""
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    columns: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    rows: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    errors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    llm_calls: int = 0
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class AgentRun(SQLModel, table=True):
    """Audit trail of every LangGraph workflow execution."""

    __tablename__ = "agent_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    intent: str = Field(index=True)
    question: Optional[str] = Field(default=None, sa_column=Column(Text))
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = "completed"
    trace: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class Report(SQLModel, table=True):
    __tablename__ = "reports"

    id: str = Field(default_factory=_uuid, primary_key=True)
    title: str
    kind: str = "literature_review"
    paper_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    markdown: str = Field(default="", sa_column=Column(Text))
    pdf_path: Optional[str] = None

    # Outcome of the agent run behind this report, so a report with empty
    # sections can explain itself instead of looking silently broken.
    agent_status: str = "completed"
    agent_errors: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=utcnow)
