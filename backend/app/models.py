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
    created_at: datetime = Field(default_factory=utcnow)


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
