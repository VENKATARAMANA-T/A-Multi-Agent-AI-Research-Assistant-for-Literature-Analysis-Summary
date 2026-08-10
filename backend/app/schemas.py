"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models import Paper, PaperStatus


class PaperSummary(BaseModel):
    id: str
    filename: str
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    page_count: int = 0
    chunk_count: int = 0
    char_count: int = 0
    size_bytes: int = 0
    status: PaperStatus
    error: Optional[str] = None
    sections: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def from_model(cls, paper: Paper) -> "PaperSummary":
        return cls(
            id=paper.id,
            filename=paper.filename,
            title=paper.title,
            authors=paper.authors or [],
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            abstract=paper.abstract,
            keywords=paper.keywords or [],
            page_count=paper.page_count,
            chunk_count=paper.chunk_count,
            char_count=paper.char_count,
            size_bytes=paper.size_bytes,
            status=paper.status,
            error=paper.error,
            sections=sorted((paper.sections or {}).keys()),
            created_at=paper.created_at,
        )


class UploadResultItem(BaseModel):
    filename: str
    paper_id: Optional[str] = None
    status: Literal["indexed", "duplicate", "failed"]
    detail: Optional[str] = None
    paper: Optional[PaperSummary] = None


class UploadResponse(BaseModel):
    uploaded: int
    indexed: int
    duplicates: int
    failed: int
    results: list[UploadResultItem]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: Optional[str] = None
    text: str
    score: float
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    took_ms: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=30)


class PaperIdsRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)


class SummarizeRequest(PaperIdsRequest):
    scope: Literal["single", "multi"] = "multi"


class GapRequest(PaperIdsRequest):
    focus: Optional[str] = Field(default=None, max_length=500)


class AgentRunResponse(BaseModel):
    intent: str
    status: str
    question: Optional[str] = None
    paper_ids: list[str] = Field(default_factory=list)
    answer: Optional[dict[str, Any]] = None
    summary: Optional[dict[str, Any]] = None
    extraction: Optional[dict[str, Any]] = None
    gaps: Optional[dict[str, Any]] = None
    graph: Optional[dict[str, Any]] = None
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    duration_ms: int = 0


class GraphNode(BaseModel):
    id: str
    name: Optional[str] = None
    type: str = "Concept"
    description: str = ""
    papers: list[str] = Field(default_factory=list)
    paper_id: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"
    evidence: str = ""
    papers: list[str] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    backend: str
    stats: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    title: Optional[str] = Field(default=None, max_length=300)
    focus: Optional[str] = Field(default=None, max_length=500)
    include_narrative: bool = True
    include_graph: bool = True


class ReportSummary(BaseModel):
    id: str
    title: str
    kind: str
    paper_ids: list[str]
    has_pdf: bool
    agent_status: str = "completed"
    agent_errors: list[str] = Field(default_factory=list)
    created_at: datetime


class ReportDetail(ReportSummary):
    markdown: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm: dict[str, Any]
    embeddings: dict[str, Any]
    vector_store: dict[str, Any]
    graph: dict[str, Any]
    papers: dict[str, Any]


class StatsResponse(BaseModel):
    papers: int
    indexed: int
    failed: int
    chunks: int
    vectors: int
    agent_runs: int
    reports: int
    graph: dict[str, Any]
    top_keywords: list[dict[str, Any]] = Field(default_factory=list)
    papers_by_year: list[dict[str, Any]] = Field(default_factory=list)
