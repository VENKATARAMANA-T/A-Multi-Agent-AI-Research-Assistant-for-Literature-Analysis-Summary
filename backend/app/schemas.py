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
    figure_count: int = 0
    char_count: int = 0
    size_bytes: int = 0
    status: PaperStatus
    error: Optional[str] = None
    sections: list[str] = Field(default_factory=list)
    text_source: str = "native"
    ocr_pages: list[int] = Field(default_factory=list)
    ocr_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None
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
            figure_count=paper.figure_count or 0,
            char_count=paper.char_count,
            size_bytes=paper.size_bytes,
            status=paper.status,
            error=paper.error,
            sections=sorted((paper.sections or {}).keys()),
            text_source=paper.text_source or "native",
            ocr_pages=paper.ocr_pages or [],
            ocr_confidence=paper.ocr_confidence,
            ocr_engine=paper.ocr_engine,
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


class FigureSummary(BaseModel):
    id: str
    paper_id: str
    paper_title: Optional[str] = None
    kind: str
    label: str
    caption: str
    page: int
    status: str
    detector: str = "caption"
    has_image: bool = False

    table_markdown: Optional[str] = None
    table_rows: int = 0
    table_cols: int = 0

    chart_type: Optional[str] = None
    description: Optional[str] = None
    takeaway: Optional[str] = None
    axes: dict[str, Any] = Field(default_factory=dict)
    series: list[Any] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    analysis_error: Optional[str] = None

    @classmethod
    def from_model(cls, figure: Any, paper_title: str | None = None) -> "FigureSummary":
        from pathlib import Path

        return cls(
            id=figure.id,
            paper_id=figure.paper_id,
            paper_title=paper_title,
            kind=figure.kind,
            label=figure.label,
            caption=figure.caption,
            page=figure.page,
            status=figure.status.value if hasattr(figure.status, "value") else str(figure.status),
            detector=figure.detector,
            has_image=bool(figure.image_path and Path(figure.image_path).exists()),
            table_markdown=figure.table_markdown,
            table_rows=figure.table_rows,
            table_cols=figure.table_cols,
            chart_type=figure.chart_type,
            description=figure.description,
            takeaway=figure.takeaway,
            axes=figure.axes or {},
            series=figure.series or [],
            findings=figure.findings or [],
            entities=figure.entities or {},
            analysis_error=figure.analysis_error,
        )


class FigureCostEstimate(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    pending: int = 0
    requests_required: int = 0
    capped_at: int = 0
    already_readable: int = 0


class FigureAnalysisRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    limit: Optional[int] = Field(default=None, ge=1, le=100)


class FigureAnalysisResponse(BaseModel):
    analysed: int
    failed: int
    remaining: int
    llm_calls: int
    duration_ms: int
    errors: list[str] = Field(default_factory=list)
    figures: list[FigureSummary] = Field(default_factory=list)
    detail: Optional[str] = None


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
    source: str = "native"
    kind: str = "text"
    figure_id: Optional[str] = None
    label: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    took_ms: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=30)
    # Continue an existing thread, so a follow-up can say "why?" and be understood.
    conversation_id: Optional[str] = None
    start_conversation: bool = False
    # "hybrid" fuses passages with graph relationships. "vector" is passages
    # only; "graph" is relationships only, which answers questions no single
    # passage contains.
    mode: Literal["hybrid", "vector", "graph"] = "hybrid"


class PaperIdsRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)


class SummarizeRequest(PaperIdsRequest):
    scope: Literal["single", "multi"] = "multi"


class GapRequest(PaperIdsRequest):
    focus: Optional[str] = Field(default=None, max_length=500)


class HighlightResponse(BaseModel):
    page: int
    rects: list[list[float]] = Field(default_factory=list)
    matched_phrase: Optional[str] = None
    found: bool = False
    page_width: float = 0
    page_height: float = 0


class ExplainRequest(BaseModel):
    text: str = Field(min_length=10, max_length=8000)
    level: Literal["simple", "standard", "technical"] = "standard"
    paper_id: Optional[str] = None
    surrounding: Optional[str] = Field(default=None, max_length=8000)


class ExplainResponse(BaseModel):
    explanation: Optional[str] = None
    terms: list[dict[str, Any]] = Field(default_factory=list)
    background: Optional[str] = None
    why_it_matters: Optional[str] = None
    caveats: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: str = "completed"


class CitationResponse(BaseModel):
    style: str
    text: str
    count: int = 1


class MatrixColumn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=400)
    type: Literal["text", "number", "boolean", "list"] = "text"


class MatrixRequest(BaseModel):
    columns: list[MatrixColumn]
    paper_ids: list[str] = Field(default_factory=list)
    name: Optional[str] = Field(default=None, max_length=200)


class MatrixResponse(BaseModel):
    id: Optional[str] = None
    name: str = ""
    columns: list[dict[str, Any]] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    duration_ms: int = 0
    status: str = "completed"


class MatrixRunSummary(BaseModel):
    id: str
    name: str
    paper_count: int
    column_count: int
    created_at: datetime


class LbdRequest(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    target: Optional[str] = Field(default=None, max_length=200)
    # "strict" is Swanson's criterion: A and C must share no paper at all.
    # "unstated" relaxes that to any pair with no direct link, which is what a
    # small corpus can actually produce.
    mode: Literal["strict", "unstated"] = "strict"
    limit: int = Field(default=15, ge=1, le=50)
    min_support: int = Field(default=1, ge=1, le=10)
    assess: bool = False
    assess_limit: int = Field(default=5, ge=1, le=20)


class LbdResponse(BaseModel):
    source: str
    mode: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    saved_ids: list[str] = Field(default_factory=list)


class ClosedLbdResponse(BaseModel):
    source: str
    target: str
    chains: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HypothesisSummary(BaseModel):
    id: str
    source_term: str
    target_term: str
    target_type: str
    mode: str
    support: int
    score: float
    verdict: Optional[str] = None
    statement: Optional[str] = None
    reasoning: Optional[str] = None
    mechanism: Optional[str] = None
    proposed_test: Optional[str] = None
    novelty: Optional[str] = None
    confidence: Optional[str] = None
    why_not: Optional[str] = None
    chains: list[Any] = Field(default_factory=list)
    starred: bool = False
    created_at: datetime


class DiscoveryResponse(BaseModel):
    query: str
    source: str = "openalex"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    excluded_known: bool = True


class AgentRunResponse(BaseModel):
    run_id: Optional[str] = None
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
    graph_facts: list[dict[str, Any]] = Field(default_factory=list)
    graph_matches: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    duration_ms: int = 0
    conversation_id: Optional[str] = None


class VerifyRequest(BaseModel):
    """Verify either a stored agent run, or a piece of text pasted in directly."""

    agent_run_id: Optional[str] = None
    text: Optional[str] = Field(default=None, max_length=20_000)
    paper_ids: list[str] = Field(default_factory=list)
    source: str = Field(default="text", max_length=32)
    subject: Optional[str] = Field(default=None, max_length=500)
    evidence_per_claim: int = Field(default=4, ge=1, le=10)
    save: bool = True


class VerifyResponse(BaseModel):
    id: Optional[str] = None
    source: str = "text"
    subject: Optional[str] = None
    status: str = "completed"
    score: float = 0.0
    checked: int = 0
    counts: dict[str, Any] = Field(default_factory=dict)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    problems: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    duration_ms: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)


class VerificationSummary(BaseModel):
    id: str
    agent_run_id: Optional[str] = None
    source: str
    subject: Optional[str] = None
    score: float
    checked: int
    counts: dict[str, Any] = Field(default_factory=dict)
    status: str
    llm_calls: int = 0
    created_at: datetime


class VerificationDetail(VerificationSummary):
    text: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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
    ocr: dict[str, Any] = Field(default_factory=dict)
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
