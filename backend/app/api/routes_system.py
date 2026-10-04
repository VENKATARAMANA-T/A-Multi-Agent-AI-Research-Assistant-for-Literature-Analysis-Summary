"""Health and corpus statistics — surfaces which backends are actually active."""

from __future__ import annotations

import logging
from collections import Counter

from fastapi import APIRouter, Depends
from sqlmodel import Session, func, select

from app import __version__
from app.config import settings
from app.api.deps import current_user, owned_paper_ids
from app.database import get_session
from app.models import AgentRun, Chunk, Paper, PaperStatus, Report, User
from app.schemas import HealthResponse, StatsResponse
from app.services import llm_cache, ocr, ocr_cache
from app.services.embeddings import get_embedder
from app.services.graph_store import get_graph_store
from app.services.llm import get_llm
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(session: Session = Depends(get_session)) -> HealthResponse:
    llm = get_llm()

    try:
        embedder = get_embedder()
        embeddings = {
            "backend": embedder.name,
            "dimension": embedder.dimension,
            "degraded": embedder.name.startswith("hashing-"),
        }
    except Exception as exc:  # pragma: no cover
        embeddings = {"backend": "unavailable", "error": str(exc), "degraded": True}

    try:
        store = get_vector_store()
        vector_store = {
            "status": "degraded" if store.embedding_backend_mismatch else "ok",
            "vectors": store.count(),
            "embedding_backend_mismatch": store.embedding_backend_mismatch,
        }
    except Exception as exc:  # pragma: no cover
        vector_store = {"status": "error", "error": str(exc)}

    try:
        graph = get_graph_store().stats()
    except Exception as exc:  # pragma: no cover
        graph = {"backend": "unavailable", "error": str(exc)}

    ocr_state = {**ocr.engine_status(), "cache": ocr_cache.summary()}

    papers = list(session.exec(select(Paper)).all())
    counts = Counter(paper.status for paper in papers)

    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        llm={
            "provider": "google-gemini",
            "model": llm.model,
            "fallback_models": getattr(llm, "fallback_models", []),
            "configured": llm.available,
            "max_concurrency": settings.llm_max_concurrency,
            "cache": llm_cache.summary(),
            "note": None if llm.available else "Set GOOGLE_API_KEY in backend/.env to enable the agents.",
        },
        embeddings=embeddings,
        vector_store=vector_store,
        graph=graph,
        ocr=ocr_state,
        papers={
            "total": sum(counts.values()),
            "indexed": counts.get(PaperStatus.INDEXED, 0),
            "failed": counts.get(PaperStatus.FAILED, 0),
            "ocr_assisted": sum(1 for paper in papers if (paper.text_source or "native") != "native"),
        },
    )


@router.get("/cache", summary="LLM cache statistics")
def cache_stats() -> dict:
    return llm_cache.summary()


@router.delete("/cache", status_code=200, summary="Clear the LLM cache")
def clear_cache() -> dict:
    removed = llm_cache.clear()
    return {"removed": removed}


@router.get("/stats", response_model=StatsResponse, summary="Corpus statistics")
def stats(
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> StatsResponse:
    """This account's corpus, not the installation's."""
    papers = list(session.exec(select(Paper).where(Paper.owner_id == user.id)).all())
    paper_ids = [paper.id for paper in papers]

    chunk_count = (
        session.exec(
            select(func.count()).select_from(Chunk).where(Chunk.paper_id.in_(paper_ids))
        ).one()
        if paper_ids
        else 0
    )
    run_count = session.exec(
        select(func.count()).select_from(AgentRun).where(AgentRun.owner_id == user.id)
    ).one()
    report_count = session.exec(
        select(func.count()).select_from(Report).where(Report.owner_id == user.id)
    ).one()

    try:
        # Counting the whole collection would report every account's
        # vectors, so this counts the chunks that belong to these papers.
        vectors = int(chunk_count)
    except Exception:  # pragma: no cover
        vectors = 0

    try:
        full = get_graph_store().stats()
        scoped = (
            get_graph_store().fetch(paper_ids=paper_ids, limit=10000)
            if paper_ids
            else {"nodes": [], "edges": []}
        )
        graph = {
            "backend": full.get("backend", "unknown"),
            "node_count": len(scoped["nodes"]),
            "edge_count": len(scoped["edges"]),
        }
    except Exception as exc:  # pragma: no cover
        graph = {"backend": "unavailable", "error": str(exc)}

    keywords = Counter()
    for paper in papers:
        for keyword in paper.keywords or []:
            keywords[keyword.strip().title()] += 1

    years = Counter(paper.year for paper in papers if paper.year)

    return StatsResponse(
        papers=len(papers),
        indexed=sum(1 for p in papers if p.status == PaperStatus.INDEXED),
        failed=sum(1 for p in papers if p.status == PaperStatus.FAILED),
        chunks=int(chunk_count),
        vectors=int(vectors),
        agent_runs=int(run_count),
        reports=int(report_count),
        graph=graph,
        top_keywords=[{"keyword": k, "count": v} for k, v in keywords.most_common(20)],
        papers_by_year=[{"year": year, "count": count} for year, count in sorted(years.items())],
    )
