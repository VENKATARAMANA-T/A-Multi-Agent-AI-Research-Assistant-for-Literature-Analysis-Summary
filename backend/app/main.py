"""ResearchCompass FastAPI application."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import api_router
from app.config import settings
from app.database import init_db

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("researchcompass")

# Chroma's bundled posthog client logs a spurious ERROR on every call even with
# telemetry disabled. Silence it so real errors stay visible.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

DESCRIPTION = """
**ResearchCompass** — a multi-agent AI research assistant for literature analysis.

Upload research papers and the system extracts their text, chunks and embeds it,
indexes it in a vector store, and turns a set of specialised agents loose on it:

* **Retrieval Agent** — semantic search with multi-query fusion
* **Summarization Agent** — single-paper and cross-paper synthesis
* **Information Extraction Agent** — datasets, methods, metrics, tasks
* **Question Answering Agent** — grounded RAG answers with citations
* **Research Gap Agent** — limitations and future research directions
* **Knowledge Graph Agent** — typed entities and relations for visualisation

The agents are orchestrated as a stateful LangGraph workflow and powered by Gemini.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_db()
    logger.info("ResearchCompass %s starting (environment=%s)", __version__, settings.environment)

    if not settings.llm_enabled:
        logger.warning(
            "GOOGLE_API_KEY is not set — ingestion and search work, but the agents "
            "will return an 'LLM unavailable' error until you configure a key."
        )

    # Ownership arrived after the data did: rows created before accounts
    # existed have no owner, and a NULL owner matches nobody's query, so the
    # corpus would disappear from the UI without this.
    try:
        from app.database import session_scope
        from app.services import bootstrap

        with session_scope() as session:
            adopted = bootstrap.run(session)
        if adopted:
            logger.info("Adopted pre-existing data: %s", adopted)
    except Exception:  # pragma: no cover - never block startup on bookkeeping
        logger.exception("Could not run the ownership bootstrap")

    # A job left RUNNING belongs to a process that no longer exists.
    from app.services import jobs

    reclaimed = jobs.reclaim_stale_jobs()
    if reclaimed:
        logger.info("Marked %d interrupted job(s) as failed", reclaimed)

    # Warm the vector store *and* the embedding model. Loading the model lazily
    # meant the first upload of every session paid a multi-second penalty that
    # looked like slow ingestion.
    try:
        from app.services.embeddings import get_embedder
        from app.services.vector_store import get_vector_store

        started = time.perf_counter()
        embedder = get_embedder()
        get_vector_store()
        logger.info(
            "Warm start: embedder=%s (dim %d), vectors=%d, %.1fs",
            embedder.name,
            embedder.dimension,
            get_vector_store().count(),
            time.perf_counter() - started,
        )
    except Exception:
        logger.warning("Warm start failed; falling back to lazy loading", exc_info=True)

    try:
        from app.services import llm_cache

        purged = llm_cache.purge_expired()
        if purged:
            logger.info("Purged %d expired LLM cache entries", purged)
    except Exception:
        logger.warning("Could not purge the LLM cache", exc_info=True)

    yield

    jobs.shutdown()
    logger.info("ResearchCompass shutting down")


app = FastAPI(
    title="ResearchCompass API",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent payloads carry full chunk text and long reports; compressing them cuts
# transfer time noticeably. The minimum size keeps small JSON uncompressed.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - started) * 1000))
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


app.include_router(api_router)


@app.get("/", tags=["system"], summary="API root")
def root() -> dict:
    return {
        "name": "ResearchCompass API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }
