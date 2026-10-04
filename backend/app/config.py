"""Application settings, loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -------------------------------------------------------------
    app_name: str = "ResearchCompass"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"

    # --- Storage -------------------------------------------------------------
    data_dir: Path = BACKEND_ROOT / "data"
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"
    chroma_dir: Path = BACKEND_ROOT / "data" / "chroma"
    report_dir: Path = BACKEND_ROOT / "data" / "reports"
    figure_dir: Path = BACKEND_ROOT / "data" / "figures"
    database_url: str = ""  # defaults to sqlite in data_dir

    # --- Ingestion -----------------------------------------------------------
    max_upload_mb: int = 50
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- OCR -----------------------------------------------------------------
    # Scanned PDFs have no text layer. OCR is applied per page, not per
    # document: papers are often born-digital with a photocopied appendix, and
    # OCRing a page whose native text is already perfect only degrades it.
    ocr_enabled: bool = True
    # "auto" uses RapidOCR when installed and falls back to Gemini vision if a
    # key is set. "rapidocr" / "gemini" / "none" pin one choice.
    ocr_engine: str = "auto"
    # 200 DPI recognises clean scans accurately; raise to 300 for poor ones.
    ocr_dpi: int = 200
    # A page with fewer than this many characters is a candidate for OCR.
    ocr_min_chars_per_page: int = 120
    # ...but only if images cover this much of it. Otherwise it is simply blank.
    ocr_min_image_coverage: float = 0.25
    # Guard against a 500-page scan silently costing an hour of CPU.
    ocr_max_pages_per_document: int = 40
    # Recognised lines below this confidence are dropped as noise.
    ocr_min_confidence: float = 0.5
    # How far the detector expands each text polygon. The engine default (1.6)
    # is tuned for signage and drops closely-spaced lines of body text —
    # measured: one line in four silently lost on a 9pt, 18pt-leading page.
    # 2.0 recovers them without merging adjacent columns.
    ocr_unclip_ratio: float = 2.0
    ocr_cache_enabled: bool = True

    # --- Figures, charts and tables ------------------------------------------
    # Extraction runs during ingestion and costs nothing. *Analysis* is one
    # vision request per figure, so it is always an explicit, opt-in action —
    # a ten-figure paper would otherwise spend half a free-tier day in one go.
    figures_enabled: bool = True
    figure_dpi: int = 150
    figure_max_per_document: int = 60
    # Ceiling on a single analysis request, as a guard against a runaway bill.
    figure_max_analysis_batch: int = 40

    # --- Discovery (OpenAlex) ------------------------------------------------
    # OpenAlex needs no API key. Supplying a contact address is optional but
    # puts requests in their faster pool.
    discovery_enabled: bool = True
    openalex_contact_email: str = ""

    # --- Conversations -------------------------------------------------------
    # How many previous turns to replay into a follow-up question. Every turn
    # costs prompt tokens, so this is bounded rather than unlimited.
    conversation_memory_turns: int = 6

    # --- Embeddings ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # "auto" prefers the ONNX (fastembed) path and falls back to PyTorch.
    # Pin to "sentence-transformers" or "fastembed" to force one.
    embedding_backend: str = "auto"
    # When true, never attempt to load sentence-transformers (useful for CI and
    # for machines without the model cached). A deterministic hashing embedder
    # is used instead, so the whole pipeline still runs end to end.
    embedding_offline_fallback: bool = False

    # --- Vector store --------------------------------------------------------
    chroma_collection: str = "researchcompass"
    retrieval_top_k: int = 8

    # --- LLM (Gemini) --------------------------------------------------------
    google_api_key: str = ""
    # "gemini-flash-latest" tracks the current Flash model and is the alias with
    # the broadest free-tier availability; pin a dated id for reproducibility.
    gemini_model: str = "gemini-flash-latest"
    # Models to try when the primary one cannot serve the request. Google
    # returns UNAVAILABLE ("experiencing high demand") under load and
    # RESOURCE_EXHAUSTED when the day's free-tier allowance is spent, and
    # neither clears by retrying the same model — the second is guaranteed not
    # to, since the daily cap is per model. Each model carries its own quota, so
    # a second one is the difference between a working afternoon and a dead one.
    # Comma-separated, tried in order. Empty disables fallback.
    gemini_fallback_models: str = "gemini-flash-lite-latest"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 8192
    # Per-request deadline. The google-genai SDK has no client-side timeout, so
    # this is enforced by the client itself; without it a stalled call hangs the
    # request indefinitely.
    llm_timeout_seconds: int = 90
    # Ceiling on one call's whole retry loop, so repeated timeouts cannot hold
    # an HTTP request open for many minutes.
    llm_total_retry_seconds: int = 180

    # Cache identical requests. On the free tier the binding limit is requests
    # per day, so this is what makes repeated runs and demos possible at all.
    llm_cache_enabled: bool = True
    llm_cache_ttl_days: int = 30

    # How many agent calls may be in flight at once. Kept low by default: the
    # free tier also caps requests per minute, and a wide fan-out trips it.
    llm_max_concurrency: int = 3

    # Requests per minute. A concurrency limit bounds how many calls run at
    # once but not how fast they are issued, and the free tier's cap is a
    # *rate* — five per minute, at which a five-paper fan-out fails entirely
    # even with only three in flight. 0 disables throttling; raise it or set 0
    # once billing is enabled.
    llm_requests_per_minute: int = 5

    # --- Neo4j ---------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "researchcompass"
    neo4j_database: str = "neo4j"
    neo4j_enabled: bool = True

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'researchcompass.db').as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.google_api_key)

    @property
    def gemini_model_chain(self) -> list[str]:
        """The primary model followed by its fallbacks, de-duplicated."""
        chain = [self.gemini_model.strip()]
        for name in self.gemini_fallback_models.split(","):
            name = name.strip()
            if name and name not in chain:
                chain.append(name)
        return [name for name in chain if name]

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.upload_dir,
            self.chroma_dir,
            self.report_dir,
            self.figure_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
