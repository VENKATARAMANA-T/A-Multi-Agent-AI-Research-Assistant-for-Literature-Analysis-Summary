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
    database_url: str = ""  # defaults to sqlite in data_dir

    # --- Ingestion -----------------------------------------------------------
    max_upload_mb: int = 50
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- Embeddings ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
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
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 8192
    llm_timeout_seconds: int = 120

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

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.chroma_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
