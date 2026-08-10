"""SQLite/SQLModel engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

_connect_args: dict = {}
_engine_kwargs: dict = {}

if settings.sqlalchemy_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
    if ":memory:" in settings.sqlalchemy_url:
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(
    settings.sqlalchemy_url,
    echo=False,
    connect_args=_connect_args,
    **_engine_kwargs,
)


def init_db() -> None:
    """Create tables. Imported for the side effect of registering models."""
    from app import models  # noqa: F401  (registers tables on SQLModel.metadata)

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(engine) as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for use outside request handlers (background tasks)."""
    with Session(engine) as session:
        yield session
