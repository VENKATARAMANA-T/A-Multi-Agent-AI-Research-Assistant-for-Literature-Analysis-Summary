"""SQLite/SQLModel engine and session helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

logger = logging.getLogger(__name__)

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
    """Create tables. Imports are for the side effect of registering models."""
    from app import models  # noqa: F401  (registers tables on SQLModel.metadata)
    from app.services import jobs  # noqa: F401
    from app.services import llm_cache  # noqa: F401
    from app.services import ocr_cache  # noqa: F401

    SQLModel.metadata.create_all(engine)
    sync_added_columns()


def sync_added_columns() -> int:
    """Add columns that exist on the models but not yet in the database.

    `create_all` creates missing *tables* but never alters existing ones, so a
    new field would otherwise require deleting the database and re-indexing
    every paper. This handles the additive case — the only kind this project has
    needed — and deliberately does nothing about renames, type changes or drops.
    A schema that outgrows this wants Alembic.
    """
    from sqlalchemy import inspect, text

    added = 0
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    # Repair rows left NULL by an earlier migration that added
                    # the column without applying the model's default. The
                    # statement is a no-op once there is nothing left to fix.
                    repair = _backfill_statement(table.name, column)
                    if repair is not None:
                        connection.execute(text(repair))
                    continue
                ddl = _add_column_ddl(table.name, column)
                if ddl is None:
                    logger.warning(
                        "Cannot auto-add %s.%s; migrate this column manually.",
                        table.name,
                        column.name,
                    )
                    continue
                connection.execute(text(ddl))

                # ALTER TABLE ADD COLUMN fills existing rows with NULL, which
                # ignores the model's default entirely — a new `int = 0` field
                # came back as None on every pre-existing row and failed
                # validation. Backfill so old rows match a freshly created one.
                backfill = _backfill_statement(table.name, column)
                if backfill is not None:
                    connection.execute(text(backfill))

                logger.info("Schema: added %s.%s", table.name, column.name)
                added += 1
    return added


def _mapped_classes():
    """Every SQLModel table class, via SQLAlchemy's registry."""
    registry = getattr(SQLModel, "_sa_registry", None)
    if registry is None:  # pragma: no cover - SQLModel internals changed
        return []
    return [
        (mapper.local_table.name, mapper.class_)
        for mapper in registry.mappers
        if getattr(mapper, "local_table", None) is not None
    ]


def _model_default(table_name: str, column_name: str):
    """The Python-side default a SQLModel field declares, if any."""
    for name, model in _mapped_classes():
        if name != table_name:
            continue
        field = getattr(model, "model_fields", {}).get(column_name)
        if field is None:
            return None
        if getattr(field, "default_factory", None) is not None:
            try:
                return field.default_factory()
            except Exception:  # pragma: no cover
                return None
        default = getattr(field, "default", None)
        # Pydantic marks "no default" with a sentinel rather than None.
        if default is None or repr(default) == "PydanticUndefined":
            return None
        return default
    return None


def _backfill_statement(table_name: str, column) -> str | None:
    default = _model_default(table_name, column.name)
    if default is None:
        return None

    if isinstance(default, bool):
        literal = "1" if default else "0"
    elif isinstance(default, (int, float)):
        literal = str(default)
    elif isinstance(default, str):
        escaped = default.replace("'", "''")
        literal = f"'{escaped}'"
    elif isinstance(default, (list, dict)):
        import json

        literal = "'" + json.dumps(default).replace("'", "''") + "'"
    elif hasattr(default, "value"):  # Enum
        escaped = str(default.value).replace("'", "''")
        literal = f"'{escaped}'"
    else:
        return None

    return f'UPDATE "{table_name}" SET "{column.name}" = {literal} WHERE "{column.name}" IS NULL'


def _add_column_ddl(table_name: str, column) -> str | None:
    """Build an ALTER TABLE ADD COLUMN statement, or None if unsafe."""
    # A NOT NULL column with no default cannot be added to a populated table.
    if not column.nullable and column.default is None and column.server_default is None:
        return None
    try:
        column_type = column.type.compile(engine.dialect)
    except Exception:  # pragma: no cover - exotic types
        return None
    return f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {column_type}'


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(engine) as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for use outside request handlers (background tasks)."""
    with Session(engine) as session:
        yield session
