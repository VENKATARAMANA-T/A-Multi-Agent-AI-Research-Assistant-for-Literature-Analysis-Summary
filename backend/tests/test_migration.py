"""Additive schema migration.

`create_all` creates missing tables but never alters existing ones, so a new
model field would otherwise mean deleting the database and re-indexing every
paper. These tests pin the two things that went wrong in practice.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlmodel import Session, select

from app.database import engine, sync_added_columns
from app.models import Paper, PaperStatus


def columns_of(table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_migration_adds_a_missing_column():
    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE "papers" DROP COLUMN "figure_count"'))
    assert "figure_count" not in columns_of("papers")

    added = sync_added_columns()

    assert added >= 1
    assert "figure_count" in columns_of("papers")


def test_existing_rows_get_the_model_default_not_null():
    """ADD COLUMN fills old rows with NULL, which ignores `int = 0` entirely.

    A pre-existing paper then failed response validation with
    "figure_count: Input should be a valid integer" — the whole API broke for
    every paper indexed before the field existed.
    """
    with Session(engine) as session:
        session.add(
            Paper(
                filename="old.pdf",
                file_path="old.pdf",
                content_hash="hash-old",
                status=PaperStatus.INDEXED,
            )
        )
        session.commit()

    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE "papers" DROP COLUMN "figure_count"'))
    sync_added_columns()

    with Session(engine) as session:
        paper = session.exec(select(Paper).where(Paper.content_hash == "hash-old")).one()
        assert paper.figure_count == 0, "existing row was left NULL"


def test_null_values_from_an_earlier_migration_are_repaired():
    """Columns added before the backfill existed still hold NULLs.

    SQLite cannot add a NOT NULL column to a populated table, so a migrated
    column is always nullable — which is how the NULLs got there in the first
    place. The sequence here reproduces that exactly.
    """
    with Session(engine) as session:
        session.add(
            Paper(
                filename="legacy.pdf",
                file_path="legacy.pdf",
                content_hash="hash-legacy",
                status=PaperStatus.INDEXED,
            )
        )
        session.commit()

    # Drop and re-add so the column is nullable, as a real migration leaves it.
    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE "papers" DROP COLUMN "figure_count"'))
    sync_added_columns()

    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "papers" SET "figure_count" = NULL WHERE "content_hash" = :h'),
            {"h": "hash-legacy"},
        )

    sync_added_columns()  # the repair pass

    with Session(engine) as session:
        paper = session.exec(select(Paper).where(Paper.content_hash == "hash-legacy")).one()
        assert paper.figure_count == 0


def test_migration_is_idempotent():
    first = sync_added_columns()
    second = sync_added_columns()
    assert first == 0 and second == 0


def test_string_defaults_are_backfilled():
    with Session(engine) as session:
        session.add(
            Paper(
                filename="src.pdf",
                file_path="src.pdf",
                content_hash="hash-src",
                status=PaperStatus.INDEXED,
            )
        )
        session.commit()

    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE "papers" DROP COLUMN "text_source"'))
    sync_added_columns()

    with engine.begin() as connection:
        connection.execute(
            text('UPDATE "papers" SET "text_source" = NULL WHERE "content_hash" = :h'),
            {"h": "hash-src"},
        )
    sync_added_columns()

    with Session(engine) as session:
        paper = session.exec(select(Paper).where(Paper.content_hash == "hash-src")).one()
        assert paper.text_source == "native"


@pytest.mark.parametrize("table", ["papers", "chunks", "figures", "jobs", "llm_cache", "ocr_cache"])
def test_every_expected_table_exists(table):
    assert table in inspect(engine).get_table_names()
