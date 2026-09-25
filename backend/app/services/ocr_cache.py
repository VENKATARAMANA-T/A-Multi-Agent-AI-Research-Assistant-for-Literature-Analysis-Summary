"""Cache for recognised page text.

OCR is the slowest step in ingestion — roughly three seconds per page against
milliseconds for everything else. Re-indexing a paper (after a chunk-size change,
say) would otherwise redo all of it. The key is derived from the document's
content hash and page number, so a hit avoids even rendering the page image.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel, delete, func, select

from app.config import settings

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OCRCacheEntry(SQLModel, table=True):
    __tablename__ = "ocr_cache"

    key: str = Field(primary_key=True)
    engine: str = Field(index=True)
    text: str = Field(default="", sa_column=Column(Text))
    confidence: float = 0.0
    columns: int = 1
    hit_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)


def lookup(key: str) -> OCRCacheEntry | None:
    if not settings.ocr_cache_enabled:
        return None
    try:
        from app.database import session_scope

        with session_scope() as session:
            entry = session.get(OCRCacheEntry, key)
            if entry is None:
                return None
            entry.hit_count += 1
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return OCRCacheEntry(**entry.model_dump())
    except Exception:  # pragma: no cover - a cache must never break ingestion
        logger.warning("OCR cache lookup failed; continuing without it", exc_info=True)
        return None


def store(key: str, *, text: str, engine: str, confidence: float, columns: int = 1) -> None:
    if not settings.ocr_cache_enabled:
        return
    try:
        from app.database import session_scope

        with session_scope() as session:
            session.merge(
                OCRCacheEntry(
                    key=key,
                    engine=engine,
                    text=text,
                    confidence=confidence,
                    columns=columns,
                )
            )
            session.commit()
    except Exception:  # pragma: no cover
        logger.warning("OCR cache write failed; continuing", exc_info=True)


def clear() -> int:
    from app.database import session_scope

    with session_scope() as session:
        total = int(session.exec(select(func.count()).select_from(OCRCacheEntry)).one())
        session.exec(delete(OCRCacheEntry))
        session.commit()
    return total


def summary() -> dict:
    payload: dict = {"enabled": settings.ocr_cache_enabled}
    try:
        from app.database import session_scope

        with session_scope() as session:
            payload["entries"] = int(
                session.exec(select(func.count()).select_from(OCRCacheEntry)).one()
            )
            saved = session.exec(select(func.sum(OCRCacheEntry.hit_count))).one()
            payload["pages_saved"] = int(saved or 0)
    except Exception:  # pragma: no cover
        payload["entries"] = 0
        payload["pages_saved"] = 0
    return payload
