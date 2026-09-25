"""Content-addressed cache for LLM responses.

On the Gemini free tier the binding constraint is requests per day, not latency.
Re-running the same analysis — during a demo, a test, or an iteration on an
unrelated part of a report — otherwise burns the daily allowance on work the
model has already done.

The cache key covers everything that can change the response: model, prompt,
system instruction, temperature, output cap, MIME type and response schema.
Anything that would produce a different answer produces a different key, so a
hit is always a legitimate substitute for the call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Column, Text
from sqlmodel import Field, Session, SQLModel, delete, func, select

from app.config import settings

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMCacheEntry(SQLModel, table=True):
    __tablename__ = "llm_cache"

    key: str = Field(primary_key=True)
    model: str = Field(index=True)
    response_text: str = Field(sa_column=Column(Text))
    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_preview: str = Field(default="", sa_column=Column(Text))
    hit_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    last_used_at: datetime = Field(default_factory=_utcnow)


def cache_key(
    *,
    model: str,
    prompt: str,
    system_instruction: str | None,
    temperature: float,
    max_output_tokens: int,
    response_mime_type: str | None,
    response_schema: dict[str, Any] | None,
) -> str:
    """Stable hash over every input that can change the response."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system_instruction or "",
            "temperature": round(float(temperature), 4),
            "max_output_tokens": int(max_output_tokens),
            "mime": response_mime_type or "",
            # sort_keys makes an equivalent schema hash identically regardless
            # of the order the caller happened to build the dict in.
            "schema": json.dumps(response_schema, sort_keys=True) if response_schema else "",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 3),
        }


_stats = CacheStats()
_lock = threading.Lock()


def get_stats() -> CacheStats:
    return _stats


def reset_stats() -> None:
    global _stats
    with _lock:
        _stats = CacheStats()


def _expiry_cutoff() -> datetime | None:
    days = settings.llm_cache_ttl_days
    if days <= 0:
        return None
    return _utcnow() - timedelta(days=days)


def _as_aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; compare them as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def lookup(key: str) -> LLMCacheEntry | None:
    """Return a live cache entry, or None on miss/expiry/error."""
    if not settings.llm_cache_enabled:
        return None

    try:
        from app.database import session_scope

        with session_scope() as session:
            entry = session.get(LLMCacheEntry, key)
            if entry is None:
                with _lock:
                    _stats.misses += 1
                return None

            cutoff = _expiry_cutoff()
            if cutoff and _as_aware(entry.created_at) < cutoff:
                session.delete(entry)
                session.commit()
                with _lock:
                    _stats.misses += 1
                return None

            entry.hit_count += 1
            entry.last_used_at = _utcnow()
            session.add(entry)
            session.commit()
            session.refresh(entry)

            with _lock:
                _stats.hits += 1
            logger.info("LLM cache hit (%s, %d total hits)", key[:12], entry.hit_count)
            return entry
    except Exception:  # pragma: no cover - a cache must never break the caller
        logger.warning("LLM cache lookup failed; continuing without cache", exc_info=True)
        return None


def store(
    key: str,
    *,
    model: str,
    response_text: str,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    prompt_preview: str = "",
) -> None:
    """Persist a response. Failures are logged and swallowed."""
    if not settings.llm_cache_enabled:
        return

    try:
        from app.database import session_scope

        with session_scope() as session:
            session.merge(
                LLMCacheEntry(
                    key=key,
                    model=model,
                    response_text=response_text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    prompt_preview=prompt_preview[:300],
                )
            )
            session.commit()
        with _lock:
            _stats.writes += 1
    except Exception:  # pragma: no cover
        logger.warning("LLM cache write failed; continuing", exc_info=True)


def purge_expired(session: Session | None = None) -> int:
    """Drop entries past the TTL. Returns the number removed."""
    cutoff = _expiry_cutoff()
    if cutoff is None:
        return 0

    def _run(active: Session) -> int:
        stale = list(
            active.exec(select(LLMCacheEntry).where(LLMCacheEntry.created_at < cutoff)).all()
        )
        for entry in stale:
            active.delete(entry)
        active.commit()
        return len(stale)

    if session is not None:
        return _run(session)

    from app.database import session_scope

    with session_scope() as active:
        return _run(active)


def clear() -> int:
    """Remove every cached response. Returns the number removed."""
    from app.database import session_scope

    with session_scope() as session:
        total = int(session.exec(select(func.count()).select_from(LLMCacheEntry)).one())
        session.exec(delete(LLMCacheEntry))
        session.commit()
    reset_stats()
    return total


def summary() -> dict[str, Any]:
    """Cache size and saved-call count, for /api/health."""
    payload: dict[str, Any] = {
        "enabled": settings.llm_cache_enabled,
        "ttl_days": settings.llm_cache_ttl_days,
        **_stats.as_dict(),
    }
    try:
        from app.database import session_scope

        with session_scope() as session:
            payload["entries"] = int(
                session.exec(select(func.count()).select_from(LLMCacheEntry)).one()
            )
            saved = session.exec(select(func.sum(LLMCacheEntry.hit_count))).one()
            payload["calls_saved"] = int(saved or 0)
    except Exception:  # pragma: no cover
        payload["entries"] = 0
        payload["calls_saved"] = 0
    return payload
