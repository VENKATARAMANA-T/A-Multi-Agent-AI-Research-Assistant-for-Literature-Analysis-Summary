"""Background job tracking for long-running work.

Ingestion is CPU- and IO-bound (PDF parsing, then embedding), so holding the
HTTP request open for a 20-file upload gives the user a spinner and no
information for minutes. Jobs move that work to a thread pool and expose
progress, so the client can show per-file state as it happens.

State lives in the database rather than in memory: progress survives a reload
of the page, and a job that was running when the process died is visible as
stale rather than silently vanishing.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from sqlalchemy import Column, Text
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel, desc, select

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    kind: str = Field(default="ingest", index=True)
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)

    total: int = 0
    completed: int = 0
    failed: int = 0

    # Per-item progress, e.g. {"name", "state", "stage", "detail", "paper_id"}.
    items: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    message: Optional[str] = None
    error: Optional[str] = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED)

    @property
    def progress(self) -> float:
        if self.total <= 0:
            return 0.0
        return round((self.completed + self.failed) / self.total, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value if isinstance(self.status, JobStatus) else str(self.status),
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "progress": self.progress,
            "items": self.items or [],
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


# --- persistence helpers -----------------------------------------------------


def create_job(kind: str, item_names: list[str], message: str | None = None) -> Job:
    from app.database import session_scope

    job = Job(
        kind=kind,
        status=JobStatus.QUEUED,
        total=len(item_names),
        items=[{"name": name, "state": "queued", "stage": None, "detail": None} for name in item_names],
        message=message,
    )
    with session_scope() as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def get_job(job_id: str) -> Job | None:
    from app.database import session_scope

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        session.refresh(job)
        # Detach a plain copy so callers can read it outside the session.
        return Job(**job.model_dump())


def list_jobs(limit: int = 25, kind: str | None = None) -> list[Job]:
    from app.database import session_scope

    with session_scope() as session:
        statement = select(Job).order_by(desc(Job.created_at)).limit(limit)
        if kind:
            statement = statement.where(Job.kind == kind)
        return [Job(**row.model_dump()) for row in session.exec(statement).all()]


def _mutate(job_id: str, apply: Callable[[Job], None]) -> None:
    """Apply a change to a job row. Never raises — progress must not break work."""
    try:
        from app.database import session_scope

        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            apply(job)
            job.updated_at = _utcnow()
            session.add(job)
            session.commit()
    except Exception:  # pragma: no cover
        logger.warning("Could not update job %s", job_id, exc_info=True)


def mark_running(job_id: str, message: str | None = None) -> None:
    def _apply(job: Job) -> None:
        job.status = JobStatus.RUNNING
        if message:
            job.message = message

    _mutate(job_id, _apply)


def update_item(
    job_id: str,
    index: int,
    *,
    state: str | None = None,
    stage: str | None = None,
    detail: str | None = None,
    paper_id: str | None = None,
) -> None:
    """Update one item's progress. `stage` is the pipeline step it is on."""

    def _apply(job: Job) -> None:
        items = list(job.items or [])
        if not 0 <= index < len(items):
            return
        item = dict(items[index])
        if state is not None:
            item["state"] = state
        if stage is not None:
            item["stage"] = stage
        if detail is not None:
            item["detail"] = detail
        if paper_id is not None:
            item["paper_id"] = paper_id
        items[index] = item
        # SQLAlchemy only notices a whole-list replacement on a JSON column.
        job.items = items

    _mutate(job_id, _apply)


def record_result(job_id: str, index: int, *, ok: bool, state: str, detail: str | None = None,
                  paper_id: str | None = None) -> None:
    def _apply(job: Job) -> None:
        items = list(job.items or [])
        if 0 <= index < len(items):
            item = dict(items[index])
            item["state"] = state
            item["stage"] = None
            if detail is not None:
                item["detail"] = detail
            if paper_id is not None:
                item["paper_id"] = paper_id
            items[index] = item
            job.items = items
        if ok:
            job.completed += 1
        else:
            job.failed += 1

    _mutate(job_id, _apply)


def finish(job_id: str, error: str | None = None) -> None:
    def _apply(job: Job) -> None:
        job.finished_at = _utcnow()
        if error:
            job.status = JobStatus.FAILED
            job.error = error
        elif job.failed and job.completed:
            job.status = JobStatus.PARTIAL
        elif job.failed:
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.COMPLETED

    _mutate(job_id, _apply)


# --- worker pool -------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
    return _executor


def submit(job_id: str, work: Callable[[str], None]) -> None:
    """Run `work(job_id)` on the pool, marking the job finished either way."""

    def _run() -> None:
        try:
            mark_running(job_id)
            work(job_id)
            finish(job_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Job %s crashed", job_id)
            finish(job_id, error=str(exc))

    get_executor().submit(_run)


def shutdown() -> None:
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None


def reclaim_stale_jobs() -> int:
    """Mark jobs left RUNNING by a previous process as failed, at startup."""
    try:
        from app.database import session_scope

        with session_scope() as session:
            stale = list(
                session.exec(
                    select(Job).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))  # type: ignore[attr-defined]
                ).all()
            )
            for job in stale:
                job.status = JobStatus.FAILED
                job.error = "Interrupted — the server restarted while this job was running."
                job.finished_at = _utcnow()
                session.add(job)
            session.commit()
            return len(stale)
    except Exception:  # pragma: no cover
        logger.warning("Could not reclaim stale jobs", exc_info=True)
        return 0
