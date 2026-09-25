"""Background job status and live progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.services import jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# How often the stream re-reads job state. Fast enough to feel live, slow
# enough that a long ingest does not hammer SQLite.
POLL_SECONDS = 0.5
# Stop streaming a job that never terminates, so a dropped client cannot leave
# the generator running forever.
MAX_STREAM_SECONDS = 1800


@router.get("", summary="List recent jobs")
def list_jobs(
    limit: int = Query(default=25, ge=1, le=200),
    kind: str | None = Query(default=None),
) -> list[dict]:
    return [job.to_dict() for job in jobs.list_jobs(limit=limit, kind=kind)]


@router.get("/{job_id}", summary="Get one job's status")
def get_job(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()


@router.get("/{job_id}/stream", summary="Stream job progress (server-sent events)")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Emit the job's state on every change until it reaches a terminal status."""
    if jobs.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_stream():
        last_payload: str | None = None
        waited = 0.0

        while waited < MAX_STREAM_SECONDS:
            if await request.is_disconnected():
                return

            job = jobs.get_job(job_id)
            if job is None:
                yield _event("error", {"detail": "Job disappeared."})
                return

            payload = json.dumps(job.to_dict())
            # Only push on change; an idle job should not spam the client.
            if payload != last_payload:
                yield _event("progress", job.to_dict())
                last_payload = payload

            if job.is_terminal:
                yield _event("done", job.to_dict())
                return

            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS

        yield _event("error", {"detail": "Stream timed out; poll the job endpoint instead."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the stream and progress arrives in one lump.
            "X-Accel-Buffering": "no",
        },
    )


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"
