"""Background ingestion and job progress tracking."""

from __future__ import annotations

import io
import time

from app.services import jobs


def wait_for_job(client, job_id: str, timeout: float = 90.0) -> dict:
    """Poll until the job reaches a terminal status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("completed", "partial", "failed"):
            return payload
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def upload_async(client, *paths):
    files = []
    for path in paths:
        files.append(("files", (path.name, path.read_bytes(), "application/pdf")))
    return client.post("/api/papers/upload", files=files)


# --- job model ---------------------------------------------------------------


def test_create_job_seeds_one_item_per_file():
    job = jobs.create_job("ingest", ["a.pdf", "b.pdf"])

    assert job.total == 2
    assert job.status == jobs.JobStatus.QUEUED
    assert [item["name"] for item in job.items] == ["a.pdf", "b.pdf"]
    assert all(item["state"] == "queued" for item in job.items)


def test_progress_is_a_fraction_of_total():
    job = jobs.create_job("ingest", ["a.pdf", "b.pdf", "c.pdf", "d.pdf"])
    jobs.record_result(job.id, 0, ok=True, state="indexed")

    assert jobs.get_job(job.id).progress == 0.25


def test_item_updates_are_persisted():
    job = jobs.create_job("ingest", ["a.pdf"])
    jobs.update_item(job.id, 0, state="running", stage="extracting", paper_id="p1")

    item = jobs.get_job(job.id).items[0]
    assert item["state"] == "running"
    assert item["stage"] == "extracting"
    assert item["paper_id"] == "p1"


def test_out_of_range_item_index_is_ignored():
    job = jobs.create_job("ingest", ["a.pdf"])
    jobs.update_item(job.id, 99, state="running")  # must not raise

    assert jobs.get_job(job.id).items[0]["state"] == "queued"


def test_finish_status_reflects_outcomes():
    all_ok = jobs.create_job("ingest", ["a", "b"])
    jobs.record_result(all_ok.id, 0, ok=True, state="indexed")
    jobs.record_result(all_ok.id, 1, ok=True, state="indexed")
    jobs.finish(all_ok.id)
    assert jobs.get_job(all_ok.id).status == jobs.JobStatus.COMPLETED

    mixed = jobs.create_job("ingest", ["a", "b"])
    jobs.record_result(mixed.id, 0, ok=True, state="indexed")
    jobs.record_result(mixed.id, 1, ok=False, state="failed")
    jobs.finish(mixed.id)
    assert jobs.get_job(mixed.id).status == jobs.JobStatus.PARTIAL

    broken = jobs.create_job("ingest", ["a"])
    jobs.record_result(broken.id, 0, ok=False, state="failed")
    jobs.finish(broken.id)
    assert jobs.get_job(broken.id).status == jobs.JobStatus.FAILED


def test_finish_with_an_error_marks_failed():
    job = jobs.create_job("ingest", ["a"])
    jobs.finish(job.id, error="disk exploded")

    finished = jobs.get_job(job.id)
    assert finished.status == jobs.JobStatus.FAILED
    assert finished.error == "disk exploded"


def test_stale_jobs_are_reclaimed_on_restart():
    """A job left RUNNING belongs to a process that no longer exists."""
    job = jobs.create_job("ingest", ["a"])
    jobs.mark_running(job.id)

    assert jobs.reclaim_stale_jobs() >= 1

    reclaimed = jobs.get_job(job.id)
    assert reclaimed.status == jobs.JobStatus.FAILED
    assert "restarted" in reclaimed.error


def test_unknown_job_returns_none():
    assert jobs.get_job("nope") is None


# --- HTTP --------------------------------------------------------------------


def test_background_upload_returns_a_job_immediately(client, sample_pdf):
    response = upload_async(client, sample_pdf)

    assert response.status_code == 202
    payload = response.json()
    assert payload["kind"] == "ingest"
    assert payload["total"] == 1
    assert payload["status"] in ("queued", "running")


def test_background_upload_indexes_the_paper(client, sample_pdf):
    job_id = upload_async(client, sample_pdf).json()["id"]
    finished = wait_for_job(client, job_id)

    assert finished["status"] == "completed"
    assert finished["completed"] == 1
    assert finished["progress"] == 1.0

    item = finished["items"][0]
    assert item["state"] == "indexed"
    assert item["paper_id"]
    assert "chunks" in item["detail"]

    papers = client.get("/api/papers").json()
    assert len(papers) == 1
    assert papers[0]["status"] == "indexed"


def test_background_upload_reports_per_file_failure(client, sample_pdf):
    files = [
        ("files", (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")),
        ("files", ("broken.pdf", io.BytesIO(b"not a pdf").read(), "application/pdf")),
    ]
    job_id = client.post("/api/papers/upload", files=files).json()["id"]
    finished = wait_for_job(client, job_id)

    assert finished["status"] == "partial"
    assert finished["completed"] == 1
    assert finished["failed"] == 1

    states = {item["name"]: item["state"] for item in finished["items"]}
    assert states[sample_pdf.name] == "indexed"
    assert states["broken.pdf"] == "failed"


def test_background_upload_detects_duplicates(client, sample_pdf):
    wait_for_job(client, upload_async(client, sample_pdf).json()["id"])
    second = wait_for_job(client, upload_async(client, sample_pdf).json()["id"])

    assert second["items"][0]["state"] == "duplicate"
    assert len(client.get("/api/papers").json()) == 1


def test_job_listing_and_lookup(client, sample_pdf):
    job_id = upload_async(client, sample_pdf).json()["id"]
    wait_for_job(client, job_id)

    listing = client.get("/api/jobs").json()
    assert any(job["id"] == job_id for job in listing)
    assert client.get(f"/api/jobs/{job_id}").json()["id"] == job_id
    assert client.get("/api/jobs/missing").status_code == 404


def test_progress_stream_emits_events_and_terminates(client, sample_pdf):
    job_id = upload_async(client, sample_pdf).json()["id"]

    with client.stream("GET", f"/api/jobs/{job_id}/stream") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(response.iter_text())

    assert "event: progress" in body
    assert "event: done" in body


def test_stream_404s_for_an_unknown_job(client):
    assert client.get("/api/jobs/nope/stream").status_code == 404
