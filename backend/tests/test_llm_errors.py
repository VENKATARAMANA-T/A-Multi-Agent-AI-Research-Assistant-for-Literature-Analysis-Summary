"""Provider-error handling.

Found live: a daily-quota 429 was retried four times with 65s backoff before
failing anyway, and the raw multi-kilobyte JSON blob was pasted into reports.
"""

from __future__ import annotations

import pytest

from app.services.llm import (
    LLMError,
    QuotaExceededError,
    TransientLLMError,
    classify_provider_error,
    summarise_provider_error,
)

DAILY_QUOTA = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota"
    "\\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.6-flash', 'status': 'RESOURCE_EXHAUSTED', 'details': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}], "
    "'retryDelay': '11s'}}"
)

MINUTE_QUOTA = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'rate limited', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '10'}], "
    "'retryDelay': '9s'}}"
)


def test_daily_quota_is_not_retryable():
    error = classify_provider_error(DAILY_QUOTA)
    assert isinstance(error, QuotaExceededError)
    assert not isinstance(error, TransientLLMError)


def test_per_minute_quota_is_retryable():
    assert isinstance(classify_provider_error(MINUTE_QUOTA), TransientLLMError)


@pytest.mark.parametrize("message", ["503 UNAVAILABLE", "504 deadline exceeded", "500 internal"])
def test_server_errors_are_retryable(message):
    assert isinstance(classify_provider_error(message), TransientLLMError)


def test_other_errors_are_terminal():
    error = classify_provider_error("400 INVALID_ARGUMENT. {'status': 'INVALID_ARGUMENT'}")
    assert isinstance(error, LLMError)
    assert not isinstance(error, (TransientLLMError, QuotaExceededError))


def test_daily_quota_summary_is_short_and_actionable():
    summary = summarise_provider_error(DAILY_QUOTA)

    assert len(summary) < 220
    assert "quota exceeded" in summary.lower()
    assert "gemini-3.6-flash" in summary
    assert "limit 20 requests per day" in summary
    assert "billing" in summary
    assert "{" not in summary  # no raw JSON blob


def test_minute_quota_summary_mentions_the_retry_delay():
    summary = summarise_provider_error(MINUTE_QUOTA)
    assert "per minute" in summary
    assert "9s" in summary


def test_quota_error_is_reported_not_retried_by_an_agent(sample_pdf, monkeypatch):
    """A hard quota failure must surface immediately, not after four backoffs."""
    import time

    from app.agents.workflow import run_workflow
    from app.database import session_scope
    from app.services.ingestion import process_paper, store_upload
    from app.services.llm import set_llm

    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes())
        process_paper(session, paper.id)

    calls = {"n": 0}

    class QuotaClient:
        model = "fake"
        available = True

        def generate_json(self, *args, **kwargs):
            calls["n"] += 1
            raise classify_provider_error(DAILY_QUOTA)

        def generate(self, *args, **kwargs):
            raise classify_provider_error(DAILY_QUOTA)

    set_llm(QuotaClient())
    started = time.perf_counter()
    result = run_workflow("gap")
    elapsed = time.perf_counter() - started

    assert calls["n"] == 1, "a daily-quota error must not be retried"
    assert elapsed < 5, "failure should be immediate, not after exponential backoff"
    assert result["status"] == "partial"
    assert any("quota exceeded" in error.lower() for error in result["errors"])
    assert all(len(error) < 300 for error in result["errors"])
