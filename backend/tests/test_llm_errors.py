"""Provider-error handling.

Found live: a daily-quota 429 was retried four times with 65s backoff before
failing anyway, and the raw multi-kilobyte JSON blob was pasted into reports.
"""

from __future__ import annotations

import pytest

from app.services.llm import (
    GeminiClient,
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
        from tests.conftest import seed_user_id

        paper, _ = store_upload(
            session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id()
        )
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


# --- model fallback ----------------------------------------------------------


class _ModelRecorder:
    """Stands in for the google-genai client, failing per model on demand."""

    def __init__(self, failures: dict[str, Exception]):
        self.failures = failures
        self.asked: list[str] = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.asked.append(model)
        if model in self.failures:
            raise self.failures[model]

        class _Response:
            text = "answered"
            usage_metadata = None

        return _Response()


def build_client(failures: dict[str, Exception], monkeypatch) -> tuple[GeminiClient, _ModelRecorder]:
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_model", "primary-model")
    monkeypatch.setattr(settings, "gemini_fallback_models", "backup-one,backup-two")
    monkeypatch.setattr(settings, "llm_total_retry_seconds", 1)

    client = GeminiClient(api_key="test-key")
    recorder = _ModelRecorder(failures)
    client._client = recorder
    return client, recorder


def test_capacity_failure_moves_to_the_next_model(monkeypatch):
    """"High demand" on one model must not end the request."""
    client, recorder = build_client(
        {"primary-model": RuntimeError("503 UNAVAILABLE: model is overloaded")}, monkeypatch
    )

    result = client.generate("hello", use_cache=False)

    assert result.text == "answered"
    assert result.model == "backup-one", "the answering model must be reported, not the primary"
    assert recorder.asked[0] == "primary-model"
    assert "backup-one" in recorder.asked


def test_a_spent_daily_quota_moves_on(monkeypatch):
    """The daily cap is per model, so retrying the same one can never clear it."""
    client, recorder = build_client(
        {
            "primary-model": RuntimeError(
                "429 RESOURCE_EXHAUSTED {'quotaId': 'FreeTierRequestsPerDay', 'quotaValue': '20'}"
            )
        },
        monkeypatch,
    )

    assert client.generate("hello", use_cache=False).model == "backup-one"


def test_it_walks_the_whole_chain(monkeypatch):
    overloaded = RuntimeError("503 UNAVAILABLE: overloaded")
    client, recorder = build_client(
        {"primary-model": overloaded, "backup-one": overloaded}, monkeypatch
    )

    assert client.generate("hello", use_cache=False).model == "backup-two"
    assert set(recorder.asked) == {"primary-model", "backup-one", "backup-two"}


def test_every_model_failing_raises_the_last_error(monkeypatch):
    overloaded = RuntimeError("503 UNAVAILABLE: overloaded")
    client, _ = build_client(
        {"primary-model": overloaded, "backup-one": overloaded, "backup-two": overloaded},
        monkeypatch,
    )

    with pytest.raises(TransientLLMError):
        client.generate("hello", use_cache=False)


def test_a_bad_request_does_not_burn_a_second_model(monkeypatch):
    """A malformed request fails identically everywhere — trying again learns nothing."""
    client, recorder = build_client(
        {"primary-model": RuntimeError("400 INVALID_ARGUMENT: schema is not valid")}, monkeypatch
    )

    with pytest.raises(LLMError):
        client.generate("hello", use_cache=False)

    assert recorder.asked == ["primary-model"], "only the primary should have been charged"


def test_an_explicit_model_is_honoured_without_fallback(monkeypatch):
    """Asking for one model must not silently return another's answer."""
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_fallback_models", "backup-one")
    client = GeminiClient(api_key="test-key", model="pinned-model")

    assert client.fallback_models == []
    assert client.model_chain() == ["pinned-model"]


def test_the_chain_is_deduplicated(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_model", "same-model")
    monkeypatch.setattr(settings, "gemini_fallback_models", "same-model, other-model ,")

    assert settings.gemini_model_chain == ["same-model", "other-model"]


def test_fallback_can_be_switched_off(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_fallback_models", "")
    assert GeminiClient(api_key="test-key").fallback_models == []
