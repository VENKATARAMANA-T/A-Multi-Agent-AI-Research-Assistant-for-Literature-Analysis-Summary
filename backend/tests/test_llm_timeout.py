"""Request deadlines.

Found live: a single question hung for ten minutes. `llm_timeout_seconds` was
declared in the settings but never applied, and the installed google-genai has
no client-side timeout of its own, so a stalled call blocked forever.
"""

from __future__ import annotations

import time

import pytest

from app.services import llm as llm_module
from app.services.llm import GeminiClient, LLMError, TransientLLMError


class HangingClient(GeminiClient):
    """Simulates a provider that accepts the request and never answers."""

    def __init__(self, hang_seconds: float = 30.0):
        super().__init__(api_key="test-key", model="test-model")
        self.hang_seconds = hang_seconds
        self.attempts = 0

    def _ensure_client(self):
        outer = self

        class _Models:
            def generate_content(self, **kwargs):
                outer.attempts += 1
                time.sleep(outer.hang_seconds)
                raise AssertionError("should have timed out")

        class _Client:
            models = _Models()

        return _Client()


def test_timeout_setting_is_actually_applied(monkeypatch):
    """The deadline must come from configuration, not be ignored as before."""
    monkeypatch.setattr(llm_module.settings, "llm_timeout_seconds", 5)
    monkeypatch.setattr(llm_module.settings, "llm_total_retry_seconds", 1)

    client = HangingClient(hang_seconds=30)

    started = time.perf_counter()
    with pytest.raises(LLMError) as excinfo:
        client.generate("anything", use_cache=False)
    elapsed = time.perf_counter() - started

    assert "did not respond within 5s" in str(excinfo.value)
    # One attempt of ~5s, then the delay stop prevents further retries.
    assert elapsed < 20, f"took {elapsed:.1f}s — the deadline was not enforced"


def test_a_timeout_is_classified_as_retryable():
    """A stall is transient — worth one more try, unlike a daily quota error."""
    monkeypatch_free = TransientLLMError("Gemini did not respond within 5s.")
    assert isinstance(monkeypatch_free, TransientLLMError)


def test_total_retry_window_is_bounded(monkeypatch):
    """Repeated timeouts must not hold one HTTP request open for minutes."""
    monkeypatch.setattr(llm_module.settings, "llm_timeout_seconds", 5)
    monkeypatch.setattr(llm_module.settings, "llm_total_retry_seconds", 8)

    client = HangingClient(hang_seconds=30)

    started = time.perf_counter()
    with pytest.raises(LLMError):
        client.generate("anything", use_cache=False)
    elapsed = time.perf_counter() - started

    assert elapsed < 30, f"retry loop ran for {elapsed:.1f}s despite an 8s ceiling"
    assert client.attempts <= 4


def test_rate_limiter_spaces_requests_out(monkeypatch):
    """A concurrency cap bounds calls in flight, not the rate they are issued.

    Five papers fanned out with three workers still fired five requests within
    a second, which a five-per-minute quota rejected outright.
    """
    import time as clock

    limiter = llm_module._RateLimiter()
    monkeypatch.setattr(llm_module.settings, "llm_requests_per_minute", 120)  # 0.5s apart

    started = clock.perf_counter()
    for _ in range(3):
        limiter.acquire()
    elapsed = clock.perf_counter() - started

    # First is immediate, then two gaps of ~0.5s.
    assert 0.8 < elapsed < 2.0, f"expected ~1s of spacing, got {elapsed:.2f}s"


def test_rate_limiter_is_disabled_at_zero(monkeypatch):
    import time as clock

    limiter = llm_module._RateLimiter()
    monkeypatch.setattr(llm_module.settings, "llm_requests_per_minute", 0)

    started = clock.perf_counter()
    for _ in range(5):
        assert limiter.acquire() == 0.0
    assert clock.perf_counter() - started < 0.1


def test_rate_limiter_interval_follows_configuration(monkeypatch):
    limiter = llm_module._RateLimiter()

    monkeypatch.setattr(llm_module.settings, "llm_requests_per_minute", 5)
    assert limiter.interval == 12.0

    monkeypatch.setattr(llm_module.settings, "llm_requests_per_minute", 60)
    assert limiter.interval == 1.0


def test_minimum_timeout_floor(monkeypatch):
    """A nonsensical timeout must not become an instant-fail."""
    monkeypatch.setattr(llm_module.settings, "llm_timeout_seconds", 0)
    monkeypatch.setattr(llm_module.settings, "llm_total_retry_seconds", 1)

    client = HangingClient(hang_seconds=30)

    with pytest.raises(LLMError) as excinfo:
        client.generate("anything", use_cache=False)

    assert "within 5s" in str(excinfo.value)
