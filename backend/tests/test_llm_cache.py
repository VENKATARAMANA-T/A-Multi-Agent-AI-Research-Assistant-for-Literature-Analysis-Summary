"""The LLM response cache.

On the free tier the binding limit is requests per day, so a cache hit is not
just faster — it is the difference between being able to re-run an analysis and
not. These tests pin down that a hit is only ever served for a genuinely
identical request.
"""

from __future__ import annotations

import pytest

from app.services import llm_cache
from app.services.llm import GeminiClient, LLMResponse

BASE = {
    "model": "gemini-flash-latest",
    "prompt": "Summarise this paper.",
    "system_instruction": "You are a summarizer.",
    "temperature": 0.2,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    "response_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
}


def key(**overrides):
    return llm_cache.cache_key(**{**BASE, **overrides})


# --- key sensitivity ---------------------------------------------------------


def test_identical_requests_share_a_key():
    assert key() == key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "gemini-2.0-flash"),
        ("prompt", "Summarise this other paper."),
        ("system_instruction", "You are an extractor."),
        ("temperature", 0.9),
        ("max_output_tokens", 512),
        ("response_mime_type", "text/plain"),
        ("response_schema", {"type": "object", "properties": {"b": {"type": "string"}}}),
    ],
)
def test_any_meaningful_change_changes_the_key(field, value):
    """Anything that could change the answer must miss the cache."""
    assert key(**{field: value}) != key()


def test_schema_key_ignores_dict_ordering():
    """An equivalent schema built in a different order is still the same request."""
    a = key(response_schema={"type": "object", "title": "X"})
    b = key(response_schema={"title": "X", "type": "object"})
    assert a == b


# --- store and lookup --------------------------------------------------------


def test_store_then_lookup_returns_the_response():
    llm_cache.store(key(), model="m", response_text='{"a": 1}', prompt_tokens=10, output_tokens=5)
    hit = llm_cache.lookup(key())

    assert hit is not None
    assert hit.response_text == '{"a": 1}'
    assert hit.prompt_tokens == 10


def test_lookup_misses_on_an_unknown_key():
    assert llm_cache.lookup("never-stored") is None


def test_hit_count_increments():
    llm_cache.store(key(), model="m", response_text="x")
    llm_cache.lookup(key())
    hit = llm_cache.lookup(key())

    assert hit.hit_count == 2


def test_clear_empties_the_cache():
    llm_cache.store(key(), model="m", response_text="x")
    removed = llm_cache.clear()

    assert removed == 1
    assert llm_cache.lookup(key()) is None


def test_summary_reports_entries_and_saved_calls():
    llm_cache.store(key(), model="m", response_text="x")
    llm_cache.lookup(key())

    summary = llm_cache.summary()
    assert summary["enabled"] is True
    assert summary["entries"] == 1
    assert summary["calls_saved"] == 1
    assert summary["hits"] >= 1


def test_disabled_cache_neither_stores_nor_reads(monkeypatch):
    monkeypatch.setattr(llm_cache.settings, "llm_cache_enabled", False)

    llm_cache.store(key(), model="m", response_text="x")
    assert llm_cache.lookup(key()) is None


# --- integration with the client --------------------------------------------


class CountingClient(GeminiClient):
    """A client that records how many real generations it performs."""

    def __init__(self):
        super().__init__(api_key="test-key", model="gemini-flash-latest")
        self.calls = 0

    def _call(self, contents, config, model=None):  # type: ignore[override]
        self.calls += 1

        class _Response:
            text = '{"answer": "42"}'
            usage_metadata = None

        return _Response()


def test_second_identical_call_is_served_from_cache():
    client = CountingClient()

    first = client.generate("What is the answer?", system_instruction="Be terse.")
    second = client.generate("What is the answer?", system_instruction="Be terse.")

    assert client.calls == 1, "the second call must not reach the provider"
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


def test_a_different_prompt_still_calls_the_provider():
    client = CountingClient()

    client.generate("Question one")
    client.generate("Question two")

    assert client.calls == 2


def test_use_cache_false_bypasses_the_cache():
    client = CountingClient()

    client.generate("Same prompt")
    client.generate("Same prompt", use_cache=False)

    assert client.calls == 2


def test_generate_json_is_cached_too():
    client = CountingClient()

    first = client.generate_json("Give me JSON")
    second = client.generate_json("Give me JSON")

    assert client.calls == 1
    assert first == second == {"answer": "42"}


def test_cache_endpoints(client, sample_pdf):
    payload = client.get("/api/cache").json()
    assert payload["enabled"] is True
    assert payload["entries"] == 0

    llm_cache.store(key(), model="m", response_text="x")
    assert client.get("/api/cache").json()["entries"] == 1

    assert client.delete("/api/cache").json()["removed"] == 1
    assert client.get("/api/cache").json()["entries"] == 0


def test_health_reports_cache_state(client):
    payload = client.get("/api/health").json()

    assert "cache" in payload["llm"]
    assert payload["llm"]["cache"]["enabled"] is True
    assert payload["llm"]["max_concurrency"] >= 1
