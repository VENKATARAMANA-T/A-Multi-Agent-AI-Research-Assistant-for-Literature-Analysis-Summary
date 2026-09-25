"""Gemini client used by every agent.

Responsibilities:
  * one place that knows about the google-genai SDK,
  * structured JSON generation with schema-guided decoding and repair,
  * retries with backoff on transient errors,
  * a clearly-labelled offline stub when GOOGLE_API_KEY is unset, so the API
    stays usable (and testable) without a key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, wait_exponential

from app.config import settings
from app.services import llm_cache

logger = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# A *global* ceiling on in-flight requests. Per-call-site limits are not enough:
# the review pipeline fans out into concurrent branches, and each branch fans out
# again per paper, so independent limits would multiply and trip the provider's
# per-minute quota. Every call passes through this one gate.
_inflight = threading.Semaphore(max(1, settings.llm_max_concurrency))

# The installed google-genai has no client-side timeout, so a stalled request
# blocks forever — observed as a ten-minute hang on a single question. We impose
# the deadline ourselves by running the blocking SDK call on a worker and giving
# up on the *wait*. The abandoned worker cannot be killed, so the pool is sized
# well above the concurrency gate to absorb a few of them.
_call_pool = ThreadPoolExecutor(
    max_workers=max(8, settings.llm_max_concurrency * 4),
    thread_name_prefix="gemini",
)

MAX_ATTEMPTS = 4


class _RateLimiter:
    """Spaces requests out to stay under a requests-per-minute cap.

    The semaphore above limits how many calls are *in flight*; it does nothing
    about how quickly they are issued. A five-paper fan-out with three workers
    still fires five requests within a second, which a five-per-minute quota
    rejects outright. This enforces a minimum gap between request starts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    @property
    def interval(self) -> float:
        rpm = settings.llm_requests_per_minute
        return 60.0 / rpm if rpm and rpm > 0 else 0.0

    def acquire(self) -> float:
        """Block until the next request may start. Returns the seconds waited."""
        interval = self.interval
        if interval <= 0:
            return 0.0

        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_allowed)
            self._next_allowed = start_at + interval
            delay = start_at - now

        if delay > 0:
            logger.debug("Rate limit: waiting %.1fs before the next request", delay)
            time.sleep(delay)
        return delay

    def reset(self) -> None:
        with self._lock:
            self._next_allowed = 0.0


_rate_limiter = _RateLimiter()


def _stop_retrying(retry_state) -> bool:
    """Stop after N attempts, or once the whole loop exceeds its time budget.

    Read as a callable rather than composed from `stop_after_delay(...)` at
    import time: a decorator argument captures the setting's value once, which
    silently ignores any later configuration change.
    """
    if retry_state.attempt_number >= MAX_ATTEMPTS:
        return True
    elapsed = retry_state.seconds_since_start or 0
    return elapsed >= max(1, settings.llm_total_retry_seconds)


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Raised when no API key is configured and a real generation was required."""


class TransientLLMError(LLMError):
    """Rate limits and 5xx — worth retrying with backoff."""


class QuotaExceededError(LLMError):
    """A hard cap (e.g. requests-per-day) that retrying cannot clear."""


# Quota ids that reset per day/project rather than per minute. Retrying these
# just burns minutes of wall-clock before failing anyway.
_HARD_QUOTA_MARKERS = ("PerDay", "per_day", "FreeTierRequestsPerDay")

_RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "deadline", "timeout", "UNAVAILABLE")


def classify_provider_error(message: str) -> LLMError:
    """Map a raw provider error to the right exception type."""
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        if any(marker in message for marker in _HARD_QUOTA_MARKERS):
            return QuotaExceededError(summarise_provider_error(message))
        return TransientLLMError(message)
    if any(marker in message for marker in _RETRYABLE_MARKERS):
        return TransientLLMError(message)
    return LLMError(summarise_provider_error(message))


def summarise_provider_error(message: str) -> str:
    """Condense a provider error into one readable line.

    Google returns a multi-kilobyte JSON blob; dumping that into a report or a
    UI banner is useless, so the useful parts are pulled out.
    """
    quota_match = re.search(r"'?quotaValue'?:\s*'?(\d+)'?", message)
    model_match = re.search(r"model:\s*([\w.-]+)", message)
    retry_match = re.search(r"retryDelay'?:\s*'?(\d+)s", message)

    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        parts = ["Gemini quota exceeded"]
        if model_match:
            parts.append(f"for model {model_match.group(1)}")
        if quota_match:
            period = "per day" if any(m in message for m in _HARD_QUOTA_MARKERS) else "per minute"
            parts.append(f"(limit {quota_match.group(1)} requests {period})")
        summary = " ".join(parts) + "."
        if any(m in message for m in _HARD_QUOTA_MARKERS):
            return summary + " The daily free-tier allowance is spent — enable billing or wait for the reset."
        if retry_match:
            return summary + f" Retry in about {retry_match.group(1)}s."
        return summary

    status = re.search(r"'status':\s*'([A-Z_]+)'", message)
    detail = re.search(r"'message':\s*'([^']{0,200})", message)
    if status or detail:
        return f"{status.group(1) if status else 'Error'}: {detail.group(1) if detail else message[:200]}"
    return message[:300]


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    stubbed: bool = False
    cached: bool = False


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.google_api_key
        self.model = model or settings.gemini_model
        self.temperature = settings.gemini_temperature if temperature is None else temperature
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self.available:
                raise LLMUnavailable(
                    "GOOGLE_API_KEY is not configured. Set it in backend/.env to enable the agents."
                )
            from google import genai  # imported lazily so the app boots without the SDK

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # --- generation ----------------------------------------------------------

    # Free-tier quota errors routinely ask for a ~60s wait, so the backoff has to
    # reach that far or a multi-agent run gives up on the first burst of 429s.
    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=_stop_retrying,
        wait=wait_exponential(multiplier=2, min=5, max=65),
        reraise=True,
    )
    def _call(self, contents: Any, config: dict[str, Any]) -> Any:
        from google.genai import types

        client = self._ensure_client()
        timeout = max(5, settings.llm_timeout_seconds)

        with _inflight:
            _rate_limiter.acquire()
            future = _call_pool.submit(
                client.models.generate_content,
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout as exc:
                future.cancel()
                raise TransientLLMError(
                    f"Gemini did not respond within {timeout}s."
                ) from exc
            except LLMError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                raise classify_provider_error(str(exc)) from exc

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = max_output_tokens or settings.gemini_max_output_tokens

        # A cache hit is a legitimate substitute only if every input that could
        # change the answer matches, so the key covers all of them.
        key = llm_cache.cache_key(
            model=self.model,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=resolved_temperature,
            max_output_tokens=resolved_max_tokens,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )
        if use_cache:
            hit = llm_cache.lookup(key)
            if hit is not None:
                return LLMResponse(
                    text=hit.response_text,
                    model=hit.model,
                    prompt_tokens=hit.prompt_tokens,
                    output_tokens=hit.output_tokens,
                    cached=True,
                )

        config: dict[str, Any] = {
            "temperature": resolved_temperature,
            "max_output_tokens": resolved_max_tokens,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        if response_mime_type:
            config["response_mime_type"] = response_mime_type
        if response_schema:
            config["response_schema"] = response_schema

        response = self._call(prompt, config)
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response (possibly blocked by a safety filter).")

        usage = getattr(response, "usage_metadata", None)
        result = LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )

        if use_cache:
            llm_cache.store(
                key,
                model=self.model,
                response_text=result.text,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                prompt_preview=prompt[:300],
            )
        return result

    def generate_from_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Generate from a prompt plus one image (page scans, figures, charts).

        Cached on a hash of the image rather than the bytes themselves, so a
        repeated request is free without storing the image in the cache row.
        """
        from google.genai import types

        image_digest = hashlib.sha256(image_bytes).hexdigest()
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = max_output_tokens or settings.gemini_max_output_tokens

        key = llm_cache.cache_key(
            model=self.model,
            prompt=f"{prompt}\n[image:{mime_type}:{image_digest}]",
            system_instruction=None,
            temperature=resolved_temperature,
            max_output_tokens=resolved_max_tokens,
            response_mime_type=None,
            response_schema=None,
        )
        if use_cache:
            hit = llm_cache.lookup(key)
            if hit is not None:
                return hit.response_text

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]
        response = self._call(
            contents,
            {"temperature": resolved_temperature, "max_output_tokens": resolved_max_tokens},
        )
        text = (getattr(response, "text", None) or "").strip()

        if use_cache and text:
            usage = getattr(response, "usage_metadata", None)
            llm_cache.store(
                key,
                model=self.model,
                response_text=text,
                prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                prompt_preview=prompt[:300],
            )
        return text

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object, repairing common formatting slips."""
        response = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            use_cache=use_cache,
        )
        return parse_json_object(response.text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    candidate = text.strip()

    block = JSON_BLOCK_RE.search(candidate)
    if block:
        candidate = block.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Model response was not JSON: {text[:300]}")
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            # Drop trailing commas, a frequent LLM slip.
            repaired = re.sub(r",(\s*[}\]])", r"\1", snippet)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Could not parse JSON from model response: {exc}") from exc

    if isinstance(parsed, list):
        return {"items": parsed}
    if not isinstance(parsed, dict):
        raise LLMError("Model returned a JSON scalar where an object was expected.")
    return parsed


_client: GeminiClient | None = None
_lock = threading.Lock()


def get_llm() -> GeminiClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = GeminiClient()
    return _client


def set_llm(client: GeminiClient | None) -> None:
    """Test/DI hook — inject a fake client."""
    global _client
    with _lock:
        _client = client
