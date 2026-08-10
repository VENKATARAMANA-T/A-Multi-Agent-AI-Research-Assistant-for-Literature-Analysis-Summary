"""Gemini client used by every agent.

Responsibilities:
  * one place that knows about the google-genai SDK,
  * structured JSON generation with schema-guided decoding and repair,
  * retries with backoff on transient errors,
  * a clearly-labelled offline stub when GOOGLE_API_KEY is unset, so the API
    stays usable (and testable) without a key.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


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
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=5, max=65),
        reraise=True,
    )
    def _call(self, contents: str, config: dict[str, Any]) -> Any:
        from google.genai import types

        client = self._ensure_client()
        try:
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
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
    ) -> LLMResponse:
        config: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": max_output_tokens or settings.gemini_max_output_tokens,
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
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object, repairing common formatting slips."""
        response = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
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
