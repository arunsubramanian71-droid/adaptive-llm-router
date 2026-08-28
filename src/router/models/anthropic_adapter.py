"""Anthropic provider adapter.

Translates between the provider-agnostic `ModelClient` interface and the
`anthropic` SDK. This is the ONLY module in the codebase allowed to import
`anthropic` or know about its response shapes.

API contract verified against current Anthropic documentation (2026-08-25):
- Messages API: `client.messages.create(model=..., max_tokens=..., messages=...)`
- Usage fields on `response.usage`: `input_tokens`, `output_tokens`,
  `cache_creation_input_tokens`, `cache_read_input_tokens`. There is no
  separate billed "reasoning/thinking token" field — thinking tokens are
  counted within `output_tokens`.
- Extended thinking: `thinking={"type": "adaptive", "display": ...}` is the
  current form; `budget_tokens` is rejected on current-generation models.
- Effort: `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}`.
- Request id: `response._request_id` (public despite the underscore).
- Retries are handled explicitly here (SDK client is constructed with
  `max_retries=0`) so every retry is counted and recorded rather than
  hidden inside the SDK.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

import anthropic

from router.config import ModelEntry
from router.models.base import ModelClient
from router.models.schemas import (
    CompletionStatus,
    GenerationConfig,
    NormalizedCompletion,
    TokenUsage,
)

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_BACKOFF_SECONDS = 30.0


def build_request_kwargs(
    prompt: str,
    model_entry: ModelEntry,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Pure translation of our config into `messages.create` kwargs.

    Kept separate from the client so it can be unit tested without a
    network call or an API key.
    """
    kwargs: dict[str, Any] = {
        "model": model_entry.id,
        "max_tokens": model_entry.max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if model_entry.temperature is not None:
        kwargs["temperature"] = model_entry.temperature
    if model_entry.top_p is not None:
        kwargs["top_p"] = model_entry.top_p
    if model_entry.top_k is not None:
        kwargs["top_k"] = model_entry.top_k
    if model_entry.stop_sequences:
        kwargs["stop_sequences"] = model_entry.stop_sequences
    if model_entry.thinking is not None and model_entry.thinking.type is not None:
        thinking: dict[str, Any] = {"type": model_entry.thinking.type}
        if model_entry.thinking.display is not None:
            thinking["display"] = model_entry.thinking.display
        if model_entry.thinking.budget_tokens is not None:
            thinking["budget_tokens"] = model_entry.thinking.budget_tokens
        kwargs["thinking"] = thinking
    if model_entry.effort is not None:
        kwargs["output_config"] = {"effort": model_entry.effort}
    if model_entry.inference_geo is not None:
        kwargs["inference_geo"] = model_entry.inference_geo
    return kwargs


def build_generation_config(model_entry: ModelEntry, system_prompt: str | None) -> GenerationConfig:
    return GenerationConfig(
        provider="anthropic",
        requested_model_id=model_entry.id,
        max_output_tokens=model_entry.max_output_tokens,
        temperature=model_entry.temperature,
        top_p=model_entry.top_p,
        top_k=model_entry.top_k,
        stop_sequences=model_entry.stop_sequences,
        thinking_type=model_entry.thinking.type if model_entry.thinking else None,
        thinking_display=model_entry.thinking.display if model_entry.thinking else None,
        thinking_budget_tokens=model_entry.thinking.budget_tokens if model_entry.thinking else None,
        effort=model_entry.effort,
        inference_geo=model_entry.inference_geo,
        system_prompt=system_prompt,
    )


def classify_error(exc: Exception) -> str:
    """Return "retryable" or "non_retryable" for a caught SDK exception."""
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return "retryable"
    if isinstance(exc, anthropic.APIStatusError):
        return "retryable" if exc.status_code >= 500 else "non_retryable"
    return "non_retryable"


def _status_for_error(exc: Exception) -> CompletionStatus:
    if isinstance(exc, anthropic.RateLimitError):
        return CompletionStatus.RATE_LIMITED
    if isinstance(exc, anthropic.APITimeoutError):
        return CompletionStatus.TIMEOUT
    return CompletionStatus.ERROR


def _parse_usage(usage_obj: Any) -> TokenUsage:
    raw = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj or {})
    return TokenUsage(
        input_tokens=getattr(usage_obj, "input_tokens", None),
        output_tokens=getattr(usage_obj, "output_tokens", None),
        cache_creation_input_tokens=getattr(usage_obj, "cache_creation_input_tokens", None),
        cache_read_input_tokens=getattr(usage_obj, "cache_read_input_tokens", None),
        reasoning_tokens=None,
        raw=raw,
    )


def _extract_text(content_blocks: list[Any]) -> str | None:
    texts = [b.text for b in content_blocks if getattr(b, "type", None) == "text"]
    return "".join(texts) if texts else None


class AnthropicModelClient(ModelClient):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        logger: logging.Logger | None = None,
    ) -> None:
        # max_retries=0: we own retry/backoff explicitly so every attempt is recorded.
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)
        self._max_retries = max_retries
        self._logger = logger or logging.getLogger(__name__)

    def complete(
        self,
        prompt: str,
        model_entry: ModelEntry,
        sample_index: int,
        system_prompt: str | None = None,
    ) -> NormalizedCompletion:
        gen_config = build_generation_config(model_entry, system_prompt)
        kwargs = build_request_kwargs(prompt, model_entry, system_prompt)

        attempt = 0
        start = time.monotonic()
        while True:
            try:
                response = self._client.messages.create(**kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                return self._normalize_success(response, gen_config, retries=attempt, latency_ms=latency_ms)
            except Exception as exc:  # noqa: BLE001 — must capture every provider failure mode
                classification = classify_error(exc)
                self._logger.warning(
                    "anthropic call failed",
                    extra={
                        "attempt": attempt,
                        "classification": classification,
                        "error_type": type(exc).__name__,
                        "model_id": model_entry.id,
                        "sample_index": sample_index,
                    },
                )
                if classification == "non_retryable" or attempt >= self._max_retries:
                    latency_ms = (time.monotonic() - start) * 1000
                    return self._normalize_error(exc, gen_config, retries=attempt, latency_ms=latency_ms)
                time.sleep(self._backoff_delay(attempt, exc))
                attempt += 1

    def _backoff_delay(self, attempt: int, exc: Exception) -> float:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        base = min(2**attempt, DEFAULT_MAX_BACKOFF_SECONDS)
        return base + random.uniform(0, 0.5)

    def _normalize_success(
        self, response: Any, gen_config: GenerationConfig, retries: int, latency_ms: float
    ) -> NormalizedCompletion:
        stop_reason = getattr(response, "stop_reason", None)
        return NormalizedCompletion(
            provider=self.provider,
            requested_model_id=gen_config.requested_model_id,
            served_model_id=getattr(response, "model", None),
            text=_extract_text(response.content),
            request_id=getattr(response, "_request_id", None),
            usage=_parse_usage(response.usage),
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=CompletionStatus.OK,
            stop_reason=stop_reason,
            truncated=stop_reason == "max_tokens",
            retries=retries,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def _normalize_error(
        self, exc: Exception, gen_config: GenerationConfig, retries: int, latency_ms: float
    ) -> NormalizedCompletion:
        return NormalizedCompletion(
            provider=self.provider,
            requested_model_id=gen_config.requested_model_id,
            served_model_id=None,
            text=None,
            request_id=getattr(exc, "request_id", None),
            usage=TokenUsage(),
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=_status_for_error(exc),
            retries=retries,
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
