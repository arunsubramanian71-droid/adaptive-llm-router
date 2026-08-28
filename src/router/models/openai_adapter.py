"""OpenAI provider adapter.

Translates between the provider-agnostic `ModelClient` interface and the
`openai` SDK's Responses API (`client.responses.create`). This is the ONLY
module in the codebase allowed to import `openai` or know about its
response shapes — mirrors `router.models.anthropic_adapter`'s role for
Anthropic.

API contract verified 2026-08-25 against current official OpenAI
documentation (developers.openai.com/api/docs/{quickstart,pricing,
guides/reasoning,guides/error-codes}) and cross-checked against the
installed `openai` package's actual exception classes:
- Responses API: `client.responses.create(model=..., input=..., ...)`,
  output text via the `response.output_text` convenience property.
- System/developer guidance: top-level `instructions=` parameter (distinct
  from Anthropic's `system=` and Gemini's `system_instruction=`).
- Reasoning control: `reasoning={"effort": ...}` — the *same* five-level
  vocabulary (`low|medium|high|xhigh|max`) `router.config.EffortLevel`
  already uses, so `ModelEntry.effort` maps straight across with no
  translation, unlike Gemini (see google_gemini_adapter.py).
- Usage: `response.usage.{input_tokens, output_tokens, total_tokens,
  input_tokens_details.cached_tokens, output_tokens_details.reasoning_tokens}`.
  `reasoning_tokens` here is a **breakdown of** `output_tokens`, not
  additional to it (confirmed via a documented example where
  input_tokens + output_tokens == total_tokens) — the same convention
  Anthropic uses, just with the breakdown exposed. `configs/pricing.yaml`
  must NOT set a `reasoning_per_mtok` for OpenAI models, or reasoning
  tokens would be billed twice (once inside output_cost, once as
  reasoning_cost).
- Request id: `response.id`.
- Error hierarchy: `openai.APIStatusError` (base for 4xx/5xx, has
  `.status_code`) with typed subclasses `RateLimitError` (429),
  `AuthenticationError` (401), etc. — structurally identical to
  Anthropic's (both SDKs are Stainless-generated), verified directly
  against the installed package, not assumed from that similarity alone.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

import openai

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
    """Pure translation of our config into `responses.create` kwargs.
    Kept separate from the client so it's unit-testable without a network
    call or an API key."""
    kwargs: dict[str, Any] = {
        "model": model_entry.id,
        "input": prompt,
        "max_output_tokens": model_entry.max_output_tokens,
    }
    if system_prompt:
        kwargs["instructions"] = system_prompt
    if model_entry.temperature is not None:
        kwargs["temperature"] = model_entry.temperature
    if model_entry.top_p is not None:
        kwargs["top_p"] = model_entry.top_p
    if model_entry.effort is not None:
        kwargs["reasoning"] = {"effort": model_entry.effort}
    # model_entry.top_k and .stop_sequences are intentionally not sent: the
    # Responses API does not document equivalents for either as of this
    # verification pass. Anthropic-style thinking (.thinking) and
    # .inference_geo are likewise Anthropic-specific and not used here.
    return kwargs


def build_generation_config(model_entry: ModelEntry, system_prompt: str | None) -> GenerationConfig:
    return GenerationConfig(
        provider="openai",
        requested_model_id=model_entry.id,
        max_output_tokens=model_entry.max_output_tokens,
        temperature=model_entry.temperature,
        top_p=model_entry.top_p,
        top_k=None,
        stop_sequences=[],
        thinking_type=None,
        thinking_display=None,
        thinking_budget_tokens=None,
        effort=model_entry.effort,
        inference_geo=None,
        system_prompt=system_prompt,
    )


def classify_error(exc: Exception) -> str:
    """Return "retryable" or "non_retryable" for a caught SDK exception."""
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return "retryable"
    if isinstance(exc, openai.APIStatusError):
        return "retryable" if exc.status_code >= 500 else "non_retryable"
    return "non_retryable"


def _status_for_error(exc: Exception) -> CompletionStatus:
    if isinstance(exc, openai.RateLimitError):
        return CompletionStatus.RATE_LIMITED
    if isinstance(exc, openai.APITimeoutError):
        return CompletionStatus.TIMEOUT
    return CompletionStatus.ERROR


def _parse_usage(usage_obj: Any) -> TokenUsage:
    raw = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj or {})
    input_details = getattr(usage_obj, "input_tokens_details", None)
    output_details = getattr(usage_obj, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(usage_obj, "input_tokens", None),
        output_tokens=getattr(usage_obj, "output_tokens", None),
        cache_creation_input_tokens=None,  # OpenAI's automatic caching exposes no separate "write" charge here
        cache_read_input_tokens=getattr(input_details, "cached_tokens", None) if input_details else None,
        # Diagnostic only -- already counted inside output_tokens, see module docstring.
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None) if output_details else None,
        raw=raw,
    )


class OpenAIModelClient(ModelClient):
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        logger: logging.Logger | None = None,
    ) -> None:
        # max_retries=0: we own retry/backoff explicitly so every attempt is recorded.
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
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
                response = self._client.responses.create(**kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                return self._normalize_success(response, gen_config, retries=attempt, latency_ms=latency_ms)
            except Exception as exc:  # noqa: BLE001 — must capture every provider failure mode
                classification = classify_error(exc)
                self._logger.warning(
                    "openai call failed",
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
        status_value = getattr(response, "status", None)
        return NormalizedCompletion(
            provider=self.provider,
            requested_model_id=gen_config.requested_model_id,
            served_model_id=getattr(response, "model", None),
            text=getattr(response, "output_text", None),
            request_id=getattr(response, "id", None),
            usage=_parse_usage(response.usage),
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=CompletionStatus.OK,
            stop_reason=status_value,
            truncated=status_value == "incomplete",
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
