"""Google Gemini provider adapter.

Translates between the provider-agnostic `ModelClient` interface and the
`google-genai` SDK's Interactions API (`client.interactions.create`). This
is the ONLY module in the codebase allowed to import `google.genai` or
know about its response shapes.

API contract verified 2026-08-25 against current official Gemini
documentation (ai.google.dev/gemini-api/docs/{quickstart,text-generation,
thinking,api-key}) and cross-checked directly against the installed
`google-genai` package's actual types/exception classes — this adapter
deliberately does NOT reuse Anthropic's or OpenAI's shapes where Gemini's
genuinely differ:

- Endpoint: ai.google.dev's product docs state "The Interactions API is
  now generally available. We recommend using this API for access to all
  the latest features and models" — used here in preference to the older
  `client.models.generate_content`, which the installed SDK still exposes
  but which the current docs no longer lead with.
- System prompt: top-level `system_instruction=` (not nested in a
  messages list, not `instructions=` like OpenAI, not `system=` like
  Anthropic).
- Generation settings (`temperature`, `max_output_tokens`, ...) nest under
  a `generation_config={}` dict, not top-level kwargs.
- Reasoning control: `generation_config={"thinking_level": "low"|"medium"|"high"}`
  — only 3 levels, unlike `router.config.EffortLevel`'s 5
  (`low|medium|high|xhigh|max`). `xhigh`/`max` are clamped down to
  `"high"` here since Gemini has no higher level to map them onto.
- Usage: `response.usage.{total_input_tokens, total_output_tokens,
  total_thought_tokens, total_cached_tokens, total_tokens}` — confirmed
  directly against the installed SDK's `Usage` model fields. Unlike
  OpenAI, `total_thought_tokens` is a genuinely SEPARATE, additive count
  from `total_output_tokens`, not a breakdown within it (confirmed via
  Google's own pricing guidance: a response with a small visible answer
  but heavy internal reasoning bills for both combined). Because of that,
  `configs/pricing.yaml` sets Gemini's `reasoning_per_mtok` equal to its
  `output_per_mtok` (thinking tokens are billed at the standard output
  rate) — the opposite of OpenAI's config, where no `reasoning_per_mtok`
  is set at all. Getting this wrong either way would double- or
  under-count real cost; see `router.cost.calculator`.
- Response shape / error handling: the Interaction object's own `status`
  field (`"completed" | "failed" | "cancelled" | "incomplete" |
  "budget_exceeded" | ...`) can indicate failure *without the SDK raising
  a Python exception* — a genuinely different failure-reporting shape
  from Anthropic/OpenAI, where a bad HTTP status always raises. This
  adapter treats `status in ("failed", "cancelled")` as
  `CompletionStatus.ERROR` even though no exception was thrown.
- Error hierarchy: `google.genai.errors.APIError` (base, `.code` is the
  HTTP status int) with `ClientError` (4xx) / `ServerError` (5xx)
  subclasses — no separate `RateLimitError`/`AuthenticationError` classes
  the way Anthropic/OpenAI have; a 429 is just a `ClientError` with
  `.code == 429`. Confirmed directly against the installed package.
- Request id: the Interaction's own `id` field.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

from google.genai import Client
from google.genai import errors as genai_errors

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

_FAILED_STATUSES = frozenset({"failed", "cancelled"})
_TRUNCATED_STATUSES = frozenset({"incomplete", "budget_exceeded"})

# Gemini's thinking_level has 3 rungs; router.config.EffortLevel has 5.
# xhigh/max both clamp to "high" -- there's nothing higher to map them onto.
_EFFORT_TO_THINKING_LEVEL: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def build_request_kwargs(
    prompt: str,
    model_entry: ModelEntry,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Pure translation of our config into `interactions.create` kwargs.
    Kept separate from the client so it's unit-testable without a network
    call or an API key."""
    kwargs: dict[str, Any] = {"model": model_entry.id, "input": prompt}
    if system_prompt:
        kwargs["system_instruction"] = system_prompt

    generation_config: dict[str, Any] = {"max_output_tokens": model_entry.max_output_tokens}
    if model_entry.temperature is not None:
        generation_config["temperature"] = model_entry.temperature
    if model_entry.top_p is not None:
        generation_config["top_p"] = model_entry.top_p
    if model_entry.top_k is not None:
        generation_config["top_k"] = model_entry.top_k
    if model_entry.effort is not None:
        generation_config["thinking_level"] = _EFFORT_TO_THINKING_LEVEL[model_entry.effort]
    kwargs["generation_config"] = generation_config
    # model_entry.stop_sequences and .thinking (Anthropic-shaped) and
    # .inference_geo are intentionally not sent -- no verified Gemini
    # Interactions API equivalent for the first and third; the second is a
    # different provider's config shape entirely.
    return kwargs


def build_generation_config(model_entry: ModelEntry, system_prompt: str | None) -> GenerationConfig:
    return GenerationConfig(
        provider="google",
        requested_model_id=model_entry.id,
        max_output_tokens=model_entry.max_output_tokens,
        temperature=model_entry.temperature,
        top_p=model_entry.top_p,
        top_k=model_entry.top_k,
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
    if isinstance(exc, genai_errors.ServerError):
        return "retryable"
    if isinstance(exc, genai_errors.ClientError):
        return "retryable" if exc.code == 429 else "non_retryable"
    return "non_retryable"


def _status_for_error(exc: Exception) -> CompletionStatus:
    code = getattr(exc, "code", None)
    if code == 429:
        return CompletionStatus.RATE_LIMITED
    if code == 408:
        return CompletionStatus.TIMEOUT
    return CompletionStatus.ERROR


def _parse_usage(usage_obj: Any) -> TokenUsage:
    if usage_obj is None:
        return TokenUsage()
    raw = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)
    return TokenUsage(
        input_tokens=getattr(usage_obj, "total_input_tokens", None),
        output_tokens=getattr(usage_obj, "total_output_tokens", None),
        cache_creation_input_tokens=None,  # Gemini context caching is a separate explicit-cache feature, not this
        cache_read_input_tokens=getattr(usage_obj, "total_cached_tokens", None),
        reasoning_tokens=getattr(usage_obj, "total_thought_tokens", None),
        raw=raw,
    )


class GoogleGeminiModelClient(ModelClient):
    provider = "google"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,  # accepted for interface parity; the SDK has no first-class base_url override
        max_retries: int = DEFAULT_MAX_RETRIES,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = Client(api_key=api_key)
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
                interaction = self._client.interactions.create(**kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                return self._normalize_response(interaction, gen_config, retries=attempt, latency_ms=latency_ms)
            except Exception as exc:  # noqa: BLE001 — must capture every provider failure mode
                classification = classify_error(exc)
                self._logger.warning(
                    "gemini call failed",
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
        base = min(2**attempt, DEFAULT_MAX_BACKOFF_SECONDS)
        return base + random.uniform(0, 0.5)

    def _normalize_response(
        self, interaction: Any, gen_config: GenerationConfig, retries: int, latency_ms: float
    ) -> NormalizedCompletion:
        status_value = getattr(interaction, "status", None)
        raw_response = interaction.model_dump() if hasattr(interaction, "model_dump") else None

        if status_value in _FAILED_STATUSES:
            errors = getattr(interaction, "errors", None) or []
            error_message = "; ".join(str(e) for e in errors) or f"interaction status={status_value}"
            return NormalizedCompletion(
                provider=self.provider,
                requested_model_id=gen_config.requested_model_id,
                served_model_id=getattr(interaction, "model", None),
                text=None,
                request_id=getattr(interaction, "id", None),
                usage=_parse_usage(getattr(interaction, "usage", None)),
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(UTC),
                generation_config=gen_config,
                status=CompletionStatus.ERROR,
                stop_reason=status_value,
                retries=retries,
                error_type=f"gemini_status_{status_value}",
                error_message=error_message[:2000],
                raw_response=raw_response,
            )

        return NormalizedCompletion(
            provider=self.provider,
            requested_model_id=gen_config.requested_model_id,
            served_model_id=getattr(interaction, "model", None),
            text=getattr(interaction, "output_text", None),
            request_id=getattr(interaction, "id", None),
            usage=_parse_usage(getattr(interaction, "usage", None)),
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=CompletionStatus.OK,
            stop_reason=status_value,
            truncated=status_value in _TRUNCATED_STATUSES,
            retries=retries,
            raw_response=raw_response,
        )

    def _normalize_error(
        self, exc: Exception, gen_config: GenerationConfig, retries: int, latency_ms: float
    ) -> NormalizedCompletion:
        return NormalizedCompletion(
            provider=self.provider,
            requested_model_id=gen_config.requested_model_id,
            served_model_id=None,
            text=None,
            request_id=None,
            usage=TokenUsage(),
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(UTC),
            generation_config=gen_config,
            status=_status_for_error(exc),
            retries=retries,
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
