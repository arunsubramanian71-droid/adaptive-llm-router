"""Provider-agnostic normalized types.

These types are the boundary between the router/evaluation system and any
specific provider SDK. A provider adapter's only job is to translate its raw
response into these shapes — nothing downstream (cost, storage, cache)
should ever look at a provider-specific object.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CompletionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class TokenUsage(BaseModel):
    """Normalized usage across providers.

    Not every provider exposes every field — leave a field `None` rather
    than guessing a value. `raw` preserves whatever the provider actually
    returned, verbatim, so nothing is lost even if this schema doesn't yet
    have a named field for it.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        parts = [self.input_tokens, self.output_tokens]
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)


class GenerationConfig(BaseModel):
    """The generation-time configuration used for one call.

    This is persisted verbatim alongside every response record so that any
    later analysis can group/filter by exact generation conditions without
    needing to re-derive them.
    """

    provider: str
    requested_model_id: str
    max_output_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    thinking_type: str | None = None
    thinking_display: str | None = None
    thinking_budget_tokens: int | None = None
    effort: str | None = None
    inference_geo: str | None = None
    system_prompt: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class NormalizedCompletion(BaseModel):
    """A single normalized model call result — success or failure."""

    provider: str
    requested_model_id: str
    served_model_id: str | None = None

    text: str | None = None
    request_id: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float | None = None
    timestamp_utc: datetime

    generation_config: GenerationConfig

    status: CompletionStatus
    stop_reason: str | None = None
    truncated: bool = False

    retries: int = 0
    error_type: str | None = None
    error_message: str | None = None

    raw_response: dict[str, Any] | None = None
