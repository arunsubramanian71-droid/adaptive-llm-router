"""Judge interface.

Mirrors `router.models.base.ModelClient` deliberately: a judge is just
another model call (usually to a strong model) that scores a response
against a rubric instead of answering a prompt. Keeping the same shape
means a real LLM-judge adapter can later reuse the Anthropic adapter's
request/retry/usage-parsing machinery — that adapter is not built here
because it would require real API calls; Stage 1 ships the interface plus
mock implementations for pipeline testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    record_id: str
    prompt_id: str
    judge_name: str

    score: float  # normalized to [0, 1]
    rationale: str | None = None

    raw_response: dict[str, Any] | None = None
    error: str | None = None


class JudgeClient(ABC):
    name: str

    @abstractmethod
    def judge(
        self,
        prompt_id: str,
        record_id: str,
        prompt: str,
        response_text: str | None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeVerdict:
        """Score one response. Must never raise for a missing/malformed
        response — return a JudgeVerdict with score=0.0 and `error` set."""
        raise NotImplementedError

    def _empty_response_verdict(self, prompt_id: str, record_id: str) -> JudgeVerdict:
        return JudgeVerdict(
            record_id=record_id,
            prompt_id=prompt_id,
            judge_name=self.name,
            score=0.0,
            error="empty or missing response text",
        )


class JudgeConfig(BaseModel):
    """Placeholder for a future real judge's generation configuration —
    kept here so `evaluation/judge/pipeline.py` and any persisted verdicts
    don't need to change shape once a real judge adapter exists."""

    judge_model_id: str
    rubric_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
