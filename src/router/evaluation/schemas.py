"""Response-level evaluation result.

One `EvalResult` per (`record_id`, `evaluator_name`) — mirrors Stage 0's
response-level (never aggregated) persistence philosophy. Aggregation into
q_hat happens later, in `router.aggregation`, purely by reading these back.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    record_id: str  # ResponseRecord.record_id this score belongs to
    prompt_id: str
    evaluator_name: str

    score: float  # normalized to [0, 1]; 1.0 = fully correct/satisfied
    passed: bool | None = None

    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
