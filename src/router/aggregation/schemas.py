"""Per-prompt aggregate: q_hat per model role and the resulting Delta.

Deliberately does NOT carry a fixed `label` — the routing label depends on
a delta threshold that is swept over multiple candidate values (ADR-0001),
so `delta_hat` is the durable quantity and `compute_label()` derives a
label for a chosen threshold on demand, purely from what's already stored
here (no re-scoring, no API call).
"""

from __future__ import annotations

from pydantic import BaseModel


class PromptAggregate(BaseModel):
    prompt_id: str
    provider: str
    strong_model_id: str
    cheap_model_id: str
    k: int

    q_hat_strong: float | None = None
    q_hat_cheap: float | None = None
    delta_hat: float | None = None

    n_samples_strong: int = 0
    n_samples_cheap: int = 0
