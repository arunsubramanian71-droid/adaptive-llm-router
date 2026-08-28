"""Aggregation of response-level scores into q_hat / Delta / labels.

Everything here reads only from already-stored `ResponseRecord`s and
score-bearing results (`EvalResult` or `JudgeVerdict` — anything with a
`.record_id` and `.score`) — no model call, ever. That's the whole point:
k and delta can be swept offline (see `kdelta_analysis.py`) by re-slicing
what's already on disk.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from router.aggregation.schemas import PromptAggregate
from router.storage.records import ResponseRecord


@runtime_checkable
class ScoredResult(Protocol):
    record_id: str
    score: float


def build_score_lookup(scored_results: list[ScoredResult]) -> dict[str, float]:
    """Last write wins if the same record_id appears twice (e.g. two
    evaluators scoring the same record) — callers wanting a specific
    evaluator's score should filter before calling this."""
    return {r.record_id: r.score for r in scored_results}


def aggregate_q_hat(
    records: list[ResponseRecord],
    score_by_record_id: dict[str, float],
    prompt_id: str,
    model_id: str,
    k: int,
) -> tuple[float | None, int]:
    """Mean score across up to `k` samples (by `sample_index < k`) of
    `model_id` for `prompt_id`. Only successful (`status == "ok"`) records
    with a known score are counted. Returns (q_hat, n_samples_used)."""
    scores = [
        score_by_record_id[r.record_id]
        for r in records
        if r.prompt_id == prompt_id
        and r.requested_model_id == model_id
        and r.sample_index < k
        and r.status == "ok"
        and r.record_id in score_by_record_id
    ]
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def compute_delta(q_hat_strong: float | None, q_hat_cheap: float | None) -> float | None:
    if q_hat_strong is None or q_hat_cheap is None:
        return None
    return q_hat_strong - q_hat_cheap


def compute_label(delta_hat: float | None, delta_threshold: float) -> int | None:
    """1 = route to strong (the quality gap exceeds the threshold), 0 =
    cheap is sufficient. None if delta_hat isn't known (missing samples)."""
    if delta_hat is None:
        return None
    return int(delta_hat > delta_threshold)


def build_prompt_aggregates(
    records: list[ResponseRecord],
    score_by_record_id: dict[str, float],
    prompt_ids: list[str],
    provider: str,
    strong_model_id: str,
    cheap_model_id: str,
    k: int,
) -> list[PromptAggregate]:
    aggregates = []
    for prompt_id in prompt_ids:
        q_strong, n_strong = aggregate_q_hat(records, score_by_record_id, prompt_id, strong_model_id, k)
        q_cheap, n_cheap = aggregate_q_hat(records, score_by_record_id, prompt_id, cheap_model_id, k)
        aggregates.append(
            PromptAggregate(
                prompt_id=prompt_id,
                provider=provider,
                strong_model_id=strong_model_id,
                cheap_model_id=cheap_model_id,
                k=k,
                q_hat_strong=q_strong,
                q_hat_cheap=q_cheap,
                delta_hat=compute_delta(q_strong, q_cheap),
                n_samples_strong=n_strong,
                n_samples_cheap=n_cheap,
            )
        )
    return aggregates
