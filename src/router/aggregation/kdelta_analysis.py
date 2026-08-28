"""Offline k / delta sweep.

Directly implements what ADR-0001 promises: any `k` up to the pilot's
collected sample count, and any `delta`, can be evaluated by re-slicing
already-stored records — no regeneration, no new API call.
"""

from __future__ import annotations

from pydantic import BaseModel

from router.aggregation.aggregator import build_prompt_aggregates, compute_label
from router.storage.records import ResponseRecord


class KDeltaSweepPoint(BaseModel):
    k: int
    delta: float
    n_prompts_total: int
    n_prompts_labeled: int
    label_positive_rate: float | None  # fraction of labeled prompts routed to "strong"
    mean_delta_hat: float | None


def sweep_k_delta(
    records: list[ResponseRecord],
    score_by_record_id: dict[str, float],
    prompt_ids: list[str],
    provider: str,
    strong_model_id: str,
    cheap_model_id: str,
    k_values: list[int],
    delta_values: list[float],
) -> list[KDeltaSweepPoint]:
    points: list[KDeltaSweepPoint] = []
    for k in k_values:
        aggregates = build_prompt_aggregates(
            records, score_by_record_id, prompt_ids, provider, strong_model_id, cheap_model_id, k
        )
        known_deltas = [a.delta_hat for a in aggregates if a.delta_hat is not None]
        mean_delta = sum(known_deltas) / len(known_deltas) if known_deltas else None

        for delta_threshold in delta_values:
            labels = [compute_label(a.delta_hat, delta_threshold) for a in aggregates]
            labeled = [label for label in labels if label is not None]
            points.append(
                KDeltaSweepPoint(
                    k=k,
                    delta=delta_threshold,
                    n_prompts_total=len(aggregates),
                    n_prompts_labeled=len(labeled),
                    label_positive_rate=(sum(labeled) / len(labeled)) if labeled else None,
                    mean_delta_hat=mean_delta,
                )
            )
    return points
