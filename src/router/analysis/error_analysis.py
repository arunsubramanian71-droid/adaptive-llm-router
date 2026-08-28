"""Error-analysis framework.

Categorizes each routing decision against a delta-threshold label, then
groups the categorized cases by any caller-supplied key (length bucket,
keyword presence, task type, ...) to surface where a policy goes wrong.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable

from pydantic import BaseModel

from router.aggregation.aggregator import compute_label
from router.aggregation.schemas import PromptAggregate
from router.policies.base import RoutingDecision


class ErrorCase(BaseModel):
    prompt_id: str
    predicted_role: str
    true_label: int | None  # 1 = should have been strong, per the delta threshold
    delta_hat: float | None
    error_type: str  # "correct" | "false_positive_strong" | "false_negative_strong" | "unlabeled"


def categorize_errors(
    decisions: list[RoutingDecision],
    aggregates_by_prompt_id: dict[str, PromptAggregate],
    delta_threshold: float,
) -> list[ErrorCase]:
    cases = []
    for d in decisions:
        aggregate = aggregates_by_prompt_id.get(d.prompt_id)
        true_label = compute_label(aggregate.delta_hat, delta_threshold) if aggregate else None
        predicted = 1 if d.selected_role == "strong" else 0

        if true_label is None:
            error_type = "unlabeled"
        elif predicted == true_label:
            error_type = "correct"
        elif predicted == 1:
            error_type = "false_positive_strong"  # routed strong; cheap would have sufficed
        else:
            error_type = "false_negative_strong"  # routed cheap; strong was needed

        cases.append(
            ErrorCase(
                prompt_id=d.prompt_id,
                predicted_role=d.selected_role,
                true_label=true_label,
                delta_hat=aggregate.delta_hat if aggregate else None,
                error_type=error_type,
            )
        )
    return cases


def group_error_counts(
    cases: list[ErrorCase], group_key_fn: Callable[[ErrorCase], str]
) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for case in cases:
        groups[group_key_fn(case)][case.error_type] += 1
    return {key: dict(counter) for key, counter in groups.items()}


def length_bucket(prompt_text: str, edges: tuple[int, ...] = (50, 150, 300)) -> str:
    length = len(prompt_text)
    for edge in edges:
        if length <= edge:
            return f"<= {edge} chars"
    return f"> {edges[-1]} chars"
