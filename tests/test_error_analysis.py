from __future__ import annotations

from router.aggregation.schemas import PromptAggregate
from router.analysis.error_analysis import categorize_errors, group_error_counts, length_bucket
from router.policies.base import RoutingDecision


def _aggregate(prompt_id: str, delta_hat: float | None) -> PromptAggregate:
    return PromptAggregate(
        prompt_id=prompt_id, provider="anthropic", strong_model_id="s", cheap_model_id="c", k=6,
        q_hat_strong=None, q_hat_cheap=None, delta_hat=delta_hat,
    )


def test_categorize_errors_all_cases():
    aggregates = {
        "correct_cheap": _aggregate("correct_cheap", delta_hat=0.1),   # below threshold -> true label 0
        "correct_strong": _aggregate("correct_strong", delta_hat=0.9),  # above threshold -> true label 1
        "false_pos": _aggregate("false_pos", delta_hat=0.1),            # true label 0, predicted strong
        "false_neg": _aggregate("false_neg", delta_hat=0.9),            # true label 1, predicted cheap
        "unlabeled": _aggregate("unlabeled", delta_hat=None),
    }
    decisions = [
        RoutingDecision(prompt_id="correct_cheap", policy_name="p", selected_role="cheap"),
        RoutingDecision(prompt_id="correct_strong", policy_name="p", selected_role="strong"),
        RoutingDecision(prompt_id="false_pos", policy_name="p", selected_role="strong"),
        RoutingDecision(prompt_id="false_neg", policy_name="p", selected_role="cheap"),
        RoutingDecision(prompt_id="unlabeled", policy_name="p", selected_role="cheap"),
    ]

    cases = {c.prompt_id: c for c in categorize_errors(decisions, aggregates, delta_threshold=0.5)}
    assert cases["correct_cheap"].error_type == "correct"
    assert cases["correct_strong"].error_type == "correct"
    assert cases["false_pos"].error_type == "false_positive_strong"
    assert cases["false_neg"].error_type == "false_negative_strong"
    assert cases["unlabeled"].error_type == "unlabeled"


def test_categorize_errors_missing_aggregate_is_unlabeled():
    decisions = [RoutingDecision(prompt_id="no-aggregate", policy_name="p", selected_role="strong")]
    cases = categorize_errors(decisions, aggregates_by_prompt_id={}, delta_threshold=0.5)
    assert cases[0].error_type == "unlabeled"
    assert cases[0].true_label is None


def test_group_error_counts_by_length_bucket():
    from router.analysis.error_analysis import ErrorCase

    cases = [
        ErrorCase(prompt_id="p1", predicted_role="cheap", true_label=0, delta_hat=0.1, error_type="correct"),
        ErrorCase(prompt_id="p2", predicted_role="strong", true_label=0, delta_hat=0.1, error_type="false_positive_strong"),
    ]
    prompt_lengths = {"p1": 10, "p2": 500}
    grouped = group_error_counts(cases, group_key_fn=lambda c: "short" if prompt_lengths[c.prompt_id] < 100 else "long")
    assert grouped["short"] == {"correct": 1}
    assert grouped["long"] == {"false_positive_strong": 1}


def test_length_bucket_edges():
    assert length_bucket("a" * 10) == "<= 50 chars"
    assert length_bucket("a" * 100) == "<= 150 chars"
    assert length_bucket("a" * 500) == "> 300 chars"
