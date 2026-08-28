from __future__ import annotations

import pytest

from router.aggregation.schemas import PromptAggregate
from router.policies import (
    AlwaysCheapPolicy,
    AlwaysStrongPolicy,
    CostConstrainedOracle,
    KeywordHeuristicPolicy,
    LengthHeuristicPolicy,
    QualityMaximizingOracle,
    RandomMatchedRatePolicy,
    RouterPolicy,
)
from router.routers import TfidfLogisticRegressionRouter


def test_always_cheap_and_strong():
    assert AlwaysCheapPolicy().decide("p1", "anything").selected_role == "cheap"
    assert AlwaysStrongPolicy().decide("p1", "anything").selected_role == "strong"


def test_random_matched_rate_is_seeded_deterministic():
    p1 = RandomMatchedRatePolicy(strong_rate=0.5, seed=42)
    p2 = RandomMatchedRatePolicy(strong_rate=0.5, seed=42)
    decisions1 = [p1.decide(f"p{i}", "x").selected_role for i in range(20)]
    decisions2 = [p2.decide(f"p{i}", "x").selected_role for i in range(20)]
    assert decisions1 == decisions2


def test_random_matched_rate_approximately_matches_rate():
    policy = RandomMatchedRatePolicy(strong_rate=0.3, seed=1)
    decisions = [policy.decide(f"p{i}", "x").selected_role for i in range(2000)]
    strong_fraction = decisions.count("strong") / len(decisions)
    assert abs(strong_fraction - 0.3) < 0.05


def test_random_matched_rate_rejects_invalid_rate():
    with pytest.raises(ValueError):
        RandomMatchedRatePolicy(strong_rate=1.5)


def test_length_heuristic():
    policy = LengthHeuristicPolicy(char_threshold=10)
    assert policy.decide("p1", "short").selected_role == "cheap"
    assert policy.decide("p1", "this prompt is definitely long enough").selected_role == "strong"


def test_keyword_heuristic():
    policy = KeywordHeuristicPolicy()
    assert policy.decide("p1", "what's 2+2?").selected_role == "cheap"
    assert policy.decide("p1", "please prove this theorem").selected_role == "strong"


def _aggregates():
    return [
        PromptAggregate(
            prompt_id="p1", provider="anthropic", strong_model_id="s", cheap_model_id="c", k=6,
            q_hat_strong=0.9, q_hat_cheap=0.9, delta_hat=0.0,
        ),
        PromptAggregate(
            prompt_id="p2", provider="anthropic", strong_model_id="s", cheap_model_id="c", k=6,
            q_hat_strong=1.0, q_hat_cheap=0.2, delta_hat=0.8,
        ),
        PromptAggregate(
            prompt_id="p3", provider="anthropic", strong_model_id="s", cheap_model_id="c", k=6,
            q_hat_strong=0.7, q_hat_cheap=0.65, delta_hat=0.05,
        ),
        PromptAggregate(
            prompt_id="p4", provider="anthropic", strong_model_id="s", cheap_model_id="c", k=6,
            q_hat_strong=None, q_hat_cheap=None, delta_hat=None,
        ),
    ]


def test_quality_maximizing_oracle_ties_go_cheap():
    decisions = {d.prompt_id: d for d in QualityMaximizingOracle().decide_batch(_aggregates())}
    assert decisions["p1"].selected_role == "cheap"  # tie
    assert decisions["p2"].selected_role == "strong"  # strong clearly better
    assert decisions["p3"].selected_role == "strong"  # strong marginally better
    assert "p4" not in decisions  # no data at all — oracle can't decide


def test_cost_constrained_oracle_respects_budget():
    aggregates = _aggregates()
    # 3 eligible prompts (p4 excluded, no delta_hat); budget 1/3 -> exactly 1 routed to strong
    oracle = CostConstrainedOracle(strong_fraction_budget=1 / 3)
    decisions = {d.prompt_id: d for d in oracle.decide_batch(aggregates)}
    assert sum(d.selected_role == "strong" for d in decisions.values()) == 1
    # p2 has the largest delta_hat (0.8) so it must be the one routed to strong.
    assert decisions["p2"].selected_role == "strong"
    assert "p4" not in decisions


def test_cost_constrained_oracle_zero_budget_routes_nothing_to_strong():
    oracle = CostConstrainedOracle(strong_fraction_budget=0.0)
    decisions = oracle.decide_batch(_aggregates())
    assert all(d.selected_role == "cheap" for d in decisions)


def test_cost_constrained_oracle_full_budget_routes_everything_eligible_to_strong():
    oracle = CostConstrainedOracle(strong_fraction_budget=1.0)
    decisions = oracle.decide_batch(_aggregates())
    assert all(d.selected_role == "strong" for d in decisions)


def test_cost_constrained_oracle_rejects_invalid_budget():
    with pytest.raises(ValueError):
        CostConstrainedOracle(strong_fraction_budget=1.5)


def test_router_policy_thresholds_a_fitted_router(synthetic_labeled_prompts):
    prompts, labels = synthetic_labeled_prompts
    router = TfidfLogisticRegressionRouter()
    router.fit(prompts, labels)

    policy = RouterPolicy(router, tau=0.5)
    hard = policy.decide("p1", "Prove that this algorithm terminates for every input.")
    easy = policy.decide("p2", "What is the capital of France?")

    assert hard.selected_role == "strong"
    assert easy.selected_role == "cheap"
    assert hard.probability is not None


def test_router_policy_decide_batch_matches_decide(synthetic_labeled_prompts):
    prompts, labels = synthetic_labeled_prompts
    router = TfidfLogisticRegressionRouter()
    router.fit(prompts, labels)
    policy = RouterPolicy(router, tau=0.5)

    batch_prompts = [("p1", prompts[0]), ("p2", prompts[1])]
    batch_decisions = policy.decide_batch(batch_prompts)
    individual_decisions = [policy.decide(pid, text) for pid, text in batch_prompts]

    assert [d.selected_role for d in batch_decisions] == [d.selected_role for d in individual_decisions]


def test_router_policy_extreme_tau():
    class ConstantProbRouter:
        name = "constant"

        def predict_proba(self, prompts):
            return [0.6 for _ in prompts]

    always_strong = RouterPolicy(ConstantProbRouter(), tau=0.0)
    always_cheap = RouterPolicy(ConstantProbRouter(), tau=1.0)
    assert always_strong.decide("p1", "x").selected_role == "strong"
    assert always_cheap.decide("p1", "x").selected_role == "cheap"


def test_router_policy_rejects_invalid_tau():
    with pytest.raises(ValueError):
        RouterPolicy(TfidfLogisticRegressionRouter(), tau=1.5)
