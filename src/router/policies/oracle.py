from __future__ import annotations

from typing import cast

from router.aggregation.schemas import PromptAggregate
from router.policies.base import OraclePolicy, Role, RoutingDecision


class QualityMaximizingOracle(OraclePolicy):
    """Picks whichever model measured higher q_hat for each prompt — the
    best quality achievable by routing at all, ignoring cost entirely.
    Ties (including "only one model has data") resolve to cheap, since
    quality is equal and cheap is cost-preferable."""

    name = "oracle_quality_maximizing"

    def decide_batch(self, aggregates: list[PromptAggregate]) -> list[RoutingDecision]:
        decisions = []
        for a in aggregates:
            if a.q_hat_strong is None and a.q_hat_cheap is None:
                continue
            q_strong = a.q_hat_strong if a.q_hat_strong is not None else float("-inf")
            q_cheap = a.q_hat_cheap if a.q_hat_cheap is not None else float("-inf")
            role: Role = "strong" if q_strong > q_cheap else "cheap"
            decisions.append(
                RoutingDecision(
                    prompt_id=a.prompt_id,
                    policy_name=self.name,
                    selected_role=role,
                    rationale=f"q_hat_strong={a.q_hat_strong}, q_hat_cheap={a.q_hat_cheap}",
                )
            )
        return decisions


class CostConstrainedOracle(OraclePolicy):
    """Given a budget on the fraction of prompts allowed to go to strong,
    greedily routes the prompts with the largest measured delta_hat to
    strong first. A Pareto-optimal reference point for a given budget,
    since it spends the strong-model budget where it helps quality most."""

    name = "oracle_cost_constrained"

    def __init__(self, strong_fraction_budget: float) -> None:
        if not 0.0 <= strong_fraction_budget <= 1.0:
            raise ValueError("strong_fraction_budget must be in [0, 1]")
        self._budget = strong_fraction_budget

    def decide_batch(self, aggregates: list[PromptAggregate]) -> list[RoutingDecision]:
        eligible = [a for a in aggregates if a.delta_hat is not None]
        eligible_sorted = sorted(eligible, key=lambda a: cast(float, a.delta_hat), reverse=True)
        n_strong = round(self._budget * len(eligible_sorted))
        strong_ids = {a.prompt_id for a in eligible_sorted[:n_strong]}

        return [
            RoutingDecision(
                prompt_id=a.prompt_id,
                policy_name=self.name,
                selected_role="strong" if a.prompt_id in strong_ids else "cheap",
                rationale=f"delta_hat={a.delta_hat}, strong_fraction_budget={self._budget}",
            )
            for a in eligible_sorted
        ]
