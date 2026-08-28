"""Deployable baseline policies — no model call, no learning."""

from __future__ import annotations

import random

from router.policies.base import Policy, Role, RoutingDecision


class AlwaysCheapPolicy(Policy):
    name = "always_cheap"

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        return RoutingDecision(prompt_id=prompt_id, policy_name=self.name, selected_role="cheap", probability=0.0)


class AlwaysStrongPolicy(Policy):
    name = "always_strong"

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        return RoutingDecision(prompt_id=prompt_id, policy_name=self.name, selected_role="strong", probability=1.0)


class RandomMatchedRatePolicy(Policy):
    """Routes to strong with fixed probability `strong_rate`, independent
    of prompt content — a null-model baseline for matching a target
    strong-routing rate without using any signal."""

    name = "random_matched_rate"

    def __init__(self, strong_rate: float, seed: int = 0) -> None:
        if not 0.0 <= strong_rate <= 1.0:
            raise ValueError("strong_rate must be in [0, 1]")
        self._rate = strong_rate
        self._rng = random.Random(seed)

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        draw = self._rng.random()
        role: Role = "strong" if draw < self._rate else "cheap"
        return RoutingDecision(
            prompt_id=prompt_id,
            policy_name=self.name,
            selected_role=role,
            probability=self._rate,
            rationale=f"draw={draw:.4f} vs rate={self._rate}",
        )


class LengthHeuristicPolicy(Policy):
    """Routes to strong when the prompt exceeds a character-length
    threshold — the "longer prompts are harder" heuristic."""

    name = "length_heuristic"

    def __init__(self, char_threshold: int = 200) -> None:
        self._threshold = char_threshold

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        length = len(prompt_text)
        role: Role = "strong" if length > self._threshold else "cheap"
        return RoutingDecision(
            prompt_id=prompt_id,
            policy_name=self.name,
            selected_role=role,
            rationale=f"length={length} chars vs threshold={self._threshold}",
        )


DEFAULT_HARD_KEYWORDS = (
    "prove",
    "derive",
    "optimi",
    "algorithm",
    "debug",
    "refactor",
    "theorem",
    "counterexample",
)


class KeywordHeuristicPolicy(Policy):
    """Routes to strong when the prompt contains any of a fixed set of
    "this looks hard" keywords."""

    name = "keyword_heuristic"

    def __init__(self, keywords: tuple[str, ...] = DEFAULT_HARD_KEYWORDS) -> None:
        self._keywords = keywords

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        text_lower = prompt_text.lower()
        matched = [kw for kw in self._keywords if kw in text_lower]
        role: Role = "strong" if matched else "cheap"
        return RoutingDecision(
            prompt_id=prompt_id,
            policy_name=self.name,
            selected_role=role,
            rationale=f"matched_keywords={matched}",
        )
