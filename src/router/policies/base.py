"""Routing policy interfaces.

Two distinct shapes, deliberately not unified:

`Policy` — deployable. `decide()` sees only the prompt (id + text), because
that's all a real router has at inference time, before either model has
been called.

`OraclePolicy` — analysis-only. `decide_batch()` sees `PromptAggregate`s,
which already know both models' measured quality for that prompt. That
information only exists after paying to call both models, so an oracle is
a reference upper bound for the cost-quality frontier, never something you
could actually ship.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from router.aggregation.schemas import PromptAggregate

Role = Literal["cheap", "strong"]


class RoutingDecision(BaseModel):
    prompt_id: str
    policy_name: str
    selected_role: Role
    probability: float | None = None  # P(needs strong), when the policy produces one
    rationale: str | None = None


class Policy(ABC):
    name: str

    @abstractmethod
    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        raise NotImplementedError

    def decide_batch(self, prompts: list[tuple[str, str]]) -> list[RoutingDecision]:
        return [self.decide(prompt_id, text) for prompt_id, text in prompts]


class OraclePolicy(ABC):
    name: str

    @abstractmethod
    def decide_batch(self, aggregates: list[PromptAggregate]) -> list[RoutingDecision]:
        raise NotImplementedError
