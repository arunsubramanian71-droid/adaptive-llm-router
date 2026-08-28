"""Adapter turning a fitted `Router` + threshold tau into a deployable
`Policy`, so learned routers can be evaluated side-by-side with the
baselines/oracles through the same threshold-sweep and metrics code."""

from __future__ import annotations

from router.policies.base import Policy, RoutingDecision
from router.routers.base import Router


class RouterPolicy(Policy):
    def __init__(self, router: Router, tau: float, name: str | None = None) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        self._router = router
        self._tau = tau
        self.name = name or f"router:{router.name}@tau={tau:.2f}"

    def decide(self, prompt_id: str, prompt_text: str) -> RoutingDecision:
        return self.decide_batch([(prompt_id, prompt_text)])[0]

    def decide_batch(self, prompts: list[tuple[str, str]]) -> list[RoutingDecision]:
        if not prompts:
            return []
        texts = [text for _, text in prompts]
        probs = self._router.predict_proba(texts)
        return [
            RoutingDecision(
                prompt_id=prompt_id,
                policy_name=self.name,
                selected_role="strong" if prob >= self._tau else "cheap",
                probability=prob,
                rationale=f"prob={prob:.4f} vs tau={self._tau:.4f}",
            )
            for (prompt_id, _), prob in zip(prompts, probs, strict=True)
        ]
