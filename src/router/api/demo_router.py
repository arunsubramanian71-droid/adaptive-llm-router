from __future__ import annotations

from router.api.demo_fixtures import demo_training_data
from router.policies.router_policy import RouterPolicy
from router.routers.tfidf_logreg import TfidfLogisticRegressionRouter

DEMO_DISCLAIMER = (
    "DEMO ONLY: this router is fit on a small set of synthetic, hand-written example "
    "prompts bundled with the repo, not on any real model response, evaluation score, "
    "or benchmark data. Its routing decisions, probabilities, and estimated costs "
    "demonstrate the API's shape only — they are not a claim about real quality or "
    "real cost savings. Estimated cost is derived from a rough word-count token guess, "
    "not from provider-reported usage (no live model call is made by this demo)."
)


def build_demo_router_policy(tau: float = 0.5) -> RouterPolicy:
    prompts, labels = demo_training_data()
    router = TfidfLogisticRegressionRouter()
    router.fit(prompts, labels)
    return RouterPolicy(router, tau=tau, name="demo_tfidf_logreg")
