"""FastAPI demo interface.

Shows the shape of a real router deployment (routing decision + calibrated
probability + selected model role + estimated cost) without ever calling a
real model. See `DEMO_DISCLAIMER` — the router behind this app is fit on
synthetic bundled prompts, not real data, and never should be mistaken for
a real trained router.

Run with (after `pip install -e ".[api]"`):

    uvicorn router.api.app:app --reload
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException

from router.api.demo_router import DEMO_DISCLAIMER, build_demo_router_policy
from router.api.schemas import InfoResponse, RouteRequest, RouteResponse
from router.config import AppConfig, ConfigError, load_app_config
from router.cost.calculator import calculate_cost
from router.models.schemas import TokenUsage

DEFAULT_TAU = 0.5
DEMO_ASSUMED_OUTPUT_TOKENS = 200


def estimate_prompt_tokens(prompt: str) -> TokenUsage:
    """Rough word-count-based token guess for a cost *preview* in the demo
    UI — never used for real billing reconciliation (see
    `router.cost.calculator`, which only ever uses provider-reported
    usage)."""
    words = len(prompt.split())
    input_tokens = max(1, round(words * 1.3))
    return TokenUsage(input_tokens=input_tokens, output_tokens=DEMO_ASSUMED_OUTPUT_TOKENS)


def select_role_model_ids(config: AppConfig, provider: str) -> tuple[str, str]:
    provider_cfg = config.models.providers.get(provider)
    if provider_cfg is None:
        raise ConfigError(f"no models configured for provider {provider!r}")
    strong_id = next((m.id for m in provider_cfg.models if m.role_hint == "candidate_strong"), None)
    cheap_id = next((m.id for m in provider_cfg.models if m.role_hint == "candidate_cheap"), None)
    if strong_id is None or cheap_id is None:
        raise ConfigError(
            f"configs/models.yaml must have one model with role_hint=candidate_strong and one "
            f"with role_hint=candidate_cheap under provider {provider!r} for the demo app"
        )
    return strong_id, cheap_id


def create_app(tau: float = DEFAULT_TAU) -> FastAPI:
    config = load_app_config()  # no API key required — never calls the provider
    strong_model_id, cheap_model_id = select_role_model_ids(config, provider="anthropic")
    policy = build_demo_router_policy(tau=tau)

    app = FastAPI(
        title="Adaptive LLM Cost Router — Demo",
        description=DEMO_DISCLAIMER,
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        return InfoResponse(
            disclaimer=DEMO_DISCLAIMER,
            strong_model_id=strong_model_id,
            cheap_model_id=cheap_model_id,
            tau=tau,
            router_name=policy.name,
        )

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        decision = policy.decide(prompt_id="demo", prompt_text=request.prompt)
        selected_model_id = strong_model_id if decision.selected_role == "strong" else cheap_model_id

        usage = estimate_prompt_tokens(request.prompt)
        cost = calculate_cost(
            usage=usage,
            model_id=selected_model_id,
            pricing=config.pricing,
            timestamp=datetime.now(UTC),
        )

        return RouteResponse(
            prompt=request.prompt,
            selected_role=decision.selected_role,
            probability=decision.probability if decision.probability is not None else 0.0,
            policy_name=decision.policy_name,
            selected_model_id=selected_model_id,
            estimated_cost_usd=cost.total_cost,
            tau=tau,
            disclaimer=DEMO_DISCLAIMER,
        )

    return app


app = create_app()
