"""Cost calculation.

Rules (non-negotiable, see docs/decisions):
- Pricing always comes from `PricingConfig`, never a literal in this module.
- Cost is only ever computed from provider-reported usage
  (`NormalizedCompletion.usage`) — never estimated from a local tokenizer.
- A usage field the provider didn't report (`None`) contributes 0 cost; it
  is never guessed at.
"""

from __future__ import annotations

from datetime import datetime

from router.config import PricingConfig
from router.cost.schemas import CostBreakdown
from router.models.schemas import TokenUsage

# Anthropic's default cache write TTL is 5 minutes unless a request opts into
# the 1-hour TTL. Stage 0 does not yet set cache_control on requests, so
# usage.cache_creation_input_tokens (when present) is costed at the 5-minute
# rate. Revisit this if/when the pilot starts using 1h cache breakpoints.
DEFAULT_CACHE_WRITE_TTL = "5m"


def calculate_cost(
    usage: TokenUsage,
    model_id: str,
    pricing: PricingConfig,
    timestamp: datetime,
    cache_write_ttl: str = DEFAULT_CACHE_WRITE_TTL,
) -> CostBreakdown:
    model_pricing = pricing.pricing_for(model_id)
    rate = model_pricing.rate_for(timestamp)

    def cost(tokens: int | None, per_mtok: float | None) -> float:
        if tokens is None or per_mtok is None:
            return 0.0
        return (tokens / 1_000_000) * per_mtok

    cache_write_rate = (
        rate.cache_write_1h_per_mtok if cache_write_ttl == "1h" else rate.cache_write_5m_per_mtok
    )

    return CostBreakdown(
        model_id=model_id,
        pricing_config_version=pricing.pricing_config_version,
        currency=pricing.currency,
        input_cost=cost(usage.input_tokens, rate.input_per_mtok),
        output_cost=cost(usage.output_tokens, rate.output_per_mtok),
        cache_write_cost=cost(usage.cache_creation_input_tokens, cache_write_rate),
        cache_read_cost=cost(usage.cache_read_input_tokens, rate.cache_read_per_mtok),
        reasoning_cost=cost(usage.reasoning_tokens, rate.reasoning_per_mtok),
    )
