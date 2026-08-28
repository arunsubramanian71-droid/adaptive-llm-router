from __future__ import annotations

from datetime import UTC, datetime

import pytest

from router.config import ConfigError, load_app_config
from router.cost.calculator import calculate_cost
from router.models.schemas import TokenUsage


def test_basic_cost_calculation(pricing_config):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    breakdown = calculate_cost(
        usage=usage,
        model_id="claude-sonnet-5",
        pricing=pricing_config,
        timestamp=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert breakdown.input_cost == pytest.approx(2.0)
    assert breakdown.output_cost == pytest.approx(10.0)
    assert breakdown.cache_write_cost == 0.0
    assert breakdown.cache_read_cost == 0.0
    assert breakdown.total_cost == pytest.approx(12.0)


def test_cost_picks_correct_rate_period_by_timestamp(pricing_config):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)

    before = calculate_cost(
        usage, "claude-sonnet-5", pricing_config, datetime(2026, 8, 1, tzinfo=UTC)
    )
    after = calculate_cost(
        usage, "claude-sonnet-5", pricing_config, datetime(2026, 9, 15, tzinfo=UTC)
    )
    assert before.input_cost == pytest.approx(2.0)  # intro rate
    assert after.input_cost == pytest.approx(3.0)  # standard rate


def test_none_usage_fields_contribute_zero_cost(pricing_config):
    usage = TokenUsage(input_tokens=None, output_tokens=None)
    breakdown = calculate_cost(
        usage, "claude-sonnet-5", pricing_config, datetime(2026, 6, 1, tzinfo=UTC)
    )
    assert breakdown.total_cost == 0.0


def test_reasoning_cost_zero_when_no_reasoning_rate_configured(pricing_config):
    # Anthropic pricing has no reasoning_per_mtok — even if usage reports
    # reasoning tokens (it never does today), cost must not be invented.
    usage = TokenUsage(input_tokens=0, output_tokens=0, reasoning_tokens=500)
    breakdown = calculate_cost(
        usage, "claude-sonnet-5", pricing_config, datetime(2026, 6, 1, tzinfo=UTC)
    )
    assert breakdown.reasoning_cost == 0.0


def test_cache_write_ttl_selects_correct_rate(pricing_config):
    usage = TokenUsage(cache_creation_input_tokens=1_000_000)
    ts = datetime(2026, 6, 1, tzinfo=UTC)

    five_min = calculate_cost(usage, "claude-sonnet-5", pricing_config, ts, cache_write_ttl="5m")
    one_hour = calculate_cost(usage, "claude-sonnet-5", pricing_config, ts, cache_write_ttl="1h")
    assert five_min.cache_write_cost == pytest.approx(2.5)
    assert one_hour.cache_write_cost == pytest.approx(4.0)


def test_unpriced_model_raises(pricing_config):
    usage = TokenUsage(input_tokens=1)
    with pytest.raises(ConfigError):
        calculate_cost(
            usage, "not-a-real-model", pricing_config, datetime(2026, 6, 1, tzinfo=UTC)
        )


# ---------------------------------------------------------------------------
# Requirement 12: at least one configured model from each of the three
# providers, using the real committed configs/pricing.yaml (not a synthetic
# fixture) so this actually exercises what a live run would bill.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_pricing():
    return load_app_config().pricing


def test_anthropic_model_cost_from_real_pricing_config(real_pricing):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    breakdown = calculate_cost(usage, "claude-haiku-4-5", real_pricing, datetime(2026, 6, 1, tzinfo=UTC))
    assert breakdown.input_cost == pytest.approx(1.00)
    assert breakdown.output_cost == pytest.approx(5.00)
    assert breakdown.reasoning_cost == 0.0  # no reasoning_per_mtok configured for Anthropic
    assert breakdown.total_cost == pytest.approx(6.00)


def test_openai_model_cost_from_real_pricing_config(real_pricing):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    breakdown = calculate_cost(usage, "gpt-5.6-luna", real_pricing, datetime(2026, 6, 1, tzinfo=UTC))
    assert breakdown.input_cost == pytest.approx(0.20)
    assert breakdown.output_cost == pytest.approx(1.20)
    assert breakdown.reasoning_cost == 0.0  # deliberately unset -- see ADR-0004 (already inside output_tokens)
    assert breakdown.total_cost == pytest.approx(1.40)


def test_openai_reasoning_tokens_do_not_double_bill(real_pricing):
    # Even if a response reports reasoning_tokens (a real, non-zero
    # sub-count of output_tokens for OpenAI), cost must not add a second
    # charge on top of output_cost.
    usage = TokenUsage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=400_000)
    breakdown = calculate_cost(usage, "gpt-5.6-sol", real_pricing, datetime(2026, 6, 1, tzinfo=UTC))
    assert breakdown.reasoning_cost == 0.0
    assert breakdown.total_cost == pytest.approx(breakdown.output_cost)


def test_google_model_cost_from_real_pricing_config(real_pricing):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    breakdown = calculate_cost(usage, "gemini-2.5-flash-lite", real_pricing, datetime(2026, 6, 1, tzinfo=UTC))
    assert breakdown.input_cost == pytest.approx(0.10)
    assert breakdown.output_cost == pytest.approx(0.40)
    assert breakdown.total_cost == pytest.approx(0.50)


def test_google_reasoning_tokens_are_billed_additively_at_output_rate(real_pricing):
    # Unlike OpenAI, Gemini's thought tokens are a genuinely separate,
    # additive count -- and configs/pricing.yaml prices them at the
    # standard output rate, so reasoning_cost must be non-zero here.
    usage = TokenUsage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=500_000)
    breakdown = calculate_cost(usage, "gemini-3.1-pro-preview", real_pricing, datetime(2026, 6, 1, tzinfo=UTC))
    assert breakdown.output_cost == pytest.approx(12.00)
    assert breakdown.reasoning_cost == pytest.approx(6.00)  # 0.5 MTok x $12/MTok
    assert breakdown.total_cost == pytest.approx(18.00)


def test_three_provider_models_have_different_costs_for_identical_usage(real_pricing):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    ts = datetime(2026, 6, 1, tzinfo=UTC)
    costs = {
        model_id: calculate_cost(usage, model_id, real_pricing, ts).total_cost
        for model_id in ("claude-haiku-4-5", "gpt-5.6-luna", "gemini-2.5-flash-lite")
    }
    assert len(set(costs.values())) == 3  # genuinely independent pricing per provider, not a shared default
