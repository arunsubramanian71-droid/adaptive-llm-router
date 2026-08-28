from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from router.config import ModelEntry, ModelPricing, PricingConfig, RatePeriod
from router.models.schemas import GenerationConfig, TokenUsage
from router.storage.records import ResponseRecord


@pytest.fixture
def sonnet_entry() -> ModelEntry:
    return ModelEntry(
        id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        role_hint="candidate_strong",
        max_output_tokens=1024,
        thinking={"type": "adaptive", "display": "omitted"},
        effort="high",
    )


@pytest.fixture
def haiku_entry() -> ModelEntry:
    return ModelEntry(
        id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        role_hint="candidate_cheap",
        max_output_tokens=1024,
    )


@pytest.fixture
def opus_entry() -> ModelEntry:
    return ModelEntry(
        id="claude-opus-5",
        display_name="Claude Opus 5",
        role_hint=None,
        max_output_tokens=1024,
        thinking={"type": "adaptive", "display": "omitted"},
        effort="high",
    )


@pytest.fixture
def pricing_config() -> PricingConfig:
    return PricingConfig(
        version="1",
        currency="USD",
        pricing_config_version="test-pricing-v1",
        models={
            "claude-sonnet-5": ModelPricing(
                provider="anthropic",
                rates=[
                    RatePeriod(
                        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
                        effective_until=datetime(2026, 9, 1, tzinfo=UTC),
                        input_per_mtok=2.0,
                        output_per_mtok=10.0,
                        cache_write_5m_per_mtok=2.5,
                        cache_write_1h_per_mtok=4.0,
                        cache_read_per_mtok=0.2,
                    ),
                    RatePeriod(
                        effective_from=datetime(2026, 9, 1, tzinfo=UTC),
                        effective_until=None,
                        input_per_mtok=3.0,
                        output_per_mtok=15.0,
                        cache_write_5m_per_mtok=3.75,
                        cache_write_1h_per_mtok=6.0,
                        cache_read_per_mtok=0.3,
                    ),
                ],
            ),
            "claude-haiku-4-5": ModelPricing(
                provider="anthropic",
                rates=[
                    RatePeriod(
                        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
                        effective_until=None,
                        input_per_mtok=1.0,
                        output_per_mtok=5.0,
                        cache_write_5m_per_mtok=1.25,
                        cache_write_1h_per_mtok=2.0,
                        cache_read_per_mtok=0.1,
                    ),
                ],
            ),
            "claude-opus-5": ModelPricing(
                provider="anthropic",
                rates=[
                    RatePeriod(
                        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
                        effective_until=None,
                        input_per_mtok=5.0,
                        output_per_mtok=25.0,
                        cache_write_5m_per_mtok=6.25,
                        cache_write_1h_per_mtok=10.0,
                        cache_read_per_mtok=0.5,
                    ),
                ],
            ),
        },
    )


@pytest.fixture
def make_response_record() -> Callable[..., ResponseRecord]:
    """Factory for synthetic ResponseRecords used across the evaluation /
    aggregation / policy / router test suites. Text and usage are made up
    for plumbing tests — never treat these as real model output."""

    def _make(
        record_id: str,
        prompt_id: str,
        model_id: str = "claude-haiku-4-5",
        sample_index: int = 0,
        response_text: str | None = "synthetic response",
        input_tokens: int = 20,
        output_tokens: int = 10,
        status: str = "ok",
        run_id: str = "run-test",
    ) -> ResponseRecord:
        return ResponseRecord(
            record_id=record_id,
            run_id=run_id,
            prompt_id=prompt_id,
            prompt_hash=f"hash-{prompt_id}",
            provider="anthropic",
            requested_model_id=model_id,
            served_model_id=model_id,
            sample_index=sample_index,
            generation_config=GenerationConfig(
                provider="anthropic", requested_model_id=model_id, max_output_tokens=1024
            ),
            status=status,
            response_text=response_text,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            latency_ms=100.0,
            timestamp_utc=datetime(2026, 6, 1, tzinfo=UTC),
        )

    return _make


_HARD_TEMPLATES = [
    "Prove that this algorithm terminates for every input.",
    "Derive a closed-form solution for this recurrence relation.",
    "Debug this multithreaded race condition in the scheduler.",
    "Optimize this dynamic programming solution for large inputs.",
    "Write a proof by induction for the following theorem.",
    "Refactor this recursive parser to avoid stack overflows.",
    "Find a counterexample that disproves this conjecture.",
    "Design an algorithm with better than quadratic time complexity.",
]
_EASY_TEMPLATES = [
    "What is the capital of France?",
    "Say hello in French.",
    "What color is a clear sky at noon?",
    "Convert 10 miles to kilometers.",
    "What day comes after Monday?",
    "Name three primary colors.",
    "How many days are in a leap year?",
    "What is the opposite of hot?",
]


@pytest.fixture
def synthetic_labeled_prompts() -> tuple[list[str], list[int]]:
    """Deterministic, clearly-separable synthetic prompts for exercising
    router fit/predict_proba and calibration — not real benchmark data."""
    prompts: list[str] = []
    labels: list[int] = []
    for i in range(6):
        for template in _HARD_TEMPLATES:
            prompts.append(f"{template} (variant {i})")
            labels.append(1)
        for template in _EASY_TEMPLATES:
            prompts.append(f"{template} (variant {i})")
            labels.append(0)
    return prompts, labels
